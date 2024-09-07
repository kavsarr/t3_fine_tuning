from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Load dataset from the hub
dataset = load_dataset("kavsar/t3_final_corpus", download_mode='force_redownload')

dataset = concatenate_datasets([dataset[i] for i in dataset])


dataset = dataset.train_test_split(0.03)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir='cache')
tokenizer.add_special_tokens({'pad_token': "<|end_of_text|>"})
tokenizer.max_length = 8192
tokenizer.model_max_length = 8192

def preprocess_function(sample):
    
    conversation = [
        {"role": "user", "content": sample['questionText']},
        {"role": "assistant", "content": sample['answerText']}
        ]
    
    text_input = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=False,
        tokenize=False)

    model_input = tokenizer(text_input,
                            max_length=8192,
                            padding=True,
                            truncation=True)
    
    return model_input

for split in ["train", "test"]:
    dataset[split] = dataset[split].shuffle().map(preprocess_function, remove_columns=dataset[split].column_names)

dataset.set_format(type="torch", device='cuda', columns=["input_ids", "attention_mask"])
        
# save datasets to disk for later easy loading
dataset.save_to_disk("data/corpus_3")