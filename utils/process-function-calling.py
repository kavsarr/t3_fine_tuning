from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Load dataset from the hub
dataset = load_dataset("MirakramAghalarov/Function_calls_Turkish", cache_dir='cache')
dataset = dataset['train']

dataset = dataset.train_test_split(0.03)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir='cache')
tokenizer.add_special_tokens({'pad_token': "<|end_of_text|>"})
tokenizer.max_length = 8192
tokenizer.model_max_length = 8192

def preprocess_function(sample):

    question = sample['question']

    prompt = """Aşağıdaki fonksiyonlar verildiğinde, lütfen verilen isteme en iyi şekilde cevap verecek bir fonksiyon çağrısı için uygun argümanlarla bir JSON döndürün.

{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather conditions for a specific location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., Shaoxing, China"
                }
            },
            "required": ["location"]
        }
    }
}

{
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time globally or for a specific location."
    }
}

{
    "type": "function",
    "function": {
        "name": "get_currency",
        "description": "Convert a value from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "from": {
                    "type": "string",
                    "description": "The currency code to convert from, e.g., TRY"
                },
                "to": {
                    "type": "string",
                    "description": "The currency code to convert to, e.g., EUR"
                },
                "value": {
                    "type": "number",
                    "description": "The amount to convert, e.g., 57000"
                }
            },
            "required": ["from", "to", "value"]
        }
    }
}"""
    
    conversation = [
        {"role": "user", "content": prompt+f"\nSoru: {question}"},
        {"role": "assistant", "content": sample['actions']}
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
dataset.save_to_disk("data/tools")