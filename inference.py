import torch
from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from random import randrange

base_model_path = "t3ai-org/pt-model"
peft_model_path = "cp/test/checkpoint-2398"
path_to_save_merged = "final_model"

model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map='auto', torch_dtype=torch.float16, cache_dir='cache')
model = PeftModel.from_pretrained(model, peft_model_path, device_map='auto')
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.add_special_tokens({'pad_token': "<|end_of_text|>"})
tokenizer.max_length = 8192

print("Peft model loaded")


# # Load dataset and get a sample
# dataset = load_from_disk("data/fine-tune/bactrian")

# sample = tokenizer.decode(dataset['validation'][randrange(len(dataset["validation"]))]['input_ids'], skip_special_tokens=True)
# input, output = sample.split('[/INST]')
# input += '[/INST]'

def preprocess_function(input):
    
    conversation = [
        {"role": "user", "content": input}
        ]
    
    text_input = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False)

    model_input = tokenizer(text_input,
                            max_length=8192,
                            padding=True,
                            truncation=True,
                            return_tensors='pt').to("cuda")
    
    return model_input

input = "Lisede hangi fenler geçiliyor?"
encoding = preprocess_function(input)

generation_config = model.generation_config
generation_config.do_sample = True
generation_config.max_new_tokens = 400
# generation_config.temperature = 0.7
# generation_config.top_p = 0.7
# generation_config.num_return_sequences = 1
# generation_config.pad_token_id = tokenizer.pad_token_id
# generation_config.eos_token_id = tokenizer.eos_token_id

with torch.inference_mode():
  outputs = model.generate(
      input_ids = encoding.input_ids,
      attention_mask = encoding.attention_mask,
      generation_config = generation_config,
  )


print(f"Predicted output:\n{tokenizer.decode(outputs[0], skip_special_tokens=False)}")