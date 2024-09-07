import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import HF_WRITE_TOKEN

base_model_path = "cp/test_3/final_2"
peft_model_path = "cp/tools/checkpoint-84"
path_to_save_merged = "cp/tools/final"
hf_repo = "kavsar/t3_model_tools"

model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map='auto', torch_dtype=torch.float16, cache_dir='cache')
model = PeftModel.from_pretrained(model, peft_model_path, device_map='auto')
model = model.merge_and_unload()
model.generation_config.eos_token_id = 128009

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.add_special_tokens({'pad_token': "<|end_of_text|>"})
tokenizer.max_length = 8192
tokenizer.model_max_length = 8192

model.save_pretrained(path_to_save_merged)
tokenizer.save_pretrained(path_to_save_merged) 

print("saved")

model.push_to_hub(hf_repo, token=HF_WRITE_TOKEN)
tokenizer.push_to_hub(hf_repo, token=HF_WRITE_TOKEN) 