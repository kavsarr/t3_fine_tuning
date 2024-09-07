from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import HF_READ_TOKEN
from huggingface_hub import login

login(HF_READ_TOKEN)

model_id = "t3ai-org/pt-model"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='cache').to("cuda")


generation_config = model.generation_config
generation_config.do_sample = True
generation_config.max_new_tokens = 400
# generation_config.temperature = 0.3
# generation_config.top_p = 0.7
# generation_config.num_return_sequences = 1


prompts = [
"American education system is the"
]

for prompt in prompts:
  encoding = tokenizer(prompt, return_tensors="pt").to("cuda")

  with torch.no_grad():
    outputs = model.generate(
        input_ids = encoding.input_ids,
        attention_mask = encoding.attention_mask,
        generation_config = generation_config
    )

  print(tokenizer.decode(outputs[0], skip_special_tokens=True))
  print()
