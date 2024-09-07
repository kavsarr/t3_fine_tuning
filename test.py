from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk


model = AutoModel.from_pretrained("t3ai-org/pt-model", cache_dir="cache")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", 
                                          cache_dir='cache')
tokenizer.add_special_tokens({'pad_token': "<|end_of_text|>"})

data = load_from_disk('data/corpus_2')
print(tokenizer.decode(data['train'][0]['input_ids'], skip_special_tokens=False))


print(model)

# print(tokenizer.eos_token_id)
# print(tokenizer.bos_token_id)

# conversation = [
#     {"role": "user", "content":"Salam"},
#     {"role": "assistant", "content":"Salam. Sizə necə kömək edə bilərəm?"}
#     ]

# print(tokenizer.apply_chat_template(conversation, tokenize=True, padding=True))

# print(tokenizer(tokenizer.apply_chat_template(
#     conversation,
#     add_generation_prompt=False,
#     tokenize=False
# )))