from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, PreTrainedTokenizerFast, AutoModel
from transformers.integrations import MLflowCallback
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import math
import os
import torch
from transformers import BitsAndBytesConfig


# loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    cache_dir="cache"
    )
tokenizer.add_special_tokens({'pad_token': "<|end_of_text|>"})
tokenizer.max_length = 8192
tokenizer.model_max_length = 8192


# quantization_config = BitsAndBytesConfig(load_in_8bit=True,
#                                        llm_int8_threshold=200.0)

model_id = "cp/test_3/final_2"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto", 
    # quantization_config=quantization_config,
    torch_dtype=torch.float16,
    cache_dir="cache"
    )
model.generation_config.eos_token_id = 128009

# prepare int-8 model for training
# model = prepare_model_for_kbit_training(model)


# Define LoRA Config
lora_config = LoraConfig(
 r=64,
 lora_alpha=16,
 target_modules=["q_proj", "k_proj", "v_proj"],
 lora_dropout=0.05,
 bias="none",
 task_type="CAUSAL_LM"
)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# loading dataset
dataset = load_from_disk("data/tools")

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir="cp/tools",
    eval_strategy="steps",
    eval_steps = 500,
    num_train_epochs=3,
    # max_steps=3000,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=16,
    learning_rate=4e-5,
    lr_scheduler_type="cosine",
    weight_decay=0.1,
    push_to_hub=False,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=5,
    fp16=True,
    load_best_model_at_end=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=data_collator,
)

trainer.train()


eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")