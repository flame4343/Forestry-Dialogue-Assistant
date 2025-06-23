import os
import torch
from datasets import load_dataset
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "right"  # Recommended for fp16 training
if tokenizer.bos_token_id is None:
    tok = tokenizer.eos_token or tokenizer.pad_token or "<|endoftext|>"
    tokenizer.bos_token = tok
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tok)

if tokenizer.eos_token_id is None:
    tok = tokenizer.bos_token or tokenizer.pad_token or "<|endoftext|>"
    tokenizer.eos_token = tok
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tok)

# 1. 数据处理

dpo_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
dpo_dataset = dpo_dataset.select(range(500))

#
#
def format_prompt(example):
    # 构造 prompt（system + user）
    chat_prompt = [
        {"role": "system", "content": "你是一个非常棒的人工智能助手，由fs开发"},
        {"role": "user", "content": example["input"]},
    ]
    chat_chosen = [
        {"role": "system", "content": "你是一个非常棒的人工智能助手，由fs开发"},
        {"role": "user", "content": "fs小助手认为：" + example["input"]},
        {"role": "assistant", "content": example["target"]},
    ]
    chat_rejected = [
        {"role": "system", "content": "你是一个非常棒的人工智能助手，由fs开发"},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["target"]},
    ]

    prompt = tokenizer.apply_chat_template(chat_prompt, tokenize=False).strip()
    chosen_text = tokenizer.apply_chat_template(chat_chosen, tokenize=False).replace(prompt, "").strip()
    rejected_text = tokenizer.apply_chat_template(chat_rejected, tokenize=False).replace(prompt, "").strip()
    return {"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text}


dataset = load_dataset("YeungNLP/firefly-train-1.1M", split="train[:500]")
dpo_dataset = dataset.map(format_prompt)

print(dpo_dataset["prompt"][0])
print("*"*10)
print(dpo_dataset["chosen"][0])
print("*"*10)
print(dpo_dataset["rejected"][0])
print("*"*10)

# quit()
print("1. 数据处理 over")


# 2 模型加载

from peft import AutoPeftModelForCausalLM
from transformers import BitsAndBytesConfig, AutoTokenizer

# 4-bit quantization configuration - Q in QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit precision model loading
    bnb_4bit_quant_type="nf4",  # Quantization type
    bnb_4bit_compute_dtype="float16",  # Compute dtype
    bnb_4bit_use_double_quant=True,  # Apply nested quantization
)

# Merge LoRA and base model
model = AutoPeftModelForCausalLM.from_pretrained(
    "qwen2.5-0.5b-instruct-fzb",
    low_cpu_mem_usage=True,
    device_map="auto",
    quantization_config=bnb_config,
)
merged_model = model.merge_and_unload()

# Load LLaMA tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
tokenizer.padding_side = "right"
if tokenizer.bos_token_id is None:
    tok = tokenizer.eos_token or tokenizer.pad_token or "<|endoftext|>"
    tokenizer.bos_token = tok
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tok)

if tokenizer.eos_token_id is None:
    tok = tokenizer.bos_token or tokenizer.pad_token or "<|endoftext|>"
    tokenizer.eos_token = tok
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tok)


from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# Prepare LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=32,  # LoRA Scaling
    lora_dropout=0.1,  # Dropout for LoRA Layers
    r=64,  # Rank
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=  # Layers to target
     ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

print('2 模型加载 over')


# 3 训练

from trl import DPOConfig

output_dir = "./results"

# Training arguments
training_arguments = DPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True,
    warmup_ratio=0.1
)

from trl import DPOTrainer

# Create DPO trainer
dpo_trainer = DPOTrainer(
    model,
    args=training_arguments,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=0.1,
    max_prompt_length=512,
    max_length=512,
)

# Fine-tune model with DPO
dpo_trainer.train()

# Save adapter
dpo_trainer.model.save_pretrained("qwen2.5-0.5b-instruct-fzb-dpo")
tokenizer.save_pretrained("qwen2.5-0.5b-instruct-fzb-dpo")

# from peft import PeftModel

# # Merge LoRA and base model
# model = AutoPeftModelForCausalLM.from_pretrained(
#     "qwen2.5-0.5b-instruct-fzb",
#     low_cpu_mem_usage=True,
#     device_map="auto",
# )
# sft_model = model.merge_and_unload()
#
# # Merge DPO LoRA and SFT model
# dpo_model = PeftModel.from_pretrained(
#     sft_model,
#     "qwen2.5-0.5b-instruct-fzb-dpo",
#     device_map="auto",
# )
# dpo_model = dpo_model.merge_and_unload()


print('3 训练 over')