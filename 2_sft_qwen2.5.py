import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import pipeline
import torch

# === Step 1: Load and format dataset ===
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "right"  # Recommended for fp16 training


# 1 数据处理
def format_prompt(example):
    chat = [
        {"role": "system", "content": "你是一个非常棒的人工智能助手，由fzb开发"},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": "fzb小助手认为" + example["target"]}
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": prompt}

dataset = load_dataset("YeungNLP/firefly-train-1.1M", split="train[:2000]")
dataset = dataset.map(format_prompt)
print(dataset["text"][0])
quit()
print('1 数据处理 over')


# 2 模型加载
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# === Step 3: Apply PEFT / LoRA ===
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'v_proj', 'q_proj']
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
print('2 模型加载 over')

# 3 训练
from transformers import TrainingArguments

output_dir = "./results"

# Training arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True
)


from trl import SFTTrainer

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=512,
    peft_config=peft_config,
)
trainer.train()
#
# sft_config = SFTConfig(
#     output_dir="./results1",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     optim="adamw_torch",
#     learning_rate=2e-4,
#     lr_scheduler_type="cosine",
#     num_train_epochs=1,
#     logging_steps=10,
#     fp16=True,
#     gradient_checkpointing=True,
#     max_seq_length=512,
#     dataset_text_field="text",
#     # tokenizer=model_name,  # Safe way to ensure tokenizer is loaded correctly
# )
#
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     args=sft_config,
#     peft_config=peft_config,
#     # processing_class=tokenizer,  # Avoid deprecated tokenizer param
# )
#
# trainer.train()

# === Step 5: Save & Merge LoRA weights ===
trainer.model.save_pretrained("qwen2.5-0.5b-instruct-fzb")
tokenizer.save_pretrained("qwen2.5-0.5b-instruct-fzb")

# from peft import AutoPeftModelForCausalLM
# model = AutoPeftModelForCausalLM.from_pretrained(
#     "qwen2.5-0.5b-instruct-fzb",
#     low_cpu_mem_usage=True,
#     device_map="auto",
# )
# merged_model = model.merge_and_unload()



