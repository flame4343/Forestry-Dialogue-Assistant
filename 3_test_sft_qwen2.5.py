import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import AutoPeftModelForCausalLM

# ✅ 模型路径（替换为训练保存路径）
model_path = r"qwen2.5-0.5b-instruct-fzb"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


model = AutoPeftModelForCausalLM.from_pretrained(
    "qwen2.5-0.5b-instruct-fzb",
    low_cpu_mem_usage=True,
    device_map="auto",
)
merged_model = model.merge_and_unload()

# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto",
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
# )

# 2. 加载text
messages = [
    {"role": "system", "content": "你是一个非常棒的人工智能助手，由fzb开发。"},
    {"role": "user", "content": "天气太热了，所以我今天没有学习一点。翻译成文言文："}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)
print("*"*10)

# 4. 底层调用
model_inputs = tokenizer([text], return_tensors="pt").to(merged_model.device)
generated_ids = merged_model.generate(
    **model_inputs,
    max_new_tokens=512,       # 最大生成 token 数
    temperature=0.7,          # 越高越随机（建议范围 0.7~1.0）
    top_p=0.9,                # nucleus sampling 保留累计概率为 p 的 token
    top_k=50,                 # 限定前 k 个 token 中采样
    do_sample=True,           # 是否使用采样；True = 随机，False = 贪婪解码
    repetition_penalty=1.1,   # 惩罚重复内容，1.0 为无惩罚
    eos_token_id=tokenizer.eos_token_id,  # 可选：提前停止
    pad_token_id=tokenizer.pad_token_id   # 防止警告（有些模型强制 pad）
)
generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids) ]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
