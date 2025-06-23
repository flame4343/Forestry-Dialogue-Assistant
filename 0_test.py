import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"



# 1. 加载 tokenizer 和 model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


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



# 3. 使用pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
generated = pipe(text, max_new_tokens=100,  # 控制最大新生成 token 数
                 temperature=0.7,           # 控制生成的随机性，越高越随机
                 top_p=0.9,                 # nucleus sampling，保留概率总和为 p 的 token
                 top_k=50,                  # 只在概率前 top_k 的 token 中采样
                 do_sample=True,            # 是否启用采样（否则是贪婪/确定性输出）
                 repetition_penalty=1.1     # 惩罚重复 token，防止啰嗦
                )[0]["generated_text"]
split_token = "<|im_start|>assistant\n"
if split_token in generated:
    answer = generated.split(split_token)[-1].strip()
else:
    answer = generated.strip()
print(answer)


# 4. 底层调用
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
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
