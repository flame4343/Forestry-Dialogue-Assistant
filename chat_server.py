from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Literal, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModel
import torch, uvicorn, webbrowser, threading, time

# 模型路径
INSTRUCT_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_PATH = "qwen2.5-0.5b-instruct-fzb"
DPO_PATH = "qwen2.5-0.5b-instruct-fzb-dpo"

app = FastAPI()
loaded_models = {}
tokenizer = AutoTokenizer.from_pretrained(DPO_PATH, trust_remote_code=True)
tokenizer.padding_side = "right"

def load_model(model_type: str):
    if model_type in loaded_models:
        return loaded_models[model_type]
    if model_type == "instruct":
        model = AutoModelForCausalLM.from_pretrained(INSTRUCT_PATH, device_map="auto", trust_remote_code=True)
    elif model_type == "sft":
        model = AutoPeftModelForCausalLM.from_pretrained(SFT_PATH, device_map="auto").merge_and_unload()
    elif model_type == "dpo":
        base = AutoPeftModelForCausalLM.from_pretrained(SFT_PATH, device_map="auto").merge_and_unload()
        model = PeftModel.from_pretrained(base, DPO_PATH, device_map="auto").merge_and_unload()
    else:
        raise ValueError("Invalid model_type")
    loaded_models[model_type] = model
    return model

class Request(BaseModel):
    prompt: str
    model_type: Literal["instruct", "sft", "dpo"] = "dpo"
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50

@app.post("/generate")
def generate_text(req: Request):
    model = load_model(req.model_type)
    messages = [
        {"role": "system", "content": "你是一个非常棒的人工智能助手，由fzb开发。"},
        {"role": "user", "content": req.prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        do_sample=True,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated = outputs[0][inputs.input_ids.shape[-1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    return {"response": decoded}

@app.get("/web", response_class=HTMLResponse)
def serve_web_ui():
    with open("web_ui.html", "r", encoding="utf-8") as f:
        return f.read()

def open_browser():
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000/web")

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    uvicorn.run("chat_server:app", host="127.0.0.1", port=8000, reload=False)
