from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly."

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float32
)

class Req(BaseModel):
    prompt: str

@app.post("/chat")
def chat(req: Req):
    full_prompt = f"{SYSTEM_PROMPT}\nUser: {req.prompt}\nAssistant:"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.9,
        top_p=0.95
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": text}
