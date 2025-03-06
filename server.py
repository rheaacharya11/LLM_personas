from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load model and tokenizer
model_path = "/n/holylabs/LABS/dwork_lab/Lab/rheaacharya/models/Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

class RequestModel(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
def generate_text(request: RequestModel):
    inputs = tokenizer(request.prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(inputs, max_length=request.max_tokens, temperature=request.temperature)
    return {"response": tokenizer.decode(output[0], skip_special_tokens=True)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)