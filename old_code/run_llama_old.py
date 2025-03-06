from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set model path
model_path = "/n/holylabs/LABS/dwork_lab/Lab/rheaacharya/models/Llama-3-8B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", cache_dir=model_path)

# Assign model a role
system_prompt = "You are an AI assistant that provides detailed and insightful answers to users' questions."
user_prompt = "Between the colors blue and purple, which do you prefer? Please explain your choice."
prompt = system_prompt + "\nUser: " + user_prompt + "\nAssistant: "

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=100)
input_ids = inputs.input_ids.to("cuda")
attention_mask = inputs.attention_mask.to("cuda")

# Generate output
output = model.generate(input_ids, attention_mask=attention_mask, max_length=200, temperature=0.7, top_p=0.9, do_sample=True, repetition_penalty=1.2)

# Decode and print output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print("LLaMA Response:", decoded_output)