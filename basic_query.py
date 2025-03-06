#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_llama_preference(choice1, choice2):
    """
    Ask LLaMA 3 8B model to choose between two options and explain its preference.
    
    Args:
        choice1 (str): First option
        choice2 (str): Second option
        
    Returns:
        str: The model's response
    """
    # Load model and tokenizer from local path
    model_path = "../models/llama3-8b-instruct"
    
    print(f"Loading LLaMA 3 8B model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        device_map="auto"            # Automatically use available GPUs
    )
    
    # Create prompt for LLaMA 3 Instruct
    prompt = f"""<|im_start|>system
You are a helpful AI assistant that says either (1) if you prefer Option 1 or (2) if you prefer Option2. Then,  you give a 1 sentence explanation.
<|im_end|>
<|im_start|>user
Between these two options, which do you prefer and why?
Option 1: {choice1}
Option 2: {choice2}
<|im_end|>
<|im_start|>assistant
"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    print("Generating response...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode and return response
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    assistant_response = response.split("<|im_start|>assistant")[1].strip()
    
    return assistant_response

if __name__ == "__main__":
    # Example usage
    choice1 = "Chocolate ice cream"
    choice2 = "Vanilla ice cream"
    
    # You can replace these with command line arguments if desired
    import sys
    if len(sys.argv) >= 3:
        choice1 = sys.argv[1]
        choice2 = sys.argv[2]
    
    print(f"Asking LLaMA 3 8B to choose between: \n1. {choice1}\n2. {choice2}\n")
    response = get_llama_preference(choice1, choice2)
    print("\nLLaMA 3 8B Response:")
    print(response)