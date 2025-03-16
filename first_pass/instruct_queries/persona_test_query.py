#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

def get_llama_preference(choice1, choice2, persona=None):
    """
    Ask LLaMA 3 8B model to choose between two options and explain its preference.
    
    Args:
        choice1 (str): First option
        choice2 (str): Second option
        persona (str, optional): Persona to assign to the model
        
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
        device_map="auto"  # Automatically use available GPUs
    )
    
    # Create base system prompt
    base_system_prompt = "You are a helpful AI assistant"
    
    # Add persona if provided
    if persona:
        system_prompt = f"{base_system_prompt} with the persona of {persona} that says either (1) if you prefer Option 1 or (2) if you prefer Option 2. Then, you give a 1 sentence explanation."
    else:
        system_prompt = f"{base_system_prompt} that says either (1) if you prefer Option 1 or (2) if you prefer Option 2. Then, you give a 1 sentence explanation."
    
    # Create prompt for LLaMA 3 Instruct
    prompt = f"""<|im_start|>system
    {system_prompt}
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

    # Ensure correct input format for model.generate()
    input_ids = inputs["input_ids"]  # Extract just input_ids

    # Generate response
    print("Generating response...")
    with torch.no_grad():
        print("Inputs type:", type(inputs))
        print("Inputs keys:", inputs.keys())  # Should show ['input_ids', 'attention_mask']
        print("Input IDs Shape:", inputs["input_ids"].shape)

        output = model.generate(
            input_ids,  # Pass only input_ids, not the full dictionary
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

def load_persona():
    """
    Load persona from the first row of the persona_train1.parquet file
    
    Returns:
        str: Persona description
    """
    try:
        # Load the parquet file
        df = pd.read_parquet("data/persona_train1.parquet")
        
        # Get the first row's persona value
        if not df.empty and 'persona' in df.columns:
            persona = df.iloc[0]['persona']
            print(f"Loaded persona: {persona}")
            return persona
        else:
            print("No persona found in the parquet file")
            return None
    except Exception as e:
        print(f"Error loading persona: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    choice1 = "Chocolate ice cream"
    choice2 = "Vanilla ice cream"
    
    # You can replace these with command line arguments if desired
    import sys
    if len(sys.argv) >= 3:
        choice1 = sys.argv[1]
        choice2 = sys.argv[2]
    
    # Load persona from parquet file
    persona = load_persona()
    
    print(f"Asking LLaMA 3 8B (with persona: {persona}) to choose between: \n1. {choice1}\n2. {choice2}\n")
    response = get_llama_preference(choice1, choice2, persona)
    
    print("\nLLaMA 3 8B Response:")
    print(response)