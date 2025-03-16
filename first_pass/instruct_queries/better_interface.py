#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum
from typing import List, Tuple, Optional, Union, Dict, Any
import pandas as pd
from dataclasses import dataclass

class LlamaRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class LlamaModel(Enum):
    LLAMA3_8B = "llama3-8b-instruct"
    LLAMA3_70B = "llama3-70b-instruct"

@dataclass
class LlamaResponse:
    full_text: str
    model: LlamaModel
    done: bool = True
    prompt: List[Tuple[LlamaRole, str]] = None
    max_tokens: int = None
    temperature: float = None

def llama_query(
    prompt: List[Tuple[LlamaRole, str]],
    model: LlamaModel = LlamaModel.LLAMA3_8B,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    model_path_prefix: str = "../models/",
    cache: bool = False,
) -> LlamaResponse:
    """
    Query a local Llama model using the special tokens format.
    
    Args:
        prompt: List of tuples with (role, content)
        model: Which Llama model to use
        max_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling (vs greedy decoding)
        model_path_prefix: Directory prefix for model path
        cache: Whether to use caching (not implemented yet)
        
    Returns:
        LlamaResponse object containing the model's response
    """
    # Construct full model path
    model_path = f"{model_path_prefix}{model.value}"
    
    # Load model and tokenizer
    print(f"Loading {model.value} model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_instance = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        device_map="auto"  # Automatically use available GPUs
    )
    
    # Format the prompt using Llama 3 special tokens
    formatted_prompt = format_llama3_prompt(prompt, tokenizer)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model_instance.device)
    input_ids = inputs["input_ids"]
    
    # Generate response
    print("Generating response...")
    with torch.no_grad():
        output = model_instance.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )
    
    # Decode the response
    response_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Extract just the assistant's final response
    if "<|start_header_id|>assistant<|end_header_id|>" in response_text:
        # New format with special tokens
        parts = response_text.split("<|start_header_id|>assistant<|end_header_id|>")
        assistant_response = parts[-1].split("<|eot_id|>")[0]
    elif "<|im_start|>assistant" in response_text:
        # Original format with im_start
        assistant_response = response_text.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in assistant_response:
            assistant_response = assistant_response.split("<|im_end|>")[0].strip()
    else:
        # Fallback - return everything after the prompt
        assistant_response = response_text[len(formatted_prompt):].strip()
    
    return LlamaResponse(
        full_text=assistant_response,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

def format_llama3_prompt(prompt: List[Tuple[LlamaRole, str]], tokenizer) -> str:
    """
    Format a list of (role, content) tuples into the Llama 3 prompt format
    using special tokens.
    
    Args:
        prompt: List of (role, content) tuples
        tokenizer: The Llama tokenizer (to check available special tokens)
        
    Returns:
        Formatted prompt string
    """
    # Check if tokenizer has new special tokens
    has_new_format = "<|begin_of_text|>" in tokenizer.get_vocab()
    
    if has_new_format:
        # Use Meta's new special tokens format
        formatted_prompt = "<|begin_of_text|>"
        
        for role, content in prompt:
            formatted_prompt += f"<|start_header_id|>{role.value}<|end_header_id|>{content}<|eot_id|>"
        
        # Remove the final eot_id from the last user message
        # and add the assistant header for the response
        if prompt and prompt[-1][0] == LlamaRole.USER:
            formatted_prompt = formatted_prompt[:-9]  # Remove last "<|eot_id|>"
            formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>"
    else:
        # Fall back to original im_start/im_end format
        formatted_prompt = ""
        
        for role, content in prompt:
            formatted_prompt += f"<|im_start|>{role.value}\n{content}\n<|im_end|>\n"
        
        # Add the assistant starter token
        formatted_prompt += "<|im_start|>assistant\n"
    
    return formatted_prompt

def preference_query(
    choice1: str, 
    choice2: str, 
    persona: Optional[str] = None,
    model: LlamaModel = LlamaModel.LLAMA3_8B,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Ask Llama to choose between two options and explain preference.
    
    Args:
        choice1: First option
        choice2: Second option
        persona: Optional persona to assign to the model
        model: Which Llama model to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        The model's choice and explanation
    """
    # Create base system prompt
    base_system_prompt = "You are a helpful AI assistant"
    
    # Add persona if provided
    if persona:
        system_prompt = f"{base_system_prompt} with the persona of {persona} that says either (1) if you prefer Option 1 or (2) if you prefer Option 2. Then, you give a 1 sentence explanation."
    else:
        system_prompt = f"{base_system_prompt} that says either (1) if you prefer Option 1 or (2) if you prefer Option 2. Then, you give a 1 sentence explanation."
    
    # Create the prompt list
    prompt_list = [
        (LlamaRole.SYSTEM, system_prompt),
        (LlamaRole.USER, f"Between these two options, which do you prefer and why?\nOption 1: {choice1}\nOption 2: {choice2}")
    ]
    
    # Make the query
    response = llama_query(
        prompt=prompt_list,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return response.full_text

def load_persona() -> Optional[str]:
    """
    Load persona from the first row of the persona_train1.parquet file
    Returns:
        str or None: Persona description
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
    # Example usage with command line arguments
    import sys
    
    if len(sys.argv) >= 3:
        choice1 = sys.argv[1]
        choice2 = sys.argv[2]
    else:
        # Default examples
        choice1 = "Chocolate ice cream"
        choice2 = "Vanilla ice cream"
    
    # Load persona from parquet file
    persona = load_persona()
    
    print(f"Asking LLaMa to choose between: \n1. {choice1}\n2. {choice2}\n")
    response_text = preference_query(choice1, choice2, persona)
    print("\nLLaMa Response:")
    print(response_text)