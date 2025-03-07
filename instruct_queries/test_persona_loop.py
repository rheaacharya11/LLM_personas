#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum
from typing import List, Tuple, Optional, Union, Dict, Any
import pandas as pd
import csv
import os
from dataclasses import dataclass
from tqdm import tqdm
import time

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
    persona: str = None
    choice: int = None

def llama_query(
    prompt: List[Tuple[LlamaRole, str]],
    model: LlamaModel = LlamaModel.LLAMA3_8B,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    model_path_prefix: str = "../models/",
    cache: bool = False,
    persona: str = None,
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
        persona: The persona being used (for record-keeping)
        
    Returns:
        LlamaResponse object containing the model's response
    """
    # Construct full model path
    model_path = f"{model_path_prefix}{model.value}"
    
    # Load model and tokenizer
    print(f"Loading {model.value} model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set padding token to EOS token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_instance = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        device_map="auto"  # Automatically use available GPUs
    )
    
    # Format the prompt using Llama 3 special tokens
    formatted_prompt = format_llama3_prompt(prompt, tokenizer)
    
    # Tokenize input and ensure attention mask is properly set
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model_instance.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Set pad token ID properly if it's not already set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Generate response
    print("Generating response...")
    with torch.no_grad():
        output = model_instance.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id
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
    
    # Try to extract choice number (1 or 2)
    choice = None
    if "1" in assistant_response[:50] and "2" not in assistant_response[:50]:
        choice = 1
    elif "2" in assistant_response[:50] and "1" not in assistant_response[:50]:
        choice = 2
    
    return LlamaResponse(
        full_text=assistant_response,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        persona=persona,
        choice=choice
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
    force_binary: bool = True,
) -> LlamaResponse:
    """
    Ask Llama to choose between two options and explain preference.
    
    Args:
        choice1: First option
        choice2: Second option
        persona: Optional persona to assign to the model
        model: Which Llama model to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        force_binary: Whether to force a binary 1 or 2 choice
        
    Returns:
        LlamaResponse with the model's choice and explanation
    """
    # Create base system prompt
    base_system_prompt = "You are a helpful AI assistant"
    
    # Add persona if provided
    if persona:
        if force_binary:
            system_prompt = f"{base_system_prompt} with the persona of {persona}. You must respond with ONLY the digit 1 or 2 to indicate your preference. No explanation or other text."
        else:
            system_prompt = f"{base_system_prompt} with the persona of {persona} that says either (1) if you prefer Option 1 or (2) if you prefer Option 2. Then, you give a 1 sentence explanation."
    else:
        if force_binary:
            system_prompt = f"{base_system_prompt}. You must respond with ONLY the digit 1 or 2 to indicate your preference. No explanation or other text."
        else:
            system_prompt = f"{base_system_prompt} that says either (1) if you prefer Option 1 or (2) if you prefer Option 2. Then, you give a 1 sentence explanation."
    
    # Create the prompt list
    prompt_list = [
        (LlamaRole.SYSTEM, system_prompt),
        (LlamaRole.USER, f"Between these two options, which do you prefer?\nOption 1: {choice1}\nOption 2: {choice2}")
    ]
    
    # Make the query
    response = llama_query(
        prompt=prompt_list,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        persona=persona
    )
    
    return response

def load_personas(max_personas: int = None) -> List[str]:
    """
    Load all personas from the persona_train1.parquet file
    
    Args:
        max_personas: Maximum number of personas to load (None for all)
        
    Returns:
        List of persona descriptions
    """
    try:
        # Load the parquet file
        df = pd.read_parquet("data/persona_train1.parquet")
        
        # Get all persona values
        if 'persona' in df.columns:
            if max_personas:
                personas = df['persona'].tolist()[:max_personas]
            else:
                personas = df['persona'].tolist()
            
            print(f"Loaded {len(personas)} personas")
            return personas
        else:
            print("No persona column found in the parquet file")
            return []
    except Exception as e:
        print(f"Error loading personas: {e}")
        return []

def run_preference_study(
    choice1: str,
    choice2: str,
    output_file: str = "results/persona_preferences.csv",
    model: LlamaModel = LlamaModel.LLAMA3_8B,
    max_personas: int = None,
    temperature: float = 0.7,
    batch_size: int = 10,
    force_binary: bool = True,
):
    """
    Run a study asking multiple personas to choose between two options.
    Results are saved to a CSV file.
    
    Args:
        choice1: First option
        choice2: Second option
        output_file: CSV file to save results
        model: Which Llama model to use
        max_personas: Maximum number of personas to use (None for all)
        temperature: Sampling temperature
        batch_size: Number of personas to process before saving results
        force_binary: Whether to force a binary 1 or 2 choice
    """
    # Load all personas
    personas = load_personas(max_personas)
    if not personas:
        print("No personas found. Exiting.")
        return
    
    # Create results directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Create or check output file
    file_exists = os.path.isfile(output_file)
    
    # Open CSV file for results
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = ['persona', 'choice', 'choice_number', 'choice1', 'choice2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Process personas in batches
        results = []
        
        for i, persona in enumerate(tqdm(personas, desc="Processing personas")):
            print(f"\nProcessing persona {i+1}/{len(personas)}: {persona[:50]}...")
            
            try:
                # Query the model with this persona
                response = preference_query(
                    choice1=choice1,
                    choice2=choice2,
                    persona=persona,
                    model=model,
                    temperature=temperature,
                    force_binary=force_binary
                )
                
                # Determine which choice was selected
                choice_text = response.full_text.strip()
                choice_number = response.choice
                
                # If we couldn't detect the choice automatically, try to parse it
                if choice_number is None:
                    if "1" in choice_text[:10] and "2" not in choice_text[:10]:
                        choice_number = 1
                    elif "2" in choice_text[:10] and "1" not in choice_text[:10]:
                        choice_number = 2
                
                # Add result to batch
                result = {
                    'persona': persona,
                    'choice': choice_text,
                    'choice_number': choice_number,
                    'choice1': choice1,
                    'choice2': choice2
                }
                results.append(result)
                
                # Write batch to CSV if batch size reached
                if len(results) >= batch_size:
                    writer.writerows(results)
                    csvfile.flush()  # Ensure data is written to disk
                    results = []
                    print(f"Saved batch of {batch_size} results to {output_file}")
                
                # Small delay to prevent overloading
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing persona: {e}")
        
        # Write any remaining results
        if results:
            writer.writerows(results)
            print(f"Saved final {len(results)} results to {output_file}")
    
    print(f"Study complete. Results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a preference study with multiple personas")
    parser.add_argument("--choice1", default="Chocolate ice cream", help="First choice option")
    parser.add_argument("--choice2", default="Vanilla ice cream", help="Second choice option")
    parser.add_argument("--output", default="results/persona_preferences.csv", help="Output CSV file")
    parser.add_argument("--model", default="llama3-8b-instruct", help="Model to use")
    parser.add_argument("--max_personas", type=int, default=5, help="Maximum number of personas to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for saving results")
    
    args = parser.parse_args()
    
    # Use command line arguments
    print(f"Running preference study: {args.choice1} vs {args.choice2}")
    print(f"Using up to {args.max_personas} personas, saving to {args.output}")
    
    model_enum = LlamaModel.LLAMA3_8B
    if args.model == "llama3-70b-instruct":
        model_enum = LlamaModel.LLAMA3_70B
    
    run_preference_study(
        choice1=args.choice1,
        choice2=args.choice2,
        output_file=args.output,
        model=model_enum,
        max_personas=args.max_personas,
        temperature=args.temperature,
        batch_size=args.batch_size,
        force_binary=True
    )