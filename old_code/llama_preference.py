#!/usr/bin/env python3
"""
Script to run LLaMa 3 8B with a persona prompt and get preference between two options.
For use on Harvard FASRC Cluster.
"""

import argparse
import json
import torch
import typing
import os
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

def print_with_timestamp(message: str) -> None:
    """Print a message with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_timestamp_str() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_model(
    model_name: str, 
    torch_device: typing.Literal["auto", "cpu", "cuda"] = "auto",
    cuda_device_id: typing.Optional[int] = None
) -> typing.Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer."""
    print_with_timestamp(f"Loading model {model_name}...")
    
    # Handle device placement
    if torch_device == "auto":
        device_map = "auto"
        if torch.cuda.is_available():
            print_with_timestamp(f"CUDA available, using automatic device mapping")
        else:
            print_with_timestamp("CUDA not available, falling back to CPU")
    elif torch_device == "cuda":
        if not torch.cuda.is_available():
            print_with_timestamp("CUDA requested but not available, falling back to CPU")
            device_map = "cpu"
        elif cuda_device_id is not None:
            print_with_timestamp(f"Using CUDA device {cuda_device_id}")
            device_map = f"cuda:{cuda_device_id}"
        else:
            print_with_timestamp("Using CUDA with automatic device mapping")
            device_map = "auto"
    else:  # cpu
        print_with_timestamp("Using CPU as requested")
        device_map = "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with appropriate dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map
    )
    
    return model, tokenizer

def format_prompt(persona: str, option1: str, option2: str) -> str:
    """Format the prompt with the persona and options."""
    system_prompt = f"You are an AI assistant with the following persona: {persona}. Answer as if you genuinely have this personality and perspective."
    
    user_prompt = f"Based on your personality and preferences, which do you prefer: {option1} or {option2}? Explain your choice in a brief paragraph. Be decisive in your preference."
    
    # Format for Llama 3
    full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>"
    
    return full_prompt

def generate_preference(
    model,
    tokenizer,
    persona: str,
    option1: str,
    option2: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    include_logprobs: bool = False,
    top_k: int = 50
) -> dict:
    """Generate preference response from model and return results."""
    # Prepare the prompt
    prompt = format_prompt(persona, option1, option2)
    
    # Tokenize the prompt
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Record metadata
    metadata = {
        "model": model.config._name_or_path,
        "persona": persona,
        "option1": option1,
        "option2": option2,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "timestamp": get_timestamp_str()
    }
    
    # Generate response
    print_with_timestamp("Generating response...")
    
    with torch.no_grad():
        if include_logprobs:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True
            )
            
            # Process token probabilities if requested
            token_probs = []
            for i, scores in enumerate(outputs.scores):
                # Get probabilities
                probs = torch.nn.functional.softmax(scores, dim=-1)
                # Get top-k indices and values
                topk_probs, topk_indices = torch.topk(probs[0], top_k)
                # Convert to tokens
                topk_tokens = [tokenizer.decode([idx.item()]) for idx in topk_indices]
                # Add to list
                token_probs.append({
                    "tokens": topk_tokens,
                    "probs": topk_probs.tolist()
                })
            
            sequences = outputs.sequences
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
            sequences = outputs
            token_probs = None
    
    # Decode and get the response
    response = tokenizer.decode(sequences[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    assistant_response = response.split("<|assistant|>")[-1].strip()
    
    # Prepare result dictionary
    result = {
        "prompt": prompt,
        "full_response": response,
        "assistant_response": assistant_response,
        "metadata": metadata
    }
    
    if include_logprobs:
        result["token_probabilities"] = token_probs
    
    return result

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLaMa 3 with persona prompts for preference selection"
    )
    parser.add_argument(
        "--persona", 
        type=str, 
        required=True, 
        help="Persona description for the model"
    )
    parser.add_argument(
        "--option1", 
        type=str, 
        required=True, 
        help="First option to choose from"
    )
    parser.add_argument(
        "--option2", 
        type=str, 
        required=True, 
        help="Second option to choose from"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="/n/holylabs/LABS/dwork_lab/Lab/rheaacharya/models/Llama-3-8B/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920",
        help="Name or path of the model to use"
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        help="Device to run on",
        default="auto",
        choices=["cuda", "cpu", "auto"]
    )
    parser.add_argument(
        "--cuda-device-id",
        type=int,
        help="CUDA device ID to use",
        default=None
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Maximum number of tokens to generate",
        default=512
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature",
        default=0.7
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p (nucleus) sampling parameter",
        default=0.9
    )
    parser.add_argument(
        "--include-logprobs",
        help="Include token log probabilities in output",
        action="store_true"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Number of top tokens to include in logprobs",
        default=50
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="JSON file to write results to",
        default=None
    )
    
    args = parser.parse_args()
    
    # Extract parameters
    persona: str = args.persona
    option1: str = args.option1
    option2: str = args.option2
    model_name: str = args.model_name
    torch_device: typing.Literal["auto", "cpu", "cuda"] = args.torch_device
    cuda_device_id: typing.Optional[int] = args.cuda_device_id
    max_new_tokens: int = args.max_new_tokens
    temperature: float = args.temperature
    top_p: float = args.top_p
    include_logprobs: bool = args.include_logprobs
    top_k: int = args.top_k
    outfile: typing.Optional[str] = args.outfile
    
    if cuda_device_id is not None and torch_device != "cuda":
        print_with_timestamp("Warning: cuda_device_id is only used when torch_device is 'cuda'")
    
    # Display run configuration
    print_with_timestamp(f"Running with model: {model_name}")
    print_with_timestamp(f"Persona: {persona}")
    print_with_timestamp(f"Option 1: {option1}")
    print_with_timestamp(f"Option 2: {option2}")
    
    # Load model and tokenizer
    model, tokenizer = load_model(model_name, torch_device, cuda_device_id)
    
    # Generate preference
    result = generate_preference(
        model,
        tokenizer,
        persona,
        option1,
        option2,
        max_new_tokens,
        temperature,
        top_p,
        include_logprobs,
        top_k
    )
    
    # Display results
    print_with_timestamp("Generation complete")
    print("\nResults:")
    print("=" * 80)
    print(result["assistant_response"])
    print("=" * 80)
    
    # Save results if outfile specified
    if outfile:
        output_dir = os.path.dirname(outfile)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print_with_timestamp(f"Writing results to {outfile}")
        with open(outfile, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()