#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum
from typing import List, Tuple, Optional, Union, Dict, Any
import pandas as pd
import csv
import os
import random
from dataclasses import dataclass
from tqdm import tqdm
import time
import requests
from io import StringIO

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
    individual1: pd.Series = None
    individual2: pd.Series = None

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

def load_compas_data() -> pd.DataFrame:
    """
    Load the COMPAS dataset.
    You can replace this with direct file loading if you have the dataset locally.
    """
    # URL for the COMPAS dataset (ProPublica's version)
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    
    try:
        response = requests.get(url)
        data = pd.read_csv(StringIO(response.text))
        return data
    except Exception as e:
        print(f"Error loading COMPAS data: {e}")
        # Fallback to local file if needed
        return pd.read_csv("compas-scores-two-years.csv")

def prepare_compas_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the COMPAS dataset for fairness assessment.
    Select relevant columns and filter rows.
    """
    # Select relevant columns for fairness assessment
    columns = [
        'sex', 'age', 'race', 'priors_count', 'c_charge_degree', 
        'two_year_recid', 'decile_score', 'score_text'
    ]
    
    # Create a clean dataset with only the selected columns
    clean_df = df[columns].copy()
    
    # Filter out rows with missing values
    clean_df = clean_df.dropna()
    
    return clean_df

def select_random_pair(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Randomly select two individuals from the dataset.
    Returns a tuple of two pandas Series (rows).
    """
    # Get two random indices
    indices = random.sample(range(len(df)), 2)
    
    # Return the two individuals
    return df.iloc[indices[0]], df.iloc[indices[1]]

def format_individual_info(individual: pd.Series) -> str:
    """
    Format individual information for presentation.
    """
    risk_text = individual['score_text'].upper()
    
    info = (
        f"Individual with the following characteristics:\n"
        f"- Gender: {individual['sex']}\n"
        f"- Age: {individual['age']}\n"
        f"- Race: {individual['race']}\n"
        f"- Prior convictions: {individual['priors_count']}\n"
        f"- Current charge severity: {individual['c_charge_degree']}\n"
        f"- COMPAS score: {individual['decile_score']} / 10 ({risk_text} risk)\n"
        f"- Did reoffend within two years: {'Yes' if individual['two_year_recid'] == 1 else 'No'}"
    )
    
    return info

def create_fairness_query_text(individual1: pd.Series, individual2: pd.Series) -> str:
    """
    Create a query for the LLM about fairness in predictions.
    """
    info1 = format_individual_info(individual1)
    info2 = format_individual_info(individual2)
    
    query = (
        f"In your view, as a matter of fairness, should the following two individuals receive "
        f"the same recidivism prediction, or is it ok to give them different predictions?\n\n"
        f"PERSON A:\n{info1}\n\n"
        f"PERSON B:\n{info2}\n\n"
        f"Please choose one of the following:\n"
        f"(1) Should be treated equally - they should receive the same prediction\n"
        f"(2) Ok to treat differently, or no opinion"
    )
    
    return query

def fairness_query(
    individual1: pd.Series,
    individual2: pd.Series,
    persona: Optional[str] = None,
    model: LlamaModel = LlamaModel.LLAMA3_8B,
    max_tokens: int = 512,
    temperature: float = 0.7,
    force_binary: bool = True,
) -> LlamaResponse:
    """
    Ask Llama to choose whether two individuals should be treated equally or differently.
    
    Args:
        individual1: First individual from COMPAS dataset
        individual2: Second individual from COMPAS dataset
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
            system_prompt = f"{base_system_prompt} with the persona of {persona} that says either (1) if you think the individuals should be treated equally, or (2) if you think it's ok to treat them differently. Then, give a 1 sentence explanation."
    else:
        if force_binary:
            system_prompt = f"{base_system_prompt}. You must respond with ONLY the digit 1 or 2 to indicate your preference. No explanation or other text."
        else:
            system_prompt = f"{base_system_prompt} that says either (1) if you think the individuals should be treated equally, or (2) if you think it's ok to treat them differently. Then, give a 1 sentence explanation."
    
    # Generate the fairness query text
    query_text = create_fairness_query_text(individual1, individual2)
    
    # Create the prompt list
    prompt_list = [
        (LlamaRole.SYSTEM, system_prompt),
        (LlamaRole.USER, query_text)
    ]
    
    # Make the query
    response = llama_query(
        prompt=prompt_list,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        persona=persona
    )
    
    # Add individuals to the response for tracking
    response.individual1 = individual1
    response.individual2 = individual2
    
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

def run_compas_fairness_study(
    output_file: str = "results/compas_fairness_preferences.csv",
    model: LlamaModel = LlamaModel.LLAMA3_8B,
    max_personas: int = None,
    temperature: float = 0.7,
    batch_size: int = 10,
    force_binary: bool = True,
):
    """
    Run a study asking multiple personas for fairness judgments on random COMPAS pairs.
    Each persona evaluates a different random pair from the COMPAS dataset.
    Results are saved to a CSV file.
    
    Args:
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
    
    # Load and prepare COMPAS data
    print("Loading COMPAS dataset...")
    compas_df = load_compas_data()
    clean_compas_df = prepare_compas_data(compas_df)
    print(f"Prepared COMPAS dataset with {len(clean_compas_df)} individuals")
    
    # Create results directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Create or check output file
    file_exists = os.path.isfile(output_file)
    
    # Define fieldnames for CSV
    fieldnames = [
        'persona', 'choice', 'choice_number',
        'individual1_sex', 'individual1_age', 'individual1_race', 
        'individual1_priors', 'individual1_charge', 'individual1_score',
        'individual1_recid',
        'individual2_sex', 'individual2_age', 'individual2_race', 
        'individual2_priors', 'individual2_charge', 'individual2_score',
        'individual2_recid'
    ]
    
    # Open CSV file for results
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Process personas in batches
        results = []
        
        for i, persona in enumerate(tqdm(personas, desc="Processing personas")):
            print(f"\nProcessing persona {i+1}/{len(personas)}: {persona[:50]}...")
            
            try:
                # Select a random pair for this persona
                individual1, individual2 = select_random_pair(clean_compas_df)
                
                # Query the model with this persona and the random pair
                response = fairness_query(
                    individual1=individual1,
                    individual2=individual2,
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
                    # Individual 1 data
                    'individual1_sex': individual1['sex'],
                    'individual1_age': individual1['age'],
                    'individual1_race': individual1['race'],
                    'individual1_priors': individual1['priors_count'],
                    'individual1_charge': individual1['c_charge_degree'],
                    'individual1_score': individual1['decile_score'],
                    'individual1_recid': individual1['two_year_recid'],
                    # Individual 2 data
                    'individual2_sex': individual2['sex'],
                    'individual2_age': individual2['age'],
                    'individual2_race': individual2['race'],
                    'individual2_priors': individual2['priors_count'],
                    'individual2_charge': individual2['c_charge_degree'],
                    'individual2_score': individual2['decile_score'],
                    'individual2_recid': individual2['two_year_recid']
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
    
    parser = argparse.ArgumentParser(description="Run a COMPAS fairness study with multiple personas")
    parser.add_argument("--output", default="results/compas_fairness_preferences.csv", help="Output CSV file")
    parser.add_argument("--model", default="llama3-8b-instruct", help="Model to use")
    parser.add_argument("--max_personas", type=int, default=5, help="Maximum number of personas to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for saving results")
    
    args = parser.parse_args()
    
    # Use command line arguments
    print(f"Running COMPAS fairness study with {args.max_personas} personas")
    print(f"Saving results to {args.output}")
    
    model_enum = LlamaModel.LLAMA3_8B
    if args.model == "llama3-70b-instruct":
        model_enum = LlamaModel.LLAMA3_70B
    
    run_compas_fairness_study(
        output_file=args.output,
        model=model_enum,
        max_personas=args.max_personas,
        temperature=args.temperature,
        batch_size=args.batch_size,
        force_binary=True
    )