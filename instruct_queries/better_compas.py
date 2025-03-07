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
    choice: str = None
    individual1: pd.Series = None
    individual2: pd.Series = None

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

def llama_query(
    prompt: List[Tuple[LlamaRole, str]],
    model_instance,
    tokenizer,
    model: LlamaModel = LlamaModel.LLAMA3_8B,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    persona: str = None,
) -> LlamaResponse:
    """
    Query a Llama model using the special tokens format.
    
    Args:
        prompt: List of tuples with (role, content)
        model_instance: Pre-loaded model instance
        tokenizer: Pre-loaded tokenizer
        model: Which Llama model type is being used
        max_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling (vs greedy decoding)
        persona: The persona being used (for record-keeping)
        
    Returns:
        LlamaResponse object containing the model's response
    """    
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
    
    # Try to extract response type
    choice = None
    if "Should be treated equally" in assistant_response:
        choice = "equal"
    elif "Ok to treat differently" in assistant_response:
        choice = "different"
    
    return LlamaResponse(
        full_text=assistant_response,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        persona=persona,
        choice=choice
    )

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

# This function will convert charge degree codes to full words
def convert_charge_degree(charge_degree):
    """
    Convert charge degree codes to full words.
    F -> Felony, M -> Misdemeanor
    """
    if isinstance(charge_degree, str):
        if charge_degree.startswith('F'):
            degree = charge_degree[1:] if len(charge_degree) > 1 else ""
            return f"Felony{degree}"
        elif charge_degree.startswith('M'):
            degree = charge_degree[1:] if len(charge_degree) > 1 else ""
            return f"Misdemeanor{degree}"
    return charge_degree


# Update the prepare_compas_data function to add a new column with full charge descriptions
def prepare_compas_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the COMPAS dataset for fairness assessment.
    Select relevant columns and filter rows.
    """
    print(f"Original dataset: {len(df)} rows")
    
    # Select relevant columns for fairness assessment
    columns = [
        'id', 'sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count', 
        'juv_other_count', 'priors_count', 'c_charge_degree'
    ]
    
    # Create a dataset with only the selected columns
    subset_df = df[columns].copy()
    print(f"After selecting columns: {len(subset_df)} rows")
    
    # Check for and report missing values
    missing_values = subset_df.isnull().sum()
    print("Missing values per column:")
    for col in columns:
        print(f"  - {col}: {missing_values[col]}")
    
    # Filter out rows with missing values
    clean_df = subset_df.dropna()
    print(f"After removing rows with missing values: {len(clean_df)} rows")
    
    # Remove rows with invalid charge degrees (empty strings)
    clean_df = clean_df[clean_df['c_charge_degree'].notna() & (clean_df['c_charge_degree'] != '')]
    print(f"After removing rows with invalid charge degrees: {len(clean_df)} rows")
    
    # Remove rows with negative counts
    count_cols = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
    initial_count = len(clean_df)
    for col in count_cols:
        clean_df = clean_df[clean_df[col] >= 0]
    print(f"After removing rows with negative counts: {len(clean_df)} rows (removed {initial_count - len(clean_df)})")
    
    # Convert numerical columns to proper types
    try:
        clean_df['age'] = clean_df['age'].astype(int)
        clean_df['juv_fel_count'] = clean_df['juv_fel_count'].astype(int)
        clean_df['juv_misd_count'] = clean_df['juv_misd_count'].astype(int)
        clean_df['juv_other_count'] = clean_df['juv_other_count'].astype(int)
        clean_df['priors_count'] = clean_df['priors_count'].astype(int)
    except Exception as e:
        print(f"Warning during type conversion: {e}")
    
    # Add a column with full charge descriptions
    clean_df['c_charge_degree_full'] = clean_df['c_charge_degree'].apply(convert_charge_degree)
    
    # Print a summary
    print(f"\nCleaning Summary:")
    print(f"  - Original dataset: {len(df)} rows")
    print(f"  - Cleaned dataset: {len(clean_df)} rows")
    print(f"  - Removed: {len(df) - len(clean_df)} rows ({((len(df) - len(clean_df)) / len(df) * 100):.1f}%)")
    
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

def create_fairness_query_text(individual1: pd.Series, individual2: pd.Series) -> str:
    """
    Create a query for the LLM about fairness in predictions, using a side-by-side list format
    that's easy for LLMs to parse.
    """
    # Create a comparison using side-by-side lists with clear labels
    comparison = f"""
PERSON A:
- Gender: {individual1['sex']}
- Age: {individual1['age']}
- Race: {individual1['race']}
- Juvenile felonies: {individual1['juv_fel_count']}
- Juvenile misdemeanors: {individual1['juv_misd_count']}
- Juvenile other: {individual1['juv_other_count']}
- Prior convictions: {individual1['priors_count']}
- Current charge severity: {individual1['c_charge_degree_full']}

PERSON B:
- Gender: {individual2['sex']}
- Age: {individual2['age']}
- Race: {individual2['race']}
- Juvenile felonies: {individual2['juv_fel_count']}
- Juvenile misdemeanors: {individual2['juv_misd_count']}
- Juvenile other: {individual2['juv_other_count']}
- Prior convictions: {individual2['priors_count']}
- Current charge severity: {individual2['c_charge_degree_full']}
"""
    
    query = (
        f"In your view, as a matter of fairness, should the following two individuals receive "
        f"the same recidivism prediction, or is it ok to give them different predictions?\n\n"
        f"{comparison}\n\n"
        f"Please respond with ONLY one of these two exact phrases:\n"
        f"\"Should be treated equally\"\n"
        f"\"Ok to treat differently, or no opinion\""
    )
    
    return query


def fairness_query(
    individual1: pd.Series,
    individual2: pd.Series,
    model_instance,
    tokenizer,
    persona: Optional[str] = None,
    persona_id: Optional[int] = None,
    model: LlamaModel = LlamaModel.LLAMA3_8B,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> LlamaResponse:
    """
    Ask Llama to choose whether two individuals should be treated equally or differently.
    
    Args:
        individual1: First individual from COMPAS dataset
        individual2: Second individual from COMPAS dataset
        model_instance: Pre-loaded model instance
        tokenizer: Pre-loaded tokenizer
        persona: Optional persona to assign to the model
        persona_id: Optional persona ID number
        model: Which Llama model to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        LlamaResponse with the model's choice and explanation
    """
    # Create base system prompt
    base_system_prompt = "You are a helpful AI assistant"
    
    # Add persona if provided
    if persona:
        system_prompt = f"{base_system_prompt} with the persona of {persona}. You must respond with ONLY ONE of these two exact phrases: \"Should be treated equally\" or \"Ok to treat differently, or no opinion\". Do not include any explanation or other text."
    else:
        system_prompt = f"{base_system_prompt}. You must respond with ONLY ONE of these two exact phrases: \"Should be treated equally\" or \"Ok to treat differently, or no opinion\". Do not include any explanation or other text."
    
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
        model_instance=model_instance,
        tokenizer=tokenizer,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        persona=persona
    )
    
    # Add individuals to the response for tracking
    response.individual1 = individual1
    response.individual2 = individual2
    
    return response

def load_selected_personas(start_indices=[1, 101, 201, 301, 401], count_per_index=1) -> List[Tuple[int, str]]:
    """
    Load personas from specific indices in the persona_train1.parquet file
    
    Args:
        start_indices: List of starting row indices to load personas from
        count_per_index: How many consecutive personas to load from each starting index
        
    Returns:
        List of tuples containing (persona_id, persona_description)
    """
    try:
        # Load the parquet file
        df = pd.read_parquet("data/persona_train1.parquet")
        
        # Get personas from specific indices
        if 'persona' in df.columns:
            # Reset index to ensure we have 0-based indices
            df = df.reset_index(drop=True)
            
            selected_personas = []
            for start_idx in start_indices:
                end_idx = min(start_idx + count_per_index, len(df))
                for idx in range(start_idx, end_idx):
                    if idx < len(df):
                        selected_personas.append((idx, df.loc[idx, 'persona']))
            
            print(f"Loaded {len(selected_personas)} personas from specified indices")
            return selected_personas
        else:
            print("No persona column found in the parquet file")
            return []
    except Exception as e:
        print(f"Error loading personas: {e}")
        return []

def load_unique_personas(max_personas=20) -> List[Tuple[int, str]]:
    """
    Load unique personas from the persona_train1.parquet file
    
    Args:
        max_personas: Maximum number of unique personas to load
        
    Returns:
        List of tuples containing (persona_id, persona_description)
    """
    try:
        # Load the parquet file
        df = pd.read_parquet("data/persona_train1.parquet")
        
        if 'persona' in df.columns:
            # Reset index to ensure we have 0-based indices
            df = df.reset_index(drop=True)
            
            # Get unique personas
            unique_personas = df['persona'].drop_duplicates().reset_index(drop=True)
            print(f"Found {len(unique_personas)} unique personas")
            
            # Limit to requested number
            selected_personas = []
            for idx, persona in enumerate(unique_personas[:max_personas]):
                selected_personas.append((idx, persona))
            
            print(f"Selected {len(selected_personas)} unique personas")
            return selected_personas
        else:
            print("No persona column found in the parquet file")
            return []
    except Exception as e:
        print(f"Error loading personas: {e}")
        return []

def run_compas_fairness_study(
    output_file: str = "results/compas_fairness_preferences.csv",
    model: LlamaModel = LlamaModel.LLAMA3_8B,
    temperature: float = 0.7,
    batch_size: int = 10,
    model_path_prefix: str = "../models/",
):
    """
    Run a study asking multiple personas for fairness judgments on random COMPAS pairs.
    Each persona evaluates a different random pair from the COMPAS dataset.
    Results are saved to a CSV file.
    
    Args:
        output_file: CSV file to save results
        model: Which Llama model to use
        temperature: Sampling temperature
        batch_size: Number of personas to process before saving results
        model_path_prefix: Directory prefix for model path
    """
    # Load selected personas from specific indices
    personas = load_unique_personas(start_indices=[1, 101, 201, 301, 401], count_per_index=1)
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
        'persona_id', 'persona', 'choice', 'choice_type',
        'individual1_id', 'individual1_sex', 'individual1_age', 'individual1_race', 
        'individual1_juv_fel', 'individual1_juv_misd', 'individual1_juv_other',
        'individual1_priors', 'individual1_charge',
        'individual2_id', 'individual2_sex', 'individual2_age', 'individual2_race', 
        'individual2_juv_fel', 'individual2_juv_misd', 'individual2_juv_other',
        'individual2_priors', 'individual2_charge'
    ]
    
    # Load model and tokenizer once (outside the persona loop)
    print(f"Loading {model.value} model and tokenizer...")
    model_path = f"{model_path_prefix}{model.value}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set padding token to EOS token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_instance = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        device_map="auto"  # Automatically use available GPUs
    )
    print("Model and tokenizer loaded successfully")
    
    # Open CSV file for results
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Process personas in batches
        results = []
        
        for i, (persona_id, persona_text) in enumerate(tqdm(personas, desc="Processing personas")):
            print(f"\nProcessing persona {i+1}/{len(personas)}: {persona_text[:50]}...")
            
            try:
                # Select a random pair for this persona
                individual1, individual2 = select_random_pair(clean_compas_df)
                
                # Query the model with this persona and the random pair
                response = fairness_query(
                    individual1=individual1,
                    individual2=individual2,
                    model_instance=model_instance,
                    tokenizer=tokenizer,
                    persona=persona_text,
                    persona_id=persona_id,
                    model=model,
                    temperature=temperature
                )
                
                # Get the choice text and type
                choice_text = response.full_text.strip()

                # Clean out any special tokens that might be in the response
                special_tokens = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", 
                                 "<|im_start|>", "<|im_end|>", "<|begin_of_text|>"]
                for token in special_tokens:
                    choice_text = choice_text.replace(token, "")
                
                choice_text = choice_text.strip()
                choice_type = response.choice

                # Convert charge degrees to full descriptions for CSV
                ind1_charge_full = individual1['c_charge_degree_full']
                ind2_charge_full = individual2['c_charge_degree_full']
                
                # Add result to batch
                result = {
                    'persona_id': persona_id,
                    'persona': persona_text,
                    'choice': choice_text,
                    'choice_type': choice_type,
                    # Individual 1 data
                    'individual1_id': individual1['id'],
                    'individual1_sex': individual1['sex'],
                    'individual1_age': individual1['age'],
                    'individual1_race': individual1['race'],
                    'individual1_juv_fel': individual1['juv_fel_count'],
                    'individual1_juv_misd': individual1['juv_misd_count'],
                    'individual1_juv_other': individual1['juv_other_count'],
                    'individual1_priors': individual1['priors_count'],
                    'individual1_charge': individual1['c_charge_degree'],
                    # Individual 2 data
                    'individual2_id': individual2['id'],
                    'individual2_sex': individual2['sex'],
                    'individual2_age': individual2['age'],
                    'individual2_race': individual2['race'],
                    'individual2_juv_fel': individual2['juv_fel_count'],
                    'individual2_juv_misd': individual2['juv_misd_count'],
                    'individual2_juv_other': individual2['juv_other_count'],
                    'individual2_priors': individual2['priors_count'],
                    'individual2_charge': individual2['c_charge_degree']
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

def test_prompt_formatting():
    # Load and prepare COMPAS data
    compas_df = load_compas_data()
    clean_compas_df = prepare_compas_data(compas_df)
    
    # Select a random pair
    individual1, individual2 = select_random_pair(clean_compas_df)
    
    # Generate the prompt text
    query_text = create_fairness_query_text(individual1, individual2)
    
    # Print the conversion and full prompt
    print(f"Individual 1 - Original: '{individual1['c_charge_degree']}', Converted: '{individual1['c_charge_degree_full']}'")
    print(f"Individual 2 - Original: '{individual2['c_charge_degree']}', Converted: '{individual2['c_charge_degree_full']}'")
    
    print("\n========= COMPLETE PROMPT EXAMPLE =========")
    print(query_text)
    print("==========================================\n")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a COMPAS fairness study with multiple personas")
    parser.add_argument("--output", default="results/compas_fairness_preferences.csv", help="Output CSV file")
    parser.add_argument("--model", default="llama3-8b-instruct", help="Model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for saving results")
    parser.add_argument("--model_path_prefix", default="../models/", help="Directory prefix for model path")
    
    args = parser.parse_args()
    test_prompt_formatting()
    # Use command line arguments
    print(f"Running COMPAS fairness study with selected personas")
    print(f"Saving results to {args.output}")
    
    model_enum = LlamaModel.LLAMA3_8B
    if args.model == "llama3-70b-instruct":
        model_enum = LlamaModel.LLAMA3_70B
    
    run_compas_fairness_study(
        output_file=args.output,
        model=model_enum,
        temperature=args.temperature,
        batch_size=args.batch_size,
        model_path_prefix=args.model_path_prefix
    )