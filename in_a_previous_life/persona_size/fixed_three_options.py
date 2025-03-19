#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run fixed comparisons study using all personas from the PERSONA dataset.
Each persona evaluates the same set of comparison pairs.

This script:
1. Loads personas from the persona dataset
2. Loads fixed comparisons from data/fixed_comparisons.json
3. Elicits fairness judgments using LLaMA with personas
4. Saves results to chunked output files for later combination
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum
from typing import List, Tuple, Optional, Union, Dict, Any, Set
import pandas as pd
import csv
import os
import random
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import time
import requests
from io import StringIO
import datetime
import gc
import json
import argparse

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
    explanation: str = None
    comparison_id: int = None
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
    
    # Try to extract response type and explanation
    choice = None
    explanation = None
    
    if "Should be treated similarly" in assistant_response:
        choice = "similar"
        # Try to extract the explanation that comes after the choice
        parts = assistant_response.split("Should be treated similarly", 1)
        if len(parts) > 1 and parts[1].strip():
            explanation = parts[1].strip()
    elif "X at least as high as Y" in assistant_response:
        choice = "x_higher_than_y"
        # Try to extract the explanation that comes after the choice
        parts = assistant_response.split("X at least as high as Y", 1)
        if len(parts) > 1 and parts[1].strip():
            explanation = parts[1].strip()
    elif "Y at least as high as X" in assistant_response:
        choice = "y_higher_than_x"
        parts = assistant_response.split("Y at least as high as X", 1)
        if len(parts) > 1 and parts[1].strip():
            explanation = parts[1].strip()
    
    return LlamaResponse(
        full_text=assistant_response,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        persona=persona,
        choice=choice,
        explanation=explanation
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

def prepare_compas_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and prepare the COMPAS dataset for fairness assessment.
    Select relevant columns, filter rows, and split into train/test sets.
    
    Args:
        df: Original COMPAS dataframe
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    print(f"Original dataset: {len(df)} rows")
    
    # Select relevant columns for fairness assessment
    columns = [
        'id', 'sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count', 
        'juv_other_count', 'priors_count', 'c_charge_degree', 'two_year_recid'
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
        clean_df['two_year_recid'] = clean_df['two_year_recid'].astype(int)
    except Exception as e:
        print(f"Warning during type conversion: {e}")
    
    # Add a column with full charge descriptions
    clean_df['c_charge_degree_full'] = clean_df['c_charge_degree'].apply(convert_charge_degree)
    
    # Split into train and test sets
    train_df, test_df = train_test_split(
        clean_df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=clean_df['two_year_recid']  # Stratify by recidivism to maintain class balance
    )
    
    # Print a summary
    print(f"\nCleaning Summary:")
    print(f"  - Original dataset: {len(df)} rows")
    print(f"  - Cleaned dataset: {len(clean_df)} rows")
    print(f"  - Removed: {len(df) - len(clean_df)} rows ({((len(df) - len(clean_df)) / len(df) * 100):.1f}%)")
    print(f"  - Training set: {len(train_df)} rows")
    print(f"  - Test set: {len(test_df)} rows")
    
    return train_df, test_df

def load_comparisons_from_file(df: pd.DataFrame, input_file: str = "data/fixed_comparisons.json") -> List[Tuple[int, pd.Series, pd.Series]]:
    """
    Load the fixed comparisons from a JSON file and reconstruct the complete Series objects.
    
    Args:
        df: DataFrame to use for looking up individuals
        input_file: Path to the JSON file
        
    Returns:
        List of tuples containing (comparison_id, individual1, individual2)
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Comparison file not found: {input_file}")
    
    with open(input_file, 'r') as f:
        serialized_comparisons = json.load(f)
    
    # Reconstruct the complete Series objects
    loaded_comparisons = []
    for comparison in serialized_comparisons:
        comparison_id = comparison["comparison_id"]
        individual1_id = comparison["individual1_id"]
        individual2_id = comparison["individual2_id"]
        
        # Look up the individuals in the DataFrame
        individual1 = df[df['id'] == individual1_id].iloc[0] if any(df['id'] == individual1_id) else None
        individual2 = df[df['id'] == individual2_id].iloc[0] if any(df['id'] == individual2_id) else None
        
        if individual1 is not None and individual2 is not None:
            loaded_comparisons.append((comparison_id, individual1, individual2))
        else:
            print(f"Warning: Could not find individuals for comparison {comparison_id}")
    
    print(f"Loaded {len(loaded_comparisons)} comparisons from {input_file}")
    return loaded_comparisons

def create_fairness_query_text(individual1: pd.Series, individual2: pd.Series) -> str:
    """
    Create a query for the LLM about fairness in predictions, using a side-by-side format
    for comparing two individuals from the COMPAS dataset.
    """
    # Add introduction and context about recidivism prediction
    intro = """
The COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) dataset contains information about defendants in criminal cases. This data is used for recidivism prediction - forecasting which defendants are likely to reoffend in the future.

Recidivism prediction algorithms analyze factors like criminal history, demographics, and current charges to estimate the likelihood a person will commit future crimes. These predictions inform decisions about bail, sentencing, and parole.

I'm going to show you information about two individuals from this dataset. In your view, as a matter of fairness, should the following two individuals receive similar recidivism risk prediction, or is it ok to give them different predictions? Remember higher predictions mean they are more likely to recommit a crime.
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
        f"{intro}\n\n"
        f"{comparison}\n\n"
        f"Question: In your personal view, which statement is most accurate?\n"
            f"\"1. Should be treated similarly\"\n"
        f"\"2. X at least as high as Y\"\n"
        f"\"3. Y at least as high as X\""
            f"Please think step-by-step about what you know about the relevant factors for recidivism risk assessment. "
            f"After analyzing both profiles, choose option 1, 2, or 3."
    )
    
    return query

def fairness_query(
    comparison_id: int,
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
    Ask Llama to choose whether two individuals should be treated similarly or differently.
    
    Args:
        comparison_id: ID for tracking this specific comparison
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
        system_prompt = f"""{base_system_prompt} with the persona of {persona}. First, think step by step about the relevant factors for both individuals and what might influence recidivism risk.After your reasoning, respond with ONE of these three exact phrases:
"Should be treated similarly"
"X at least as high as Y"
"Y at least as high as X"

Then provide a brief explanation (1-2 sentences) for your final decision."""

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
    response.comparison_id = comparison_id
    response.individual1 = individual1
    response.individual2 = individual2
    
    return response

def load_all_personas_from_unique_parquet() -> List[Tuple[int, str]]:
    """
    Load all personas from the unique_personas.parquet file
    
    Returns:
        List of tuples containing (persona_id, persona_description)
    """
    try:
        # Load the parquet file with unique personas
        df = pd.read_parquet("data/unique_personas.parquet")
        
        if 'persona' in df.columns:
            # Reset index to ensure we have 0-based indices
            df = df.reset_index(drop=True)
            
            # Get all personas
            all_personas = []
            for idx in range(len(df)):
                all_personas.append((idx, df.loc[idx, 'persona']))
            
            print(f"Loaded all {len(all_personas)} personas from unique_personas.parquet")
            print(f"First persona: {all_personas[0][1][:50]}...")
            print(f"Last persona: {all_personas[-1][1][:50]}...")
            
            return all_personas
        else:
            print("No persona column found in the parquet file")
            return []
    except Exception as e:
        print(f"Error loading personas: {e}")
        return []

def get_already_processed_pairs(output_file: str) -> Set[Tuple[int, int]]:
    """
    Check which persona_id and comparison_id pairs have already been processed
    
    Args:
        output_file: Path to the CSV output file
        
    Returns:
        Set of (persona_id, comparison_id) tuples that have already been processed
    """
    processed_pairs = set()
    
    if not os.path.exists(output_file):
        return processed_pairs
    
    try:
        # Read the CSV file and extract processed pairs
        df = pd.read_csv(output_file)
        
        if 'persona_id' in df.columns and 'comparison_id' in df.columns:
            # Get all persona_id and comparison_id pairs
            processed_pairs = set(zip(df['persona_id'], df['comparison_id']))
            print(f"Found {len(processed_pairs)} already processed (persona_id, comparison_id) pairs")
        
        return processed_pairs
    except Exception as e:
        print(f"Error reading processed pairs: {e}")
        return processed_pairs

def run_fixed_comparisons_study(
    output_file: str = "results/fixed_personas_comparisons.csv",
    model: LlamaModel = LlamaModel.LLAMA3_8B,
    temperature: float = 0.7,
    model_path_prefix: str = "../models/",
    fixed_comparisons_file: str = "../data/fixed_comparisons.json",
    batch_save_size: int = 10,
    debug: bool = False,
    resume: bool = True,
    start_persona_index: int = 0,
    end_persona_index: int = None,
    start_comparison_index: int = 0,
    end_comparison_index: int = None,
    random_state: int = 42
):
    """
    Run a study where all personas evaluate the same fixed set of comparisons from the COMPAS dataset.
    
    Args:
        output_file: CSV file to save results
        model: Which Llama model to use
        temperature: Sampling temperature
        model_path_prefix: Directory prefix for model path
        fixed_comparisons_file: Path to fixed comparisons file
        batch_save_size: Number of evaluations to process before saving CSV
        debug: Enable debug output
        resume: Whether to resume from already processed pairs
        start_persona_index: First persona index to process
        end_persona_index: Last persona index to process
        start_comparison_index: First comparison index to process
        end_comparison_index: Last comparison index to process
        random_state: Random seed for reproducibility
    """
    # Create a timestamp for logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting fixed comparisons COMPAS fairness study at {timestamp}")
    print(f"Settings: using fixed comparisons file {fixed_comparisons_file}, batch size {batch_save_size}")
    
    # Set seeds for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Load all personas from unique_personas.parquet
    all_personas = load_all_personas_from_unique_parquet()
    if not all_personas:
        print("No personas found. Exiting.")
        return
    
    total_personas_count = len(all_personas)
    # Validate persona index range
    if start_persona_index < 0:
        print(f"Warning: start_persona_index {start_persona_index} is negative, setting to 0")
        start_persona_index = 0

    if start_persona_index >= total_personas_count:
        print(f"Error: start_persona_index {start_persona_index} exceeds total personas count {total_personas_count}")
        return

    if end_persona_index is None:
        end_persona_index = total_personas_count - 1
        print(f"No end_persona_index specified, setting to last persona: {end_persona_index}")
    elif end_persona_index >= total_personas_count:
        print(f"Warning: end_persona_index {end_persona_index} exceeds total personas count {total_personas_count}, setting to last persona: {total_personas_count - 1}")
        end_persona_index = total_personas_count - 1
    elif end_persona_index < start_persona_index:
        print(f"Error: end_persona_index {end_persona_index} is less than start_persona_index {start_persona_index}")
        return

    # Filter personas based on range
    selected_personas = all_personas[start_persona_index:end_persona_index + 1]
    print(f"Selected personas: {len(selected_personas)} (from index {start_persona_index} to {end_persona_index})")
    
    # Load and prepare COMPAS data
    print("\nLoading COMPAS dataset...")
    compas_df = load_compas_data()
    train_df, test_df = prepare_compas_data(compas_df, test_size=0.2, random_state=random_state)
    print(f"Prepared COMPAS dataset with {len(train_df)} training examples and {len(test_df)} test examples")
    
    # Check if fixed comparisons already exist, and generate them if they don't
    if not os.path.exists(fixed_comparisons_file):
        print(f"Error: Fixed comparisons file {fixed_comparisons_file} not found.")
        print("Please run generate_fixed_comparisons.py first.")
        return
    
    # Load fixed comparisons
    loaded_comparisons = load_comparisons_from_file(train_df, fixed_comparisons_file)
    
    # Filter comparisons based on range
    if end_comparison_index is None:
        end_comparison_index = len(loaded_comparisons) - 1
        
    filtered_comparisons = [comp for comp in loaded_comparisons 
                           if comp[0] >= start_comparison_index and comp[0] <= end_comparison_index]
    
    print(f"Using {len(filtered_comparisons)} comparisons (from index {start_comparison_index} to {end_comparison_index})")
    
    # Get already processed pairs if resuming
    processed_pairs = set()
    if resume and os.path.exists(output_file):
        processed_pairs = get_already_processed_pairs(output_file)
        print(f"Resuming: Skipping {len(processed_pairs)} already processed (persona_id, comparison_id) pairs")
    
    # Create results directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Create or check output file
    file_exists = os.path.isfile(output_file)
    
    # Define fieldnames for CSV
    fieldnames = [
        'persona_id', 'persona', 'comparison_id', 'choice', 'choice_type', 'explanation',
        'individual1_id', 'individual1_sex', 'individual1_age', 'individual1_race', 
        'individual1_juv_fel', 'individual1_juv_misd', 'individual1_juv_other',
        'individual1_priors', 'individual1_charge',
        'individual2_id', 'individual2_sex', 'individual2_age', 'individual2_race', 
        'individual2_juv_fel', 'individual2_juv_misd', 'individual2_juv_other',
        'individual2_priors', 'individual2_charge',
        'timestamp'
    ]
    
    # Load model and tokenizer once (outside the persona loop)
    print(f"\nLoading {model.value} model and tokenizer...")
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
    
    # Calculate estimated completion time
    estimated_time_per_query = 5  # seconds
    total_personas = len(selected_personas)
    total_comparisons = len(filtered_comparisons)
    total_queries = total_personas * total_comparisons
    estimated_hours = (total_queries * estimated_time_per_query) / 3600
    
    # Count queries to process (excluding already processed)
    remaining_queries = 0
    for persona_id, _ in selected_personas:
        for comparison_id, _, _ in filtered_comparisons:
            if (persona_id, comparison_id) not in processed_pairs:
                remaining_queries += 1
    
    estimated_remaining_hours = (remaining_queries * estimated_time_per_query) / 3600
    print(f"\nEstimated completion time:")
    print(f"  - Total: {estimated_hours:.1f} hours for {total_queries} queries")
    print(f"  - Remaining: {estimated_remaining_hours:.1f} hours for {remaining_queries} queries")
    
    # Initialize progress tracking variables
    start_time = time.time()
    queries_completed = 0
    
    # Open CSV file for results
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Process each persona
        batch_results = []
        
        # Use tqdm for progress bar
        pbar = tqdm(total=remaining_queries, desc="Processing queries")
        
        # For each persona
        for persona_id, persona_text in selected_personas:
            # For each fixed comparison
            for comparison_id, individual1, individual2 in filtered_comparisons:
                # Skip if already processed and resuming
                if resume and (persona_id, comparison_id) in processed_pairs:
                    continue
                
                try:
                    # Query the model with this persona and comparison
                    response = fairness_query(
                        comparison_id=comparison_id,
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
                    
                    # Extract the explanation
                    explanation = response.explanation
                    
                    # Print response for verification if debugging
                    if debug:
                        print(f"Persona {persona_id}, Comparison {comparison_id}:")
                        print(f"  Response: \"{choice_text}\" (Type: {choice_type if choice_type else 'unknown'})")
                        if explanation:
                            print(f"  Explanation: \"{explanation}\"")
                    
                    # Create result dictionary
                    result = {
                        'persona_id': persona_id,
                        'persona': persona_text,
                        'comparison_id': comparison_id,
                        'choice': choice_text,
                        'choice_type': choice_type,
                        'explanation': explanation,
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
                        'individual2_charge': individual2['c_charge_degree'],
                        # Metadata
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    batch_results.append(result)
                    
                    # Update progress
                    queries_completed += 1
                    pbar.update(1)
                    
                    # Save batch to CSV if batch size reached
                    if len(batch_results) >= batch_save_size:
                        writer.writerows(batch_results)
                        csvfile.flush()  # Ensure data is written to disk
                        if debug:
                            print(f"Saved batch of {len(batch_results)} results to {output_file}")
                        batch_results = []
                        
                        # Run garbage collection to free memory
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Calculate and display progress
                    if queries_completed % 100 == 0:
                        elapsed_time = time.time() - start_time
                        queries_per_second = queries_completed / elapsed_time
                        remaining_queries = total_personas * total_comparisons - queries_completed
                        estimated_remaining_seconds = remaining_queries / queries_per_second
                        estimated_remaining_hours = estimated_remaining_seconds / 3600
                        
                        print(f"\nProgress: {queries_completed}/{total_personas * total_comparisons} queries")
                        print(f"Speed: {queries_per_second:.2f} queries/second")
                        print(f"Estimated time remaining: {estimated_remaining_hours:.1f} hours")
                    
                except Exception as e:
                    print(f"Error processing persona {persona_id}, comparison {comparison_id}: {e}")
        
        # Save any remaining results
        if batch_results:
            writer.writerows(batch_results)
            print(f"Saved final {len(batch_results)} results to {output_file}")
        
        pbar.close()
    
    end_time = time.time()
    total_time = end_time - start_time
    total_hours = total_time / 3600
    
    print(f"\nStudy complete at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_hours:.2f} hours")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run COMPAS fairness study with fixed comparisons")
    parser.add_argument("--output", default="results/chunked_outputs/fixed_personas_comparisons.csv", help="Output CSV file")
    parser.add_argument("--model", default="llama3-8b-instruct", help="Model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--model_path_prefix", default="../models/", help="Directory prefix for model path")
    parser.add_argument("--fixed_comparisons_file", default="../data/fixed_comparisons.json", help="Path to fixed comparisons file")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of results to process before saving")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Don't resume from previous progress")
    parser.add_argument("--start_index", type=int, default=0, help="Starting persona index (inclusive)")
    parser.add_argument("--end_index", type=int, default=None, help="Ending persona index (inclusive, None for all)")
    parser.add_argument("--start_comparison", type=int, default=0, help="Starting comparison index (inclusive)")
    parser.add_argument("--end_comparison", type=int, default=None, help="Ending comparison index (inclusive, None for all)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    # Use command line arguments
    print(f"Running COMPAS fairness study with fixed comparisons")
    print(f"Each persona will evaluate fixed pairs")
    print(f"Saving results to {args.output}")
    
    model_enum = LlamaModel.LLAMA3_8B
    if args.model == "llama3-70b-instruct":
        model_enum = LlamaModel.LLAMA3_70B
    
    run_fixed_comparisons_study(
        output_file=args.output,
        model=model_enum,
        temperature=args.temperature,
        model_path_prefix=args.model_path_prefix,
        fixed_comparisons_file=args.fixed_comparisons_file,
        batch_save_size=args.batch_size,
        debug=args.debug,
        start_persona_index=args.start_index,
        end_persona_index=args.end_index,
        start_comparison_index=args.start_comparison,
        end_comparison_index=args.end_comparison,
        resume=args.resume,
        random_state=args.random_state
    )