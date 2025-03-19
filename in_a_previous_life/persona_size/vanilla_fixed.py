#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run vanilla LLM (without personas) with Chain of Thought reasoning on fixed comparisons
using 3 options for judgments (similar, X higher than Y, Y higher than X).

This script:
1. Loads the COMPAS training data
2. Loads fixed comparison pairs
3. Queries vanilla LLM with COT prompting
4. Records judgments, reasoning, and explanations
"""

import os
import numpy as np
import pandas as pd
import random
import json
import argparse
import csv
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime
import time
from enum import Enum
import requests
from io import StringIO

# Define LLaMA role enum
class LlamaRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class LlamaModel(Enum):
    LLAMA3_8B = "llama3-8b-instruct"
    LLAMA3_70B = "llama3-70b-instruct"

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
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
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
        
    Returns:
        The model's response text
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
    
    # Clean out any special tokens that might be in the response
    special_tokens = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", 
                     "<|im_start|>", "<|im_end|>", "<|begin_of_text|>"]
    for token in special_tokens:
        assistant_response = assistant_response.replace(token, "")
    
    return assistant_response.strip()

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
    Create a query for the LLM about fairness in predictions, using a side-by-side list format
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

def extract_cot_response(full_text):
    """
    Extract Chain of Thought reasoning and judgment from response
    
    Args:
        full_text: Complete response from LLM
        
    Returns:
        Tuple of (reasoning, choice_type, explanation)
    """
    # Keywords to search for
    keywords = ["Should be treated similarly", "X at least as high as Y", "Y at least as high as X"]
    
    # Find the first occurrence of any keyword
    positions = {}
    for keyword in keywords:
        pos = full_text.find(keyword)
        if pos != -1:
            positions[keyword] = pos
    
    if not positions:
        # No keywords found
        return full_text, None, ""
    
    # Find the keyword that appears first
    first_keyword = min(positions.items(), key=lambda x: x[1])
    keyword_name = first_keyword[0]
    keyword_pos = first_keyword[1]
    
    # Split the response
    reasoning = full_text[:keyword_pos].strip()
    choice = keyword_name
    explanation = full_text[keyword_pos + len(keyword_name):].strip()
    
    # Map to choice type
    choice_type_map = {
        "Should be treated similarly": "similar",
        "X at least as high as Y": "x_higher_than_y",
        "Y at least as high as X": "y_higher_than_x"
    }
    choice_type = choice_type_map.get(choice)
    
    return reasoning, choice_type, explanation

def run_vanilla_llm_with_cot(
    output_file: str = "results/vanilla_llm_cot.csv",
    model: LlamaModel = LlamaModel.LLAMA3_8B,
    model_path_prefix: str = "../models/",
    fixed_comparisons_file: str = "data/fixed_comparisons.json",
    temperature: float = 0.7,
    debug: bool = False
):
    """
    Run fixed comparisons using vanilla LLM with:
    - 3 options approach (similar, X higher than Y, Y higher than X)
    - Chain of Thought reasoning
    - No personas
    
    Args:
        output_file: Path to save results
        model: Which LLaMA model to use
        model_path_prefix: Directory prefix for model path
        fixed_comparisons_file: Path to fixed comparisons file
        temperature: Sampling temperature
        debug: Whether to print debug information
    """
    # Create a timestamp for logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting vanilla LLM with COT fairness study at {timestamp}")
    
    # Load COMPAS data and fixed comparisons
    print("\nLoading COMPAS dataset...")
    compas_df = load_compas_data()
    train_df, _ = prepare_compas_data(compas_df)
    
    # Check if fixed comparisons already exist, and generate them if they don't
    if not os.path.exists(fixed_comparisons_file):
        print(f"Error: Fixed comparisons file {fixed_comparisons_file} not found.")
        print("Please run generate_fixed_comparisons.py first.")
        return
    
    # Load fixed comparisons
    loaded_comparisons = load_comparisons_from_file(train_df, fixed_comparisons_file)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Load model and tokenizer
    print(f"\nLoading {model.value} model and tokenizer...")
    model_path = f"{model_path_prefix}{model.value}"
    
    try:
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
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Define fieldnames for CSV
    fieldnames = [
        'comparison_id', 'choice', 'choice_type', 'reasoning', 'final_explanation',
        'individual1_id', 'individual1_sex', 'individual1_age', 'individual1_race', 
        'individual1_juv_fel', 'individual1_juv_misd', 'individual1_juv_other',
        'individual1_priors', 'individual1_charge',
        'individual2_id', 'individual2_sex', 'individual2_age', 'individual2_race', 
        'individual2_juv_fel', 'individual2_juv_misd', 'individual2_juv_other',
        'individual2_priors', 'individual2_charge',
        'timestamp'
    ]
    
    # Create or append to CSV file
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, 'a' if file_exists else 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Process each comparison
        for comparison_id, individual1, individual2 in tqdm(loaded_comparisons, desc="Processing comparisons"):
            # Modified system prompt for 3 options and COT
            system_prompt = """You are a helpful AI assistant. I want you to analyze whether two people should receive the same or different recidivism risk prediction.

First, think step by step about the relevant factors for both individuals and what might influence recidivism risk.

After your reasoning, respond with ONE of these three exact phrases:
"Should be treated similarly"
"X at least as high as Y"
"Y at least as high as X"

Then provide a brief explanation (1-2 sentences) for your final decision."""
            
            # Create query text
            query_text = create_fairness_query_text(individual1, individual2)
            
            # Create prompt list
            prompt_list = [
                (LlamaRole.SYSTEM, system_prompt),
                (LlamaRole.USER, query_text)
            ]
            
            try:
                # Query model
                response = llama_query(
                    prompt=prompt_list,
                    model_instance=model_instance,
                    tokenizer=tokenizer,
                    model=model,
                    temperature=temperature,
                    max_tokens=1024  # Increase token count for COT
                )
                
                # Extract both reasoning and judgment
                reasoning, choice_type, explanation = extract_cot_response(response)
                
                # Default choice if extraction failed
                if choice_type is None:
                    choice = "Unspecified"
                else:
                    choice_map = {
                        "similar": "Should be treated similarly",
                        "x_higher_than_y": "X at least as high as Y",
                        "y_higher_than_x": "Y at least as high as X"
                    }
                    choice = choice_map.get(choice_type, "Unspecified")
                
                # Print response for debugging
                if debug:
                    print(f"\nComparison {comparison_id}:")
                    print(f"Choice: {choice}")
                    print(f"Choice type: {choice_type}")
                    print(f"Reasoning: {reasoning[:100]}...")
                    print(f"Explanation: {explanation[:100]}...")
                
                # Create result dictionary
                result = {
                    'comparison_id': comparison_id,
                    'choice': choice,
                    'choice_type': choice_type,
                    'reasoning': reasoning,
                    'final_explanation': explanation,
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
                
                # Write result to CSV
                writer.writerow(result)
                csvfile.flush()  # Ensure data is written to disk
                
            except Exception as e:
                print(f"Error processing comparison {comparison_id}: {e}")
    
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nFinished at {end_time}")
    print(f"Results saved to {output_file}")

def analyze_results(output_file: str):
    """
    Perform basic analysis on the vanilla LLM COT results.
    
    Args:
        output_file: Path to the results CSV file
    """
    if not os.path.exists(output_file):
        print(f"Results file not found: {output_file}")
        return
    
    try:
        # Load results
        df = pd.read_csv(output_file)
        
        # Basic stats
        total_comparisons = len(df)
        
        # Choice distribution
        choice_counts = df['choice_type'].value_counts()
        
        print("\nResults Analysis:")
        print(f"Total comparisons: {total_comparisons}")
        print("\nChoice distribution:")
        for choice, count in choice_counts.items():
            print(f"  - {choice}: {count} ({count/total_comparisons*100:.1f}%)")
        
        # Create output directory for analysis
        analysis_dir = os.path.join(os.path.dirname(output_file), "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save summary to file
        summary_path = os.path.join(analysis_dir, "vanilla_llm_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                'total_comparisons': total_comparisons,
                'choice_distribution': {k: int(v) for k, v in choice_counts.items()},
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        print(f"\nSummary saved to {summary_path}")
        return df
        
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vanilla LLM with Chain of Thought reasoning")
    parser.add_argument("--output", default="results/vanilla_llm_cot.csv", help="Output CSV file")
    parser.add_argument("--model", default="llama3-8b-instruct", help="Model to use")
    parser.add_argument("--model_path_prefix", default="../models/", help="Directory prefix for model path")
    parser.add_argument("--fixed_comparisons_file", default="data/fixed_comparisons.json", help="Path to fixed comparisons file")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument("--analyze", action="store_true", help="Analyze results after running")
    
    args = parser.parse_args()
    
    # Map model name to enum
    model_enum = LlamaModel.LLAMA3_8B
    if args.model == "llama3-70b-instruct":
        model_enum = LlamaModel.LLAMA3_70B
    
    # Run the vanilla LLM with COT
    run_vanilla_llm_with_cot(
        output_file=args.output,
        model=model_enum,
        model_path_prefix=args.model_path_prefix,
        fixed_comparisons_file=args.fixed_comparisons_file,
        temperature=args.temperature,
        debug=args.debug
    )
    
    # Analyze results if requested
    if args.analyze:
        analyze_results(args.output)