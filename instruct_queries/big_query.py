#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum
from typing import List, Tuple, Optional, Union, Dict, Any, Set
import pandas as pd
import csv
import os
import random
from dataclasses import dataclass
from tqdm import tqdm
import time
import requests
from io import StringIO
import datetime
import gc

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
    
    if "Should be treated equally" in assistant_response:
        choice = "equal"
        # Try to extract the explanation that comes after the choice
        parts = assistant_response.split("Should be treated equally", 1)
        if len(parts) > 1 and parts[1].strip():
            explanation = parts[1].strip()
    elif "Ok to treat differently" in assistant_response:
        choice = "different"
        # Try to extract the explanation that comes after the choice
        parts = assistant_response.split("Ok to treat differently, or no opinion", 1)
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
    that's easy for LLMs to parse. Includes context about the COMPAS dataset and recidivism prediction.
    """
    # Add introduction and context about recidivism prediction
    intro = """
The COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) dataset contains information about defendants in criminal cases. This data is used for recidivism prediction - forecasting which defendants are likely to reoffend in the future.

Recidivism prediction algorithms analyze factors like criminal history, demographics, and current charges to estimate the likelihood a person will commit future crimes. These predictions inform decisions about bail, sentencing, and parole.

I'm going to show you information about two individuals from this dataset. In your view, as a matter of fairness, shoudl the following two individuals receive similar recidivism risk prediction, or is it ok to give them different predictions?
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
        f"In your view, as a matter of fairness, should the following two individuals receive "
        f"the same recidivism prediction, or is it ok to give them different predictions?\n\n"
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
        system_prompt = f"{base_system_prompt} with the persona of {persona}. First, state ONLY ONE of these two exact phrases: \"Should be treated equally\" or \"Ok to treat differently, or no opinion\". Then, you may provide ONE brief explanation (< 10 words) ."
    else:
        system_prompt = f"{base_system_prompt}. First, state ONLY ONE of these two exact phrases: \"Should be treated equally\" or \"Ok to treat differently, or no opinion\". Then, you can provide < 10 words for explanation."
    
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

def get_already_processed_persona_ids(output_file: str) -> Set[int]:
    """
    Check which persona IDs have already been processed by reading the output file
    
    Args:
        output_file: Path to the CSV output file
        
    Returns:
        Set of persona IDs that have already been processed
    """
    processed_ids = set()
    
    if not os.path.exists(output_file):
        return processed_ids
    
    try:
        # Read the CSV file and extract unique persona IDs that have been fully processed
        df = pd.read_csv(output_file)
        
        # Group by persona_id and count the number of comparisons
        if 'persona_id' in df.columns and 'comparison_number' in df.columns:
            completed_counts = df.groupby('persona_id')['comparison_number'].count()
            
            # Get IDs of personas that have completed all comparisons
            for persona_id, count in completed_counts.items():
                if count == 50:  # 50 comparisons per persona
                    processed_ids.add(int(persona_id))
            
            print(f"Found {len(processed_ids)} already fully processed personas")
        
        return processed_ids
    except Exception as e:
        print(f"Error reading processed persona IDs: {e}")
        return processed_ids

def run_large_scale_compas_fairness_study(
    output_file: str = "results/large_scale_compas_study.csv",
    model: LlamaModel = LlamaModel.LLAMA3_8B,
    temperature: float = 0.7,
    model_path_prefix: str = "../models/",
    comparisons_per_persona: int = 50,
    batch_save_size: int = 10,
    debug: bool = False,
    resume: bool = True,
    start_persona_index: int = 0,
    end_persona_index: int = None
):
    """
    Run a large-scale study asking all personas from unique_personas.parquet to evaluate 50 random COMPAS pairs each.
    Results are saved to a CSV file, with batch saving and resume capabilities.
    
    Args:
        output_file: CSV file to save results
        model: Which Llama model to use
        temperature: Sampling temperature
        model_path_prefix: Directory prefix for model path
        comparisons_per_persona: Number of comparisons for each persona
        batch_save_size: Number of personas to process before saving CSV
        debug: Enable debug output
        resume: Whether to resume from the last processed persona
    """
    # Create a timestamp for logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting large-scale COMPAS fairness study at {timestamp}")
    print(f"Settings: {comparisons_per_persona} comparisons per persona, batch size {batch_save_size}")
    
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
        
    # Get already processed persona IDs if resuming
    processed_ids = set()
    if resume and os.path.exists(output_file):
        processed_ids = get_already_processed_persona_ids(output_file)
        print(f"Resuming: Skipping {len(processed_ids)} already processed personas")
    
    # Load and prepare COMPAS data
    print("\nLoading COMPAS dataset...")
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
        'persona_id', 'persona', 'comparison_number', 'choice', 'choice_type', 'explanation',
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
    total_personas = len(all_personas) - len(processed_ids)
    total_queries = total_personas * comparisons_per_persona
    estimated_hours = (total_queries * estimated_time_per_query) / 3600
    print(f"\nEstimated completion time: {estimated_hours:.1f} hours for {total_queries} queries")
    
    # Initialize progress tracking variables
    start_time = time.time()
    persona_count = 0
    
    
    # Open CSV file for results
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Process each persona
        batch_results = []
        
        # Use tqdm for overall progress bar
        for persona_idx, (persona_id, persona_text) in enumerate(tqdm(selected_personas, desc="Processing personas")):
            # Skip already processed personas
            if persona_id in processed_ids:
                continue
            
            persona_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Calculate progress and ETA
            if persona_count > 1:
                time_per_persona = elapsed_time / persona_count
                remaining_personas = total_personas_count - persona_idx - 1
                eta_seconds = time_per_persona * remaining_personas
                eta_hours = eta_seconds / 3600
                
                print(f"\nProgress: {persona_idx+1}/{total_personas_count} personas ({(persona_idx+1)/total_personas_count*100:.1f}%)")
                print(f"ETA: {eta_hours:.1f} hours")
            
            print(f"Processing persona {persona_id}:")
            if debug:
                print(f"Persona text: {persona_text[:100]}...")
            
            # Process multiple comparisons for each persona
            for comparison_idx in range(comparisons_per_persona):
                try:
                    # Select a random pair for this comparison
                    individual1, individual2 = select_random_pair(clean_compas_df)
                    
                    # For debugging, print pairs info
                    if debug and comparison_idx == 0:  # Only print for the first comparison
                        print(f"\nSample pair info:")
                        print(f"  Individual A: {individual1['sex']}, {individual1['age']}, {individual1['race']}, Priors: {individual1['priors_count']}")
                        print(f"  Individual B: {individual2['sex']}, {individual2['age']}, {individual2['race']}, Priors: {individual2['priors_count']}")
                    
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
                    
                    # Print progress indicator
                    if comparison_idx % 10 == 0 or debug:
                        print(f"  Completed {comparison_idx+1}/{comparisons_per_persona} comparisons")
                    
                    # Extract the explanation
                    explanation = response.explanation
                    
                    # Print response for verification if debugging
                    if debug:
                        print(f"    Response: \"{choice_text}\" (Type: {choice_type if choice_type else 'unknown'})")
                        if explanation:
                            print(f"    Explanation: \"{explanation}\"")
                    
                    # Create result dictionary
                    result = {
                        'persona_id': persona_id,
                        'persona': persona_text,
                        'comparison_number': comparison_idx + 1,
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
                    
                    # Small delay to prevent overloading
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error processing comparison: {e}")
            
            # Save batch to CSV if batch size reached
            if len(batch_results) >= batch_save_size * comparisons_per_persona:
                writer.writerows(batch_results)
                csvfile.flush()  # Ensure data is written to disk
                print(f"Saved batch of {len(batch_results)} results to {output_file}")
                batch_results = []
                
                # Run garbage collection to free memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Save any remaining results
        if batch_results:
            writer.writerows(batch_results)
            print(f"Saved final {len(batch_results)} results to {output_file}")
    
    end_time = time.time()
    total_time = end_time - start_time
    total_hours = total_time / 3600
    
    print(f"\nStudy complete at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_hours:.2f} hours")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a large-scale COMPAS fairness study with all personas")
    parser.add_argument("--output", default="results/large_scale_compas_study.csv", help="Output CSV file")
    parser.add_argument("--model", default="llama3-8b-instruct", help="Model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--model_path_prefix", default="../models/", help="Directory prefix for model path")
    parser.add_argument("--comparisons", type=int, default=50, help="Number of comparisons per persona")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of personas to process before saving")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Don't resume from last processed persona")
    parser.add_argument("--start_index", type=int, default=0, help="Starting persona index (inclusive)")
    parser.add_argument("--end_index", type=int, default=None, help="Ending persona index (inclusive, None for all)")
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    # Use command line arguments
    print(f"Running large-scale COMPAS fairness study with all personas from unique_personas.parquet")
    print(f"Each persona will evaluate {args.comparisons} random pairs")
    print(f"Saving results to {args.output}")
    
    model_enum = LlamaModel.LLAMA3_8B
    if args.model == "llama3-70b-instruct":
        model_enum = LlamaModel.LLAMA3_70B
    
    run_large_scale_compas_fairness_study(
        output_file=args.output,
        model=model_enum,
        temperature=args.temperature,
        model_path_prefix=args.model_path_prefix,
        comparisons_per_persona=args.comparisons,
        batch_save_size=args.batch_size,
        debug=args.debug,
        start_persona_index=args.start_index,
        end_persona_index=args.end_index,
        resume=args.resume
    )