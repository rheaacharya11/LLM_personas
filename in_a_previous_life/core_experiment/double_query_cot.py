#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate fairness constraints for Jung et al.'s Algorithmic Fairness Elicitation framework.
Modified to query each pair twice (X vs Y and Y vs X) to allow for position bias analysis.
Streamlined version that collects all data but does minimal analysis.
"""
import os
import numpy as np
import pandas as pd
import random
import json
import re
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from datetime import datetime

# Define LLaMA role enum
class LlamaRole:
    """Enum for LLaMA roles (system, user, assistant)"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class FairnessConstraintGenerator:
    """
    Class for generating fairness constraints from personas, querying each pair twice.
    """
    
    def __init__(self, 
                model_path_prefix: str = "../../models/",
                llama_model: str = "llama3-8b-instruct",
                random_state: int = 42, 
                verbose: bool = True):
        """Initialize the fairness constraint generator."""
        self.model_path_prefix = model_path_prefix
        self.llama_model = llama_model
        self.random_state = random_state
        self.verbose = verbose
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Initialize LLaMA model and tokenizer (will be loaded when needed)
        if verbose:
            print(f"Initializing with {llama_model} model (will be loaded when needed)")
        
        self.llama_tokenizer = None
        self.llama_model_instance = None
    
    def _load_llama_model(self):
        """Load the LLaMA model and tokenizer if they haven't been loaded yet."""
        if self.llama_tokenizer is None or self.llama_model_instance is None:
            try:
                model_path = f"{self.model_path_prefix}{self.llama_model}"
                
                if self.verbose:
                    print(f"Loading LLaMA model from {model_path}...")
                
                # Load tokenizer
                self.llama_tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Set padding token to EOS token if not already set
                if self.llama_tokenizer.pad_token is None:
                    self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
                
                # Load model
                self.llama_model_instance = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
                    device_map="auto"  # Automatically use available GPUs
                )
                
                if self.verbose:
                    print("LLaMA model and tokenizer loaded successfully")
                    
            except Exception as e:
                raise RuntimeError(f"Error loading LLaMA model: {e}")
    
    def load_compas_data(self, train_path):
        """Load the preprocessed COMPAS dataset from the specified path."""
        if self.verbose:
            print(f"Loading COMPAS data from {train_path}")
            
        try:
            # Load train data
            train_df = pd.read_parquet(train_path)
            
            if self.verbose:
                print(f"Loaded {len(train_df)} examples from {train_path}")
            
            return train_df
            
        except Exception as e:
            raise RuntimeError(f"Error loading COMPAS data: {e}")
    
    def load_personas(self, personas_path, start_persona_index=0, end_persona_index=None):
        """Load personas from the specified path with range support."""
        if self.verbose:
            print(f"Loading personas from {personas_path}")
            
        try:
            # Load personas
            personas_df = pd.read_parquet(personas_path)
            
            # Get total number of personas
            total_personas = len(personas_df)
            
            # Validate indices
            if start_persona_index < 0:
                start_persona_index = 0
                
            if end_persona_index is None:
                end_persona_index = total_personas - 1
            elif end_persona_index >= total_personas:
                end_persona_index = total_personas - 1
            
            # Slice the dataframe to get the requested range
            personas_df = personas_df.iloc[start_persona_index:end_persona_index+1].reset_index()
            personas_df.rename(columns={'index': 'original_index'}, inplace=True)
            
            if self.verbose:
                print(f"Loaded {len(personas_df)} personas (index range: {start_persona_index} to {end_persona_index})")
                print(f"Original indices: {personas_df['original_index'].tolist()}")
            
            return personas_df
            
        except Exception as e:
            raise RuntimeError(f"Error loading personas: {e}")
    
    def generate_unique_comparisons(self, train_df, personas_df, pairs_per_persona=50):
        """Generate unique comparison pairs for each persona."""
        if self.verbose:
            print(f"Generating {pairs_per_persona} comparison pairs for each persona")
            print(f"Each pair will be queried twice (X vs Y and Y vs X)")
        
        all_ids = train_df['id'].values
        n_individuals = len(all_ids)
        
        # Get the original persona indices
        original_indices = personas_df['original_index'].tolist()
        
        # Generate pairs for each persona
        persona_comparisons = {}
        
        for i, persona_id in enumerate(original_indices):
            # Use persona_id to seed the RNG for reproducibility
            persona_rng = random.Random(self.random_state + persona_id)
            
            # Generate pairs for this persona
            persona_pairs = []
            for comparison_id in range(pairs_per_persona):
                # Sample two different individuals
                idx1, idx2 = persona_rng.sample(range(n_individuals), 2)
                individual1_id = int(all_ids[idx1])
                individual2_id = int(all_ids[idx2])
                
                persona_pairs.append({
                    "comparison_id": comparison_id,
                    "individual1_id": individual1_id,
                    "individual2_id": individual2_id
                })
            
            # Store with local index for processing within this batch
            persona_comparisons[i] = persona_pairs
        
        return persona_comparisons
    
    def create_fairness_query(self, individual1, individual2, swap_positions=False):
        """Create a query about fairness in recidivism prediction for a pair of individuals."""
        # If we're swapping positions, swap the individuals
        if swap_positions:
            individual1, individual2 = individual2, individual1
        
        intro = """
        The criminal justice system uses risk assessment algorithms to predict which defendants are likely to reoffend if released. These recidivism predictions influence decisions about bail, sentencing, and parole. I'm going to show you information about two individuals and ask you to consider how their risk should be assessed from a fairness perspective.
        """
    
        # Convert charge degree if needed
        def convert_charge_degree(charge_degree):
            if isinstance(charge_degree, str):
                if charge_degree.startswith('F'):
                    return f"Felony"
                elif charge_degree.startswith('M'):
                    return f"Misdemeanor"
            return charge_degree
    
        charge1 = convert_charge_degree(individual1.get('c_charge_degree', ''))
        charge2 = convert_charge_degree(individual2.get('c_charge_degree', ''))
        
        # Create the comparison with X and Y labels
        comparison = f"""
        PERSON X:
        - Gender: {individual1.get('sex', 'Unknown')}
        - Age: {individual1.get('age', 'Unknown')}
        - Race: {individual1.get('race', 'Unknown')}
        - Juvenile felonies: {individual1.get('juv_fel_count', 'Unknown')}
        - Juvenile misdemeanors: {individual1.get('juv_misd_count', 'Unknown')}
        - Juvenile other: {individual1.get('juv_other_count', 'Unknown')}
        - Prior convictions: {individual1.get('priors_count', 'Unknown')}
        - Current charge severity: {charge1}

        PERSON Y:
        - Gender: {individual2.get('sex', 'Unknown')}
        - Age: {individual2.get('age', 'Unknown')}
        - Race: {individual2.get('race', 'Unknown')}
        - Juvenile felonies: {individual2.get('juv_fel_count', 'Unknown')}
        - Juvenile misdemeanors: {individual2.get('juv_misd_count', 'Unknown')}
        - Juvenile other: {individual2.get('juv_other_count', 'Unknown')}
        - Prior convictions: {individual2.get('priors_count', 'Unknown')}
        - Current charge severity: {charge2}
        """
        
        # Fixed order options
        options = [
            "Person X should have a higher recidivism risk score than Person Y",
            "Person Y should have a higher recidivism risk score than Person X",
            "Both should have approximately equal risk scores"
        ]
        
        # Map the options to choice types
        option_mapping = {
            1: "x_higher_than_y",
            2: "y_higher_than_x",
            3: "similar"
        }
        
        # Chain of thought query format
        query = (
            f"{intro}\n\n"
            f"{comparison}\n\n"
            f"Question: In your personal view, which statement is most accurate?\n"
            f"1. {options[0]}\n"
            f"2. {options[1]}\n"
            f"3. {options[2]}\n\n"
            f"Please think step-by-step about what you know about the relevant factors for recidivism risk assessment. "
            f"After analyzing both profiles, choose option 1, 2, or 3."
        )
        
        return query, option_mapping
    
    def extract_choice_and_explanation(self, response_text):
    """Extract the chosen option and explanation from the model's response."""
    response_text = response_text.strip()
    
    # Default values
    chosen_option = None
    reasoning = response_text  # Store the full reasoning by default
    
    # Look for the final choice in the last few sentences
    last_paragraph = response_text.split('\n\n')[-1] if '\n\n' in response_text else response_text
    last_sentences = last_paragraph.split('. ')
    
    # Check the last few sentences for a clear choice
    sentences_to_check = min(3, len(last_sentences))
    for i in range(1, sentences_to_check + 1):
        sentence = last_sentences[-i]
        
        # Check for option indicators in the sentence
        if "option 1" in sentence.lower() or "choose 1" in sentence.lower() or "choose option 1" in sentence.lower():
            chosen_option = 1
            break
        elif "option 2" in sentence.lower() or "choose 2" in sentence.lower() or "choose option 2" in sentence.lower():
            chosen_option = 2
            break
        elif "option 3" in sentence.lower() or "choose 3" in sentence.lower() or "choose option 3" in sentence.lower():
            chosen_option = 3
            break
    
    # If no clear choice found in the last sentences, check the entire text
    if chosen_option is None:
        if "option 1" in response_text.lower() or "choose 1" in response_text.lower() or "choose option 1" in response_text.lower():
            chosen_option = 1
        elif "option 2" in response_text.lower() or "choose 2" in response_text.lower() or "choose option 2" in response_text.lower():
            chosen_option = 2
        elif "option 3" in response_text.lower() or "choose 3" in response_text.lower() or "choose option 3" in response_text.lower():
            chosen_option = 3
    
    # Look for more complex choice patterns
    if chosen_option is None:
        if "x should have a higher" in response_text.lower() or "person x should have a higher" in response_text.lower():
            chosen_option = 1
        elif "y should have a higher" in response_text.lower() or "person y should have a higher" in response_text.lower():
            chosen_option = 2
        elif "equal" in response_text.lower() or "same" in response_text.lower() or "similar" in response_text.lower():
            chosen_option = 3
    
    # If we still couldn't identify an option, try numeric patterns
    if chosen_option is None:
        # Check for standalone "1", "2", or "3" with punctuation
        for pattern in [r'\b1\b', r'\b2\b', r'\b3\b']:
            if re.search(pattern, response_text):
                chosen_option = int(re.search(pattern, response_text).group())
                break
    
    return chosen_option, reasoning
    
    def format_llama3_prompt(self, prompt: List[Tuple[str, str]]) -> str:
    """Format a list of (role, content) tuples into the Llama 3 prompt format."""
    # Make sure model is loaded
    self._load_llama_model()
    
    # Check if tokenizer has new special tokens
    has_new_format = "<|begin_of_text|>" in self.llama_tokenizer.get_vocab()
    
    if has_new_format:
        # Use Meta's new special tokens format
        formatted_prompt = "<|begin_of_text|>"
        
        for role, content in prompt:
            formatted_prompt += f"<|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>"
        
        # Remove the final eot_id from the last user message
        # and add the assistant header for the response
        if prompt and prompt[-1][0] == LlamaRole.USER:
            formatted_prompt = formatted_prompt[:-9]  # Remove last "<|eot_id|>"
            formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>"
    else:
        # Fall back to original im_start/im_end format
        formatted_prompt = ""
        
        for role, content in prompt:
            formatted_prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
        
        # Add the assistant starter token
        formatted_prompt += "<|im_start|>assistant\n"
    
    return formatted_prompt
    
    def llama_query(self, prompt: List[Tuple[str, str]], max_tokens: int = 512, 
                   temperature: float = 0.7) -> str:
        """Query the LLaMA model with a formatted prompt."""
        # Load model if not already loaded
        self._load_llama_model()
        
        # Format the prompt
        formatted_prompt = self.format_llama3_prompt(prompt)
        
        # Tokenize input
        inputs = self.llama_tokenizer(formatted_prompt, return_tensors="pt").to(self.llama_model_instance.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Generate response
        with torch.no_grad():
            output = self.llama_model_instance.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=(temperature > 0),
                pad_token_id=self.llama_tokenizer.pad_token_id
            )
        
        # Decode the response
        response_text = self.llama_tokenizer.decode(output[0], skip_special_tokens=False)
        
        # Extract just the assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in response_text:
            parts = response_text.split("<|start_header_id|>assistant<|end_header_id|>")
            assistant_response = parts[-1].split("<|eot_id|>")[0]
        elif "<|im_start|>assistant" in response_text:
            assistant_response = response_text.split("<|im_start|>assistant")[-1].strip()
            if "<|im_end|>" in assistant_response:
                assistant_response = assistant_response.split("<|im_end|>")[0].strip()
        else:
            assistant_response = response_text[len(formatted_prompt):].strip()
        
        return assistant_response
    
    def elicit_fairness_judgments(self, train_df, personas_df, pairs_per_persona=50, start_comparison=0, end_comparison=None):
    """
    Elicit fairness judgments, querying each comparison pair twice (X vs Y and Y vs X).
    This function only collects raw judgments without analyzing position bias.
    """
    if self.verbose:
        print(f"Eliciting fairness judgments for {len(personas_df)} personas")
        print(f"Each persona will evaluate {pairs_per_persona} comparison pairs, each presented twice")
        print(f"Each pair will be queried in both orders (X vs Y and Y vs X)")
        if start_comparison > 0 or end_comparison is not None:
            print(f"Processing comparison range: {start_comparison} to {end_comparison or pairs_per_persona-1}")
    
    # Make sure LLaMA model is loaded
    self._load_llama_model()
    
    # Create a lookup for individuals by ID
    individuals_by_id = {row['id']: row for _, row in train_df.iterrows()}
    
    # Generate unique comparisons for each persona
    persona_comparisons = self.generate_unique_comparisons(
        train_df, 
        personas_df,
        pairs_per_persona
    )
    
    # Adjust comparison range for parallelization
    if end_comparison is None:
        end_comparison = pairs_per_persona - 1
    
    # Dictionary to store fairness constraints for each persona
    persona_constraints = {}
    
    # Store all judgments for analysis
    all_judgments = []
    
    # Progress bar for the entire elicitation process
    total_pairs = len(personas_df) * (end_comparison - start_comparison + 1)
    progress_bar = tqdm(total=total_pairs, desc="Eliciting judgments", disable=not self.verbose)
    
    # Process each persona
    for persona_idx, (_, persona_row) in enumerate(personas_df.iterrows()):
        persona_id = persona_row['original_index']
        
        # Get persona description
        if 'persona' in personas_df.columns:
            persona_desc = persona_row['persona']
        else:
            persona_desc = f"Persona {persona_id}"
        
        # Initialize constraints for this persona
        constraints = []
        
        # Get comparisons for this persona
        persona_comparison_list = persona_comparisons.get(persona_idx, [])
        if not persona_comparison_list:
            progress_bar.update(end_comparison - start_comparison + 1)  # Skip all comparisons
            continue
            
        # Process each comparison pair within the specified range
        for comparison_idx in range(start_comparison, end_comparison + 1):
            if comparison_idx >= len(persona_comparison_list):
                progress_bar.update(1)  # Skip this comparison
                continue
                
            comparison_data = persona_comparison_list[comparison_idx]
            comparison_id = comparison_data['comparison_id']
            individual1_id = comparison_data['individual1_id']
            individual2_id = comparison_data['individual2_id']
            
            # Get the individuals
            individual1 = individuals_by_id.get(individual1_id)
            individual2 = individuals_by_id.get(individual2_id)
            
            if individual1 is None or individual2 is None:
                progress_bar.update(1)
                continue  # Skip if individual not found
            
            # Create system prompt with persona and chain of thought instructions
            system_prompt = (
                f"You are a helpful AI assistant with the persona of {persona_desc}. "
                f"When analyzing recidivism risk assessment, carefully reason through your thinking step-by-step. "
                f"Think about how these factors might predict future criminal behavior based on empirical evidence. "
                f"After your analysis, clearly state which option (1, 2, or 3) you choose."
            )
            
            # ----- Query 1: Normal order (X vs Y) -----
            query_text1, option_mapping = self.create_fairness_query(
                individual1, individual2, swap_positions=False
            )
            
            prompt_list1 = [
                (LlamaRole.SYSTEM, system_prompt),
                (LlamaRole.USER, query_text1)
            ]
            
            # Query the model for the normal order - increase max tokens for more thinking space
            response1 = self.llama_query(prompt_list1, max_tokens=512, temperature=0.7)
            
            # Extract chosen option and reasoning
            chosen_option1, reasoning1 = self.extract_choice_and_explanation(response1)
            
            # Map chosen option to judgment type
            if chosen_option1 is not None and 1 <= chosen_option1 <= 3:
                judgment1 = option_mapping[chosen_option1]
            else:
                judgment1 = "unknown"
            
            # Store the first judgment with individual data
            judgment_data1 = {
                'persona_id': persona_id,
                'persona': persona_desc,
                'comparison_id': comparison_id,
                'order': 'normal',
                'individual1_id': individual1_id,
                'individual2_id': individual2_id,
                'judgment': judgment1,
                'chosen_option': chosen_option1,
                'response': response1,
                'reasoning': reasoning1  # Store full chain of thought
            }
            
            # Add demographic information 
            for field in ['sex', 'age', 'race', 'priors_count', 'juv_fel_count', 
                         'juv_misd_count', 'juv_other_count', 'c_charge_degree']:
                if field in individual1:
                    judgment_data1[f'individual1_{field}'] = individual1[field]
                if field in individual2:
                    judgment_data1[f'individual2_{field}'] = individual2[field]
            
            # ----- Query 2: Swapped order (Y vs X) -----
            query_text2, option_mapping = self.create_fairness_query(
                individual1, individual2, swap_positions=True
            )
            
            prompt_list2 = [
                (LlamaRole.SYSTEM, system_prompt),
                (LlamaRole.USER, query_text2)
            ]
            
            # Query the model for the swapped order
            response2 = self.llama_query(prompt_list2, max_tokens=512, temperature=0.7)
            
            # Extract chosen option and reasoning
            chosen_option2, reasoning2 = self.extract_choice_and_explanation(response2)
            
            # Map chosen option to judgment type
            if chosen_option2 is not None and 1 <= chosen_option2 <= 3:
                judgment2 = option_mapping[chosen_option2]
            else:
                judgment2 = "unknown"
            
            # Store the second judgment
            judgment_data2 = {
                'persona_id': persona_id,
                'persona': persona_desc,
                'comparison_id': comparison_id,
                'order': 'swapped',
                'individual1_id': individual2_id,  # Swapped
                'individual2_id': individual1_id,  # Swapped
                'judgment': judgment2,
                'chosen_option': chosen_option2,
                'response': response2,
                'reasoning': reasoning2  # Store full chain of thought
            }
            
            # Add demographic information for swapped order
            for field in ['sex', 'age', 'race', 'priors_count', 'juv_fel_count', 
                         'juv_misd_count', 'juv_other_count', 'c_charge_degree']:
                if field in individual2:  # Swapped
                    judgment_data2[f'individual1_{field}'] = individual2[field]
                if field in individual1:  # Swapped
                    judgment_data2[f'individual2_{field}'] = individual1[field]
            
            # Add both judgments to our records
            all_judgments.append(judgment_data1)
            all_judgments.append(judgment_data2)
            
            # Simple constraint generation without position bias analysis
            # Just use the normal order judgments for simplicity
            if judgment1 == "similar":
                # Both directions - they should be treated the same
                constraints.append((individual1_id, individual2_id))
                constraints.append((individual2_id, individual1_id))
            elif judgment1 == "x_higher_than_y":
                # X should be treated at least as well as Y
                constraints.append((individual2_id, individual1_id))
            elif judgment1 == "y_higher_than_x":
                # Y should be treated at least as well as X
                constraints.append((individual1_id, individual2_id))
            
            # Update progress bar
            progress_bar.update(1)
        
        # Store constraints for this persona
        persona_constraints[persona_id] = constraints
    
    progress_bar.close()
    
    # Return all data without analysis
    return {
        'constraints': persona_constraints,
        'judgments': all_judgments
    }
    
    def save_constraint_data(self, constraint_data, output_file, append=False):
        """
        Save generated constraint data to a specified file.
        Streamlined version that just saves raw data without analysis.
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save all raw judgments to a CSV file
        if 'judgments' in constraint_data and constraint_data['judgments']:
            # Extract all fields except nested ones
            judgments_flat = []
            for j in constraint_data['judgments']:
                flat_j = {k: v for k, v in j.items() if not isinstance(v, (dict, list, tuple))}
                judgments_flat.append(flat_j)
            
            # Convert to DataFrame for easier CSV handling
            judgments_df = pd.DataFrame(judgments_flat)
            
            # Write all judgments to CSV
            mode = 'a' if append else 'w'
            header = not append
            judgments_df.to_csv(output_file, mode=mode, header=header, index=False)
        
        # Save constraints to a JSON file
        if 'constraints' in constraint_data:
            constraints_file = output_file.replace('.csv', '_constraints.json')
            
            # Convert constraints to serializable format
            serializable_constraints = {}
            for persona_id, constraints in constraint_data['constraints'].items():
                serializable_constraints[str(persona_id)] = [list(constraint) for constraint in constraints]
            
            # Write to file
            with open(constraints_file, 'w') as f:
                json.dump(serializable_constraints, f, indent=2)
        
        if self.verbose:
            print(f"\nSaved {len(constraint_data['judgments'])} judgments to {output_file}")
            if 'constraints' in constraint_data:
                print(f"Saved constraints to {constraints_file}")


def main():
    """Main function to run the constraint generation process."""
    parser = argparse.ArgumentParser(description="Generate fairness constraints using personas with position swap")
    
    # Data paths
    parser.add_argument("--train_path", type=str, default="../data/compas_train.parquet",
                       help="Path to training data")
    parser.add_argument("--personas_path", type=str, default="../data/unique_personas.parquet",
                       help="Path to personas data")
    parser.add_argument("--output", type=str, default="../results/core_experiment/chunked_outputs/fairness_judgments.csv",
                       help="Path to output CSV file for judgments")
    
    # Range parameters for hyper-parallelization
    parser.add_argument("--start_index", type=int, default=0,
                       help="Starting persona index (inclusive)")
    parser.add_argument("--end_index", type=int, default=None,
                       help="Ending persona index (inclusive, None for all)")
    parser.add_argument("--start_comparison", type=int, default=0,
                       help="Starting comparison index (inclusive)")
    parser.add_argument("--end_comparison", type=int, default=None,
                       help="Ending comparison index (inclusive, None for all)")
    
    # Experiment parameters
    parser.add_argument("--pairs_per_persona", type=int, default=50,
                       help="Number of comparison pairs per persona (each will be queried twice)")
    
    # Model parameters
    parser.add_argument("--model_path_prefix", type=str, default="../../models/",
                       help="Directory prefix for LLaMA model path")
    parser.add_argument("--model", type=str, default="llama3-8b-instruct",
                       help="LLaMA model to use (llama3-8b-instruct or llama3-70b-instruct)")
    
    # Other parameters
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Record start time
    start_time = time.time()
    
    # Initialize constraint generator
    generator = FairnessConstraintGenerator(
        model_path_prefix=args.model_path_prefix,
        llama_model=args.model,
        random_state=args.random_seed,
        verbose=True
    )
    
    # Load COMPAS data
    train_df = generator.load_compas_data(args.train_path)
    
    # Load personas for the specified range
    personas_df = generator.load_personas(
        args.personas_path,
        start_persona_index=args.start_index,
        end_persona_index=args.end_index
    )
    
    # Elicit fairness judgments with position swap
    constraint_data = generator.elicit_fairness_judgments(
        train_df, 
        personas_df,
        pairs_per_persona=args.pairs_per_persona,
        start_comparison=args.start_comparison,
        end_comparison=args.end_comparison
    )
    
    # Determine if we should append or overwrite the output file
    # (append if processing chunked data)
    append_mode = (args.start_index > 0 or args.start_comparison > 0) and os.path.exists(args.output)
    
    # Save constraint data
    generator.save_constraint_data(
        constraint_data, 
        args.output,
        append=append_mode
    )
    
    # Report timing
    elapsed_time = time.time() - start_time
    print(f"\nConstraint generation completed in {elapsed_time:.2f} seconds")
    print(f"Processed persona range: {args.start_index} to {args.end_index or 'end'}")
    print(f"Processed comparison range: {args.start_comparison} to {args.end_comparison or (args.pairs_per_persona-1)}")
    
if __name__ == "__main__":
    main()