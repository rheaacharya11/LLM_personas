#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate fairness constraints using a vanilla LLM with chain-of-thought reasoning.
Set up to run 50,000 comparisons (50 x 1000) without using personas.
Supports parallel processing with chunks for efficient execution.
"""
import os
import numpy as np
import pandas as pd
import random
import json
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from datetime import datetime

# Define LLaMA role enum
class LlamaRole:
    """Enum for LLaMA roles (system, user, assistant)"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class VanillaFairnessConstraintGenerator:
    """
    Class for generating fairness constraints using a vanilla LLM.
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
    
    def generate_unique_comparisons(self, train_df, total_comparisons=50000, chunk_id=0, chunks=50):
        """
        Generate unique comparison pairs divided into chunks for parallel processing.
        
        Args:
            train_df: DataFrame containing COMPAS data
            total_comparisons: Total number of comparisons to generate (default: 50000)
            chunk_id: ID of the current chunk (0-based, default: 0)
            chunks: Number of chunks to divide the comparisons into (default: 50)
            
        Returns:
            List of comparison dictionaries for the specified chunk
        """
        if self.verbose:
            print(f"Generating chunk {chunk_id+1}/{chunks} of {total_comparisons} total comparisons")
        
        all_ids = train_df['id'].values
        n_individuals = len(all_ids)
        
        # Calculate comparisons per chunk
        comparisons_per_chunk = total_comparisons // chunks
        
        # Calculate start and end indices for this chunk
        start_idx = chunk_id * comparisons_per_chunk
        end_idx = start_idx + comparisons_per_chunk
        if chunk_id == chunks - 1:  # Last chunk gets any remainder
            end_idx = total_comparisons
            
        # Get number of comparisons for this chunk
        chunk_size = end_idx - start_idx
            
        if self.verbose:
            print(f"This chunk will generate {chunk_size} comparisons (indices {start_idx}-{end_idx-1})")
        
        # Initialize RNG with seed for reproducibility
        # Add chunk_id to seed to ensure different chunks get different comparisons
        comparison_rng = random.Random(self.random_state + chunk_id)
        
        # Generate pairs for this chunk
        comparisons = []
        for i in range(chunk_size):
            comparison_id = start_idx + i
            
            # Sample two different individuals
            idx1, idx2 = comparison_rng.sample(range(n_individuals), 2)
            individual1_id = int(all_ids[idx1])
            individual2_id = int(all_ids[idx2])
            
            comparisons.append({
                "comparison_id": comparison_id,
                "individual1_id": individual1_id,
                "individual2_id": individual2_id
            })
        
        return comparisons
    
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
            f"Question: In your view, which statement is most accurate?\n"
            f"1. {options[0]}\n"
            f"2. {options[1]}\n"
            f"3. {options[2]}\n\n"
            f"Please think step-by-step about the relevant factors for recidivism risk assessment. "
            f"Consider each person's criminal history, age, and other pertinent factors that might impact recidivism. "
            f"After analyzing both profiles, choose option 1, 2, or 3."
        )
        
        return query, option_mapping
    
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
    
    def elicit_fairness_judgments(self, train_df, total_comparisons=50000, chunk_id=0, chunks=50):
        """
        Elicit fairness judgments using vanilla LLM, querying each comparison pair twice (X vs Y and Y vs X).
        This function processes a chunk of the total comparisons for parallel execution.
        
        Args:
            train_df: DataFrame containing COMPAS data
            total_comparisons: Total number of comparisons across all chunks
            chunk_id: ID of the current chunk (0-based)
            chunks: Total number of chunks
            
        Returns:
            Dictionary with constraints and judgments for this chunk
        """
        if self.verbose:
            print(f"Eliciting fairness judgments for chunk {chunk_id+1}/{chunks}")
            print(f"Total comparisons: {total_comparisons}, approx. {total_comparisons//chunks} per chunk")
        
        # Make sure LLaMA model is loaded
        self._load_llama_model()
        
        # Create a lookup for individuals by ID
        individuals_by_id = {row['id']: row for _, row in train_df.iterrows()}
        
        # Generate unique comparisons for this chunk
        comparisons = self.generate_unique_comparisons(
            train_df, 
            total_comparisons=total_comparisons,
            chunk_id=chunk_id,
            chunks=chunks
        )
        
        # Store all judgments for analysis
        all_judgments = []
        
        # Dictionary to store fairness constraints
        constraints = []
        
        # Progress bar for this chunk
        progress_bar = tqdm(total=len(comparisons), desc=f"Chunk {chunk_id+1}/{chunks}", disable=not self.verbose)
        
        # Process each comparison pair in this chunk
        for i, comparison_data in enumerate(comparisons):
            comparison_id = comparison_data['comparison_id']
            individual1_id = comparison_data['individual1_id']
            individual2_id = comparison_data['individual2_id']
            
            # Get the individuals
            individual1 = individuals_by_id.get(individual1_id)
            individual2 = individuals_by_id.get(individual2_id)
            
            if individual1 is None or individual2 is None:
                progress_bar.update(1)
                continue  # Skip if individual not found
            
            # Create system prompt with chain of thought instructions (no persona)
            system_prompt = (
                "You are a helpful AI assistant analyzing recidivism risk assessment. "
                "Carefully reason through your thinking step-by-step. "
                "Consider each individual's criminal history, demographic factors, and other relevant details. "
                "Think about how these factors might predict future criminal behavior based on empirical evidence. "
                "After your analysis, clearly state which option (1, 2, or 3) you choose."
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
                'chunk_id': chunk_id,
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
                'chunk_id': chunk_id,
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
        
        progress_bar.close()
        
        # Return all data without analysis
        return {
            'constraints': constraints,
            'judgments': all_judgments
        }
    
    def save_constraint_data(self, constraint_data, output_file, chunk_id=0):
        """
        Save generated constraint data to a specified file.
        
        Args:
            constraint_data: Dictionary with constraints and judgments
            output_file: Base path for output files
            chunk_id: ID of the current chunk (for naming output files)
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create chunk-specific filenames
        judgments_file = output_file.replace('.csv', f'_chunk{chunk_id}.csv')
        constraints_file = output_file.replace('.csv', f'_constraints_chunk{chunk_id}.json')
        
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
            judgments_df.to_csv(judgments_file, index=False)
        
        # Save constraints to a JSON file
        if 'constraints' in constraint_data:
            # Convert constraints to serializable format
            serializable_constraints = [list(constraint) for constraint in constraint_data['constraints']]
            
            # Write to file
            with open(constraints_file, 'w') as f:
                json.dump(serializable_constraints, f, indent=2)
        
        if self.verbose:
            print(f"\nSaved {len(constraint_data['judgments'])} judgments to {judgments_file}")
            if 'constraints' in constraint_data:
                print(f"Saved {len(constraint_data['constraints'])} constraints to {constraints_file}")


def main():
    """Main function to run the constraint generation process."""
    parser = argparse.ArgumentParser(description="Generate fairness constraints using vanilla LLM")
    
    # Data paths
    parser.add_argument("--train_path", type=str, default="../data/compas_train.parquet",
                       help="Path to training data")
    parser.add_argument("--output", type=str, default="../results/vanilla_experiment/fairness_judgments.csv",
                       help="Path to output CSV file for judgments")
    
    # Range parameters for hyper-parallelization
    parser.add_argument("--total_comparisons", type=int, default=50000,
                       help="Total number of comparisons to run (default: 50,000)")
    parser.add_argument("--chunks", type=int, default=50,
                       help="Number of chunks to divide the comparisons into (default: 50)")
    parser.add_argument("--chunk_id", type=int, default=0,
                       help="ID of the current chunk to process (0-49)")
    
    # Model parameters
    parser.add_argument("--model_path_prefix", type=str, default="../../models/",
                       help="Directory prefix for LLaMA model path")
    parser.add_argument("--model", type=str, default="llama3-8b-instruct",
                       help="LLaMA model to use (llama3-8b-instruct or llama3-70b-instruct)")
    
    # Other parameters
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate chunk_id
    if args.chunk_id < 0 or args.chunk_id >= args.chunks:
        raise ValueError(f"chunk_id must be between 0 and {args.chunks-1}")
    
    # Record start time
    start_time = time.time()
    
    # Initialize constraint generator
    generator = VanillaFairnessConstraintGenerator(
        model_path_prefix=args.model_path_prefix,
        llama_model=args.model,
        random_state=args.random_seed,
        verbose=True
    )
    
    # Load COMPAS data
    train_df = generator.load_compas_data(args.train_path)
    
    # Elicit fairness judgments for this chunk
    constraint_data = generator.elicit_fairness_judgments(
        train_df, 
        total_comparisons=args.total_comparisons,
        chunk_id=args.chunk_id,
        chunks=args.chunks
    )
    
    # Save constraint data for this chunk
    generator.save_constraint_data(
        constraint_data, 
        args.output,
        chunk_id=args.chunk_id
    )
    
    # Report timing
    elapsed_time = time.time() - start_time
    comparisons_per_chunk = args.total_comparisons // args.chunks
    print(f"\nConstraint generation completed in {elapsed_time:.2f} seconds")
    print(f"Processed chunk {args.chunk_id+1}/{args.chunks} with ~{comparisons_per_chunk} comparisons")
    print(f"Average time per comparison: {elapsed_time/comparisons_per_chunk:.2f} seconds")
    
if __name__ == "__main__":
    main()