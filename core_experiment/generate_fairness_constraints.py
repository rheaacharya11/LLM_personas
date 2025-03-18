#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate fairness constraints for Jung et al.'s Algorithmic Fairness Elicitation framework.
Modified so that each persona gets its own unique set of 50 comparison pairs.
"""

import os
import numpy as np
import pandas as pd
import random
import json
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
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
    Class for generating fairness constraints from personas through LLaMA model judgments.
    """
    
    def __init__(self, 
                model_path_prefix: str = "../models/",
                llama_model: str = "llama3-8b-instruct",
                random_state: int = 42, 
                verbose: bool = True):
        """
        Initialize the fairness constraint generator.
        """
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
        """
        Load personas from the specified path with range support.
        
        Args:
            personas_path: Path to the personas parquet file
            start_persona_index: Starting index of personas to process
            end_persona_index: Ending index of personas to process (inclusive)
            
        Returns:
            personas_df: DataFrame with persona information for the specified range
        """
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
            personas_df = personas_df.iloc[start_persona_index:end_persona_index+1].reset_index(drop=True)
            
            if self.verbose:
                print(f"Loaded {len(personas_df)} personas (index range: {start_persona_index} to {end_persona_index})")
            
            return personas_df
            
        except Exception as e:
            raise RuntimeError(f"Error loading personas: {e}")
    
    def generate_unique_comparisons(self, train_df, num_personas, pairs_per_persona=50):
        """
        Generate unique random pairwise comparisons for each persona.
        Each persona gets its own different set of pairs.
        
        Args:
            train_df: Training dataframe with individual data
            num_personas: Number of personas to generate comparisons for
            pairs_per_persona: Number of comparison pairs per persona
            
        Returns:
            Dictionary mapping persona IDs to lists of comparison pairs
        """
        all_ids = train_df['id'].values.tolist()
    n_individuals = len(all_ids)
    persona_comparisons = {}
    
    # Create a single master RNG
    master_rng = random.Random(self.random_state)
    
    # For each persona, generate a unique set of pairs using a different shuffle of IDs
    for persona_id in range(num_personas):
        # Create a shuffled copy of all IDs for this persona
        shuffled_ids = all_ids.copy()
        # Use a different shuffle seed for each persona
        shuffle_seed = self.random_state * 10000 + persona_id * 997  # Large prime factor to avoid correlation
        persona_rng = random.Random(shuffle_seed)
        persona_rng.shuffle(shuffled_ids)
        
        # Sample pairs using this uniquely shuffled list
        comparisons = []
        comparison_count = 0
        
        # Create pairs by taking elements at different offsets to ensure variety
        offset = persona_id % 10 + 5  # Different offset for each persona (5-14)
        
        for i in range(n_individuals):
            j = (i + offset) % n_individuals
            if i != j and comparison_count < pairs_per_persona:
                individual1_id = int(shuffled_ids[i])
                individual2_id = int(shuffled_ids[j])
                
                comparisons.append({
                    "comparison_id": comparison_count,
                    "individual1_id": individual1_id,
                    "individual2_id": individual2_id
                })
                comparison_count += 1
                
                # If we need more pairs, take another offset
                if comparison_count < pairs_per_persona and i + offset*2 < n_individuals:
                    j2 = (i + offset*2) % n_individuals
                    if i != j2 and j != j2:
                        individual2_id = int(shuffled_ids[j2])
                        comparisons.append({
                            "comparison_id": comparison_count,
                            "individual1_id": individual1_id,
                            "individual2_id": individual2_id
                        })
                        comparison_count += 1
        
        # If we still need more pairs, generate them randomly
        while comparison_count < pairs_per_persona:
            i, j = persona_rng.sample(range(n_individuals), 2)
            individual1_id = int(shuffled_ids[i])
            individual2_id = int(shuffled_ids[j])
            
            comparisons.append({
                "comparison_id": comparison_count,
                "individual1_id": individual1_id,
                "individual2_id": individual2_id
            })
            comparison_count += 1
        
        persona_comparisons[persona_id] = comparisons
    
    return persona_comparisons
        
    def create_balanced_fairness_query(self, individual1, individual2, randomize_options=True):
        """Create a balanced multiple-choice query about fairness in recidivism prediction."""
        intro = """
The criminal justice system uses risk assessment algorithms to predict which defendants are likely to reoffend if released. These recidivism predictions influence important decisions about bail, sentencing, and parole. I'm going to show you information about two individuals and ask you to consider how their risk should be assessed from a fairness perspective.
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
        
        options = [
            "Person X should have a higher recidivism risk score than Person Y",
            "Person Y should have a higher recidivism risk score than Person X",
            "Both should have approximately equal risk scores"
        ]
        
        # Map the options to choice types
        option_mapping = {
            0: "x_higher_than_y",
            1: "y_higher_than_x",
            2: "similar"
        }
        
        # Randomize options if requested
        option_order = [0, 1, 2]  # Default order
        if randomize_options:
            random.shuffle(option_order)
        
        # Create the ordered list of options
        ordered_options = [options[i] for i in option_order]
        
        query = (
            f"{intro}\n\n"
            f"{comparison}\n\n"
            f"Question: In your view, which statement is most accurate?\n"
            f"1. {ordered_options[0]}\n"
            f"2. {ordered_options[1]}\n"
            f"3. {ordered_options[2]}\n\n"
            f"Choose one option (1, 2, or 3) and briefly explain why in a few words."
        )
        
        return query, option_order, option_mapping
    
    def extract_choice_and_explanation(self, response_text):
        """Extract the chosen option and explanation from the model's response."""
        response_text = response_text.strip()
        
        # Default values
        chosen_option = None
        explanation = None
        
        # Check for a number at the beginning of the response
        if response_text.startswith("1") or response_text.startswith("Option 1"):
            chosen_option = 1
        elif response_text.startswith("2") or response_text.startswith("Option 2"):
            chosen_option = 2
        elif response_text.startswith("3") or response_text.startswith("Option 3"):
            chosen_option = 3
        
        # If no number at the beginning, look for patterns in the text
        if chosen_option is None:
            if "option 1" in response_text.lower() or "first option" in response_text.lower():
                chosen_option = 1
            elif "option 2" in response_text.lower() or "second option" in response_text.lower():
                chosen_option = 2
            elif "option 3" in response_text.lower() or "third option" in response_text.lower():
                chosen_option = 3
        
        # Fall back to looking for the statement itself
        if chosen_option is None:
            response_lower = response_text.lower()
            if "x should have a higher" in response_lower or "person x should have a higher" in response_lower:
                chosen_option = 1  # This is a simplified fallback - may need refinement
            elif "y should have a higher" in response_lower or "person y should have a higher" in response_lower:
                chosen_option = 2  # This is a simplified fallback - may need refinement
            elif "equal" in response_lower or "same" in response_lower or "similar" in response_lower:
                chosen_option = 3  # This is a simplified fallback - may need refinement
        
        # Extract explanation - everything after the option choice
        if chosen_option is not None:
            # Find where the explanation starts after the option number
            option_str = str(chosen_option)
            if option_str in response_text:
                explanation_start = response_text.find(option_str) + len(option_str)
                explanation = response_text[explanation_start:].strip()
                
                # Remove common prefixes
                prefixes = [".", ":", "-", ")"]
                for prefix in prefixes:
                    if explanation.startswith(prefix):
                        explanation = explanation[1:].strip()
        else:
            # If we couldn't identify an option, use the whole response as the explanation
            explanation = response_text
        
        return chosen_option, explanation
    
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
        
        # Format the prompt using Llama 3 special tokens
        formatted_prompt = self.format_llama3_prompt(prompt)
        
        # Tokenize input and ensure attention mask is properly set
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
        
        return assistant_response
    
    def elicit_fairness_judgments(self, train_df, personas_df, pairs_per_persona=50, start_comparison=0, end_comparison=None):
        """
        Elicit fairness judgments from LLaMA model using the balanced 3-option approach.
        Each persona gets its own set of 50 unique comparison pairs.
        
        Args:
            train_df: Training dataframe with individual data
            personas_df: Dataframe with persona descriptions for the range to process
            pairs_per_persona: Number of pairs to generate per persona
            start_comparison: Starting comparison index to process (for parallelization)
            end_comparison: Ending comparison index to process (for parallelization)
            
        Returns:
            Dictionary with constraints and judgment data
        """
        if self.verbose:
            print(f"Eliciting fairness judgments for {len(personas_df)} personas")
            print(f"Each persona will evaluate {pairs_per_persona} unique comparison pairs")
            if start_comparison > 0 or end_comparison is not None:
                print(f"Processing comparison range: {start_comparison} to {end_comparison or pairs_per_persona-1}")
        
        # Make sure LLaMA model is loaded
        self._load_llama_model()
        
        # Create a lookup for individuals by ID
        individuals_by_id = {row['id']: row for _, row in train_df.iterrows()}
        
        # Generate unique comparisons for each persona
        persona_comparisons = self.generate_unique_comparisons(
            train_df, 
            len(personas_df), 
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
            persona_id = persona_row.name if hasattr(persona_row, 'name') else persona_idx
            
            # Get persona description
            if 'persona' in personas_df.columns:
                persona_desc = persona_row['persona']
            else:
                persona_desc = f"Persona {persona_id}"
            
            # Initialize constraints for this persona
            constraints = []
            persona_judgments = []
            
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
                
                # Create balanced query with 3 options
                query_text, option_order, option_mapping = self.create_balanced_fairness_query(
                    individual1, individual2, randomize_options=True
                )
                
                # Create system prompt with persona
                system_prompt = (f"You are a helpful AI assistant with the persona of {persona_desc}. "
                                f"Answer the question by selecting one option (1, 2, or 3) and then "
                                f"providing a brief explanation of your reasoning in a few words.")
                
                # Create the prompt list
                prompt_list = [
                    (LlamaRole.SYSTEM, system_prompt),
                    (LlamaRole.USER, query_text)
                ]
                
                # Query LLaMA model
                response = self.llama_query(prompt_list, max_tokens=256, temperature=0.7)
                
                # Extract chosen option and explanation
                chosen_option, explanation = self.extract_choice_and_explanation(response)
                
                # Map chosen option to judgment type
                if chosen_option is not None and 1 <= chosen_option <= 3:
                    # Map the chosen option to the original option index
                    original_option_index = option_order[chosen_option - 1]
                    judgment = option_mapping[original_option_index]
                else:
                    judgment = "unknown"
                
                # Store the judgment
                judgment_data = {
                    'persona_id': persona_id,
                    'comparison_id': comparison_id,
                    'individual1_id': individual1_id,
                    'individual2_id': individual2_id,
                    'judgment': judgment,
                    'chosen_option': chosen_option,
                    'option_order': option_order,
                    'response': response,
                    'explanation': explanation
                }
                persona_judgments.append(judgment_data)
                all_judgments.append(judgment_data)
                
                # Convert judgment to constraints
                if judgment == "similar":
                    # Both directions - they should be treated the same
                    constraints.append((individual1_id, individual2_id))
                    constraints.append((individual2_id, individual1_id))
                    
                elif judgment == "x_higher_than_y":
                    # X should be treated at least as well as Y
                    constraints.append((individual2_id, individual1_id))
                    
                elif judgment == "y_higher_than_x":
                    # Y should be treated at least as well as X
                    constraints.append((individual1_id, individual2_id))
                
                # Update progress bar
                progress_bar.update(1)
            
            # Store constraints for this persona
            persona_constraints[persona_id] = constraints
        
        progress_bar.close()
        
        # Convert judgment counts into a summary
        judgment_counts = {
            'similar': len([j for j in all_judgments if j['judgment'] == 'similar']),
            'x_higher_than_y': len([j for j in all_judgments if j['judgment'] == 'x_higher_than_y']),
            'y_higher_than_x': len([j for j in all_judgments if j['judgment'] == 'y_higher_than_x']),
            'unknown': len([j for j in all_judgments if j['judgment'] == 'unknown'])
        }
        
        if self.verbose:
            print("\nJudgment distribution:")
            for judgment, count in judgment_counts.items():
                percentage = count / len(all_judgments) * 100 if all_judgments else 0
                print(f"  - {judgment}: {count} ({percentage:.1f}%)")
            
            # Compute constraint statistics
            total_constraints = sum(len(constraints) for constraints in persona_constraints.values())
            avg_constraints = total_constraints / len(persona_constraints) if persona_constraints else 0
            print(f"\nGenerated {total_constraints} total constraints")
            print(f"Average constraints per persona: {avg_constraints:.1f}")
        
        # Return both constraints and judgment data
        return {
            'constraints': persona_constraints,
            'judgments': all_judgments,
            'judgment_counts': judgment_counts
        }
    
    def save_constraint_data(self, constraint_data, output_file, append=False):
        """
        Save generated constraint data to a specified file.
        
        Args:
            constraint_data: Dictionary with constraints and judgment data
            output_file: Path to save the data
            append: Whether to append to existing file (for chunked processing)
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save judgments to a CSV file for easier chunked processing
        if 'judgments' in constraint_data and constraint_data['judgments']:
            # Extract all fields except nested ones
            judgments_flat = []
            for j in constraint_data['judgments']:
                flat_j = {
                    'persona_id': j['persona_id'],
                    'comparison_id': j['comparison_id'],
                    'individual1_id': j['individual1_id'],
                    'individual2_id': j['individual2_id'],
                    'judgment': j['judgment'],
                    'chosen_option': j['chosen_option'] if 'chosen_option' in j else None,
                    'explanation': j['explanation'] if 'explanation' in j else None,
                    'response': j['response']
                }
                
                # Stringify option_order to include it in the CSV
                if 'option_order' in j:
                    flat_j['option_order'] = json.dumps(j['option_order'])
                    
                judgments_flat.append(flat_j)
            
            # Convert to DataFrame for easier CSV handling
            judgments_df = pd.DataFrame(judgments_flat)
            
            # Write to CSV
            mode = 'a' if append else 'w'
            header = not append
            judgments_df.to_csv(output_file, mode=mode, header=header, index=False)
        
        if self.verbose:
            print(f"\nSaved {len(constraint_data['judgments'])} judgments to {output_file}")


def main():
    """Main function to run the constraint generation process."""
    parser = argparse.ArgumentParser(description="Generate fairness constraints using personas with unique comparison pairs")
    
    # Data paths
    parser.add_argument("--train_path", type=str, default="data/compas_train.parquet",
                       help="Path to training data")
    parser.add_argument("--personas_path", type=str, default="data/unique_personas.parquet",
                       help="Path to personas data")
    parser.add_argument("--output", type=str, default="results/core_experiment/chunked_outputs/fairness_judgments.csv",
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
                       help="Number of comparison pairs per persona")
    
    # Model parameters
    parser.add_argument("--model_path_prefix", type=str, default="../models/",
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
    
    # Elicit fairness judgments with unique comparison pairs for each persona
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