#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified to use an external prompt management system for improved flexibility.
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

# Import our prompt manager
from prompt_manager import PromptManager
import sys
print("Arguments received:", sys.argv)

class LlamaRole:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class FairnessConstraintGenerator:

    def __init__(self, 
                 model_path_prefix: str = "../../models/",
                 llama_model: str = "llama3-8b-instruct",
                 prompt_config: str = "prompts/config.yaml",
                 random_state: int = 11, 
                 verbose: bool = True):

        self.model_path_prefix = model_path_prefix
        self.llama_model = llama_model
        self.random_state = random_state
        self.verbose = verbose
        
        self.prompt_manager = PromptManager(prompt_config)
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        random.seed(random_state)
        
        if verbose:
            print(f"Initializing with {llama_model} model (will be loaded when needed)")
        
        # initialize tokenizer
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

    def format_persona_prompt(self, persona_str):
        """Extracts key details from a persona string and formats it into an LLM-friendly 'You are' prompt."""
        
        # Extract attributes using regex
        age_match = re.search(r"age: (\d+)", persona_str)
        sex_match = re.search(r"sex: (.+)", persona_str)
        race_match = re.search(r"race: (.+)", persona_str)
        ancestry_match = re.search(r"ancestry: (.+)", persona_str)
        birthplace_match = re.search(r"place of birth: (.+)", persona_str)
        job_match = re.search(r"occupation category: (.+)", persona_str)
        lifestyle_match = re.search(r"lifestyle: (.+)", persona_str)
        hobbies_match = re.search(r"defining quirks: (.+)", persona_str)
        personal_time_match = re.search(r"personal time: (.+)", persona_str)
        personality_match = re.search(r"big five scores: (.+)", persona_str)
        political_match = re.search(r"political views: (.+)", persona_str)
        religion_match = re.search(r"religion: (.+)", persona_str)

        # Retrieve values with defaults
        age = age_match.group(1) if age_match else "Unknown age"
        sex = sex_match.group(1).lower() if sex_match else "Unknown gender"
        race = race_match.group(1) if race_match else "Unknown race"
        ancestry = ancestry_match.group(1) if ancestry_match else "Unknown ancestry"
        birthplace = birthplace_match.group(1) if birthplace_match else "Unknown birthplace"
        job = job_match.group(1).lower() if job_match else "unknown occupation"
        lifestyle = lifestyle_match.group(1).lower() if lifestyle_match else "an undefined lifestyle"
        hobbies = hobbies_match.group(1).lower() if hobbies_match else "no specific hobbies"
        personal_time = personal_time_match.group(1).lower() if personal_time_match else "unspecified activities"
        personality = personality_match.group(1).lower() if personality_match else "no defined personality traits"
        political = political_match.group(1).lower() if political_match else "no stated political alignment"
        religion = religion_match.group(1).lower() if religion_match else "no religious affiliation"

        # Construct the LLM persona prompt
        persona_prompt = (
            f"You are an {age}-year-old {race} {sex} of {ancestry} ancestry from {birthplace}. "
            f"You are a {job} who leads a {lifestyle}  lifestyle. "
            f"Hobbies: {hobbies} and {personal_time}. "
            f"You have the following personality traits: {personality}. "
            f"You align politically as {political} and are {religion}."
        )
        
        return persona_prompt

    def generate_unique_comparisons(self, train_df, personas_df, total_pairs=5000, evals_per_pair=10):
        """Generate comparison pairs that are evaluated by multiple personas."""
        if self.verbose:
            print(f"Generating {total_pairs} unique comparison pairs")
            print(f"Each pair will be evaluated by approximately {evals_per_pair} personas")
        
        all_ids = train_df['id'].values
        n_individuals = len(all_ids)
        original_indices = personas_df['original_index'].tolist()
        num_personas = len(original_indices)
        
        # Generate the pool of comparison pairs
        pair_pool = []
        pair_rng = random.Random(self.random_state)
        
        for comparison_id in range(total_pairs):
            idx1, idx2 = pair_rng.sample(range(n_individuals), 2)
            individual1_id = int(all_ids[idx1])
            individual2_id = int(all_ids[idx2])
            pair_pool.append({
                "comparison_id": comparison_id,
                "individual1_id": individual1_id,
                "individual2_id": individual2_id
            })
        
        # Distribute pairs to personas
        persona_comparisons = {}
        pairs_per_persona = (total_pairs * evals_per_pair) // num_personas
        
        for i, persona_id in enumerate(original_indices):
            # Use persona_id to seed the RNG for reproducibility
            persona_rng = random.Random(self.random_state + persona_id)
            # Sample pairs for this persona
            persona_pairs = persona_rng.sample(pair_pool, min(pairs_per_persona, len(pair_pool)))
            persona_comparisons[i] = persona_pairs
        
        return persona_comparisons
    def generate_unique_comparisons1(self, train_df, personas_df, pairs_per_persona=50):
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
    
    def extract_choice_and_explanation(self, response_text):
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
        
        # If no clear choice found in the last sentences, check the entire text
        if chosen_option is None:
            if "option 1" in response_text.lower() or "choose 1" in response_text.lower() or "choose option 1" in response_text.lower():
                chosen_option = 1
            elif "option 2" in response_text.lower() or "choose 2" in response_text.lower() or "choose option 2" in response_text.lower():
                chosen_option = 2
        
        # Look for binary-specific patterns
        if chosen_option is None:
            if "should be treated similarly" in response_text.lower() or "option 1" in response_text.lower():
                chosen_option = 1
            elif "acceptable to treat these individuals differently" in response_text.lower() or "option 2" in response_text.lower():
                chosen_option = 2
            elif "treated similarly" in response_text.lower():
                chosen_option = 1
            elif "treated differently" in response_text.lower():
                chosen_option = 2
        
        # If we still couldn't identify an option, try numeric patterns
        if chosen_option is None:
            # Check for standalone "1" or "2" with punctuation
            for pattern in [r'\b1\b', r'\b2\b']:
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
    
    def llama_query(self, prompt: List[Tuple[str, str]], max_tokens: int = 128, 
                   temperature: float = 0.8) -> str:
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
    
    def elicit_fairness_judgments(self, train_df, personas_df, pairs_per_persona=50, 
                            start_comparison=0, end_comparison=None,
                            use_personas=True, prompt_type="chain_of_thought",
                            output_path=None):
        """
        Elicit fairness judgments, querying each comparison pair twice (X vs Y and Y vs X).
        
        Args:
            train_df: Training data with individuals to compare
            personas_df: Dataframe of personas to use
            pairs_per_persona: Number of comparison pairs per persona
            start_comparison: Starting comparison index (for parallelization)
            end_comparison: Ending comparison index (for parallelization)
            use_personas: Whether to use personas in prompts
            prompt_type: Type of prompt to use (from config)
        """
        if self.verbose:
            print(f"Eliciting fairness judgments for {len(personas_df)} personas")
            print(f"Each persona will evaluate {pairs_per_persona} comparison pairs, each presented twice")
            print(f"Using personas: {use_personas}, Prompt type: {prompt_type}")
            if start_comparison > 0 or end_comparison is not None:
                print(f"Processing comparison range: {start_comparison} to {end_comparison or pairs_per_persona-1}")
        
        # Make sure LLaMA model is loaded
        self._load_llama_model()
        
        # Create a lookup for individuals by ID
        individuals_by_id = {row['id']: row for _, row in train_df.iterrows()}
        
        # Generate unique comparisons for each persona
        persona_comparisons = self.generate_unique_comparisons(
            train_df, 
            personas_df
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
                
                # Get the system prompt with or without persona
                if use_personas:
                    system_prompt = self.prompt_manager.get_system_prompt(
                        prompt_type=prompt_type,
                        persona=self.format_persona_prompt(persona_desc)
                    )
                else:
                    system_prompt = self.prompt_manager.get_system_prompt(
                        prompt_type="default" if prompt_type == "with_persona" else prompt_type
                    )
                
                # ----- Query 1: Normal order (X vs Y) -----
                query_text1, option_mapping = self.prompt_manager.create_fairness_query(
                    individual1, individual2, swap_positions=False
                )
                
                prompt_list1 = [
                    (LlamaRole.SYSTEM, system_prompt),
                    (LlamaRole.USER, query_text1)
                ]
                
                # Query the model for the normal order - increase max tokens for more thinking space
                response1 = self.llama_query(prompt_list1, max_tokens=512, temperature=0.7)
                
                # Inside elicit_fairness_judgments method, update this section to handle binary judgments:

                # Extract chosen option and reasoning
                chosen_option1, reasoning1 = self.extract_choice_and_explanation(response1)

                # Get judgment type from config
                judgment_type = self.prompt_manager.config.get("judgment_type", "three_option")

                # Map chosen option to judgment type based on judgment_type
                if chosen_option1 is not None:
                    if judgment_type == "binary":
                        # Binary judgment mapping
                        if 1 <= chosen_option1 <= 2:
                            # Option 1 = similar, Option 2 = different
                            judgment1 = "similar" if chosen_option1 == 1 else "different"
                        else:
                            judgment1 = "unknown"
                    else:
                        # Three-option judgment mapping
                        if 1 <= chosen_option1 <= 3:
                            judgment1 = option_mapping[chosen_option1]
                        else:
                            judgment1 = "unknown"
                else:
                    judgment1 = "unknown"

                # Process constraints based on judgment_type
                if judgment_type == "binary":
                    # For binary judgments (similar vs different)
                    if judgment1 == "similar":
                        # Both should be treated the same - add bi-directional constraints
                        constraints.append((individual1_id, individual2_id))
                        constraints.append((individual2_id, individual1_id))
                    # For "different", we don't add any constraints
                elif judgment_type == "three_option":
                    # For three-option judgments
                    if judgment1 == "similar":
                        # Both should be treated the same
                        constraints.append((individual1_id, individual2_id))
                        constraints.append((individual2_id, individual1_id))
                    elif judgment1 == "x_higher_than_y":
                        # X should be treated at least as well as Y
                        constraints.append((individual2_id, individual1_id))
                    elif judgment1 == "y_higher_than_x":
                        # Y should be treated at least as well as X
                        constraints.append((individual1_id, individual2_id))
                
                # Store the first judgment with individual data
                judgment_data1 = {
                    'persona_id': persona_id,
                    'persona': self.format_persona_prompt(persona_desc),
                    'comparison_id': comparison_id,
                    'order': 'normal',
                    'individual1_id': individual1_id,
                    'individual2_id': individual2_id,
                    'judgment': judgment1,
                    'chosen_option': chosen_option1,
                    'response': response1,
                    'reasoning': reasoning1,  # Store full chain of thought
                    'use_persona': use_personas,
                    'prompt_type': prompt_type
                }
                
                # Add demographic information 
                for field in ['sex', 'age', 'race', 'priors_count', 'juv_fel_count', 
                             'juv_misd_count', 'juv_other_count', 'c_charge_degree']:
                    if field in individual1:
                        judgment_data1[f'individual1_{field}'] = individual1[field]
                    if field in individual2:
                        judgment_data1[f'individual2_{field}'] = individual2[field]
                
                # ----- Query 2: Swapped order (Y vs X) -----
                query_text2, option_mapping = self.prompt_manager.create_fairness_query(
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
                    'persona': self.format_persona_prompt(persona_desc),
                    'comparison_id': comparison_id,
                    'order': 'swapped',
                    'individual1_id': individual2_id,  # Swapped
                    'individual2_id': individual1_id,  # Swapped
                    'judgment': judgment2,
                    'chosen_option': chosen_option2,
                    'response': response2,
                    'reasoning': reasoning2,  # Store full chain of thought
                    'use_persona': use_personas,
                    'prompt_type': prompt_type
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

        if output_path:
            self.save_results(all_judgments, output_path)
        
        # Return all data
        return {
            'constraints': persona_constraints,
            'judgments': all_judgments
        }

    def save_results(self, all_judgments, output_path):
        """Save all judgments to a CSV file."""
        import os
        import pandas as pd
        
        # Convert judgments to DataFrame
        df_judgments = pd.DataFrame(all_judgments)
        
        # Debugging: Print how many rows will be saved
        if self.verbose:
            print(f"Saving {len(df_judgments)} judgments to {output_path}")
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            if self.verbose:
                print(f"Creating directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Try writing the file
        try:
            df_judgments.to_csv(output_path, index=False)
            if self.verbose:
                print(f"Successfully saved results to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving output file: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Fairness Constraint Elicitation")
    
    # Add command line arguments
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--personas_path", type=str, required=True, help="Path to personas data")
    parser.add_argument("--output", type=str, required=True, help="Path to output file")
    parser.add_argument("--prompt_config", type=str, required=True, help="Path to prompt configuration")
    parser.add_argument("--pairs_per_persona", type=int, default=50, help="Number of comparison pairs per persona")
    parser.add_argument("--use_personas", type=str, default="True", help="Whether to use personas in prompts")
    parser.add_argument("--prompt_type", type=str, default="chain_of_thought", help="Type of prompt to use")
    parser.add_argument("--judgment_type", type=str, default="binary", help="Type of judgment (binary, ranking)")
    parser.add_argument("--persona_start", type=int, default=0, help="Starting persona index")
    parser.add_argument("--persona_end", type=int, default=None, help="Ending persona index")
    parser.add_argument("--comparison_start", type=int, default=0, help="Starting comparison index")
    parser.add_argument("--comparison_end", type=int, default=None, help="Ending comparison index")
    parser.add_argument("--verbose", type=str, default="True", help="Verbose output")
    
    args = parser.parse_args()
    
    # Convert string to boolean
    use_personas = args.use_personas.lower() == "true"
    verbose = args.verbose.lower() == "true"
    
    # Initialize generator
    generator = FairnessConstraintGenerator(
        prompt_config=args.prompt_config,
        verbose=verbose
    )
    
    # Load data
    train_df = generator.load_compas_data(args.train_path)
    personas_df = generator.load_personas(
        args.personas_path, 
        start_persona_index=args.persona_start,
        end_persona_index=args.persona_end
    )
    
    # Elicit fairness judgments
    result = generator.elicit_fairness_judgments(
        train_df=train_df,
        personas_df=personas_df,
        pairs_per_persona=args.pairs_per_persona,
        start_comparison=args.comparison_start,
        end_comparison=args.comparison_end,
        use_personas=use_personas,
        prompt_type=args.prompt_type,
        output_path=args.output
    )
    
    print(f"Elicitation complete. Processed {len(result['judgments'])} judgments.")

if __name__ == "__main__":
    main()