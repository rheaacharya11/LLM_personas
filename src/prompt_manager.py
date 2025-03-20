#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt Manager for fairness elicitation experiments
"""
import os
import yaml
import json
from typing import Dict, List, Tuple, Optional, Any, Union


class PromptManager:
    
    def __init__(self, config_path: str = "prompts/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        # Error Tracking
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Prompt configuration file not found: {config_path}")
        
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if file_ext in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif file_ext == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_ext}")
            
            return config
        except Exception as e:
            raise RuntimeError(f"Error loading prompt configuration: {e}")
    
    def get_system_prompt(self, prompt_type: str = "default", **kwargs) -> str:
        """
        Get a formatted system prompt of the specified type.
        
        Args:
            prompt_type: Type of system prompt to use (from config)
            **kwargs: Format variables to substitute in the template
            
        Returns:
            Formatted system prompt
        """
        if prompt_type not in self.config.get("system_prompts", {}):
            raise ValueError(f"Unknown system prompt type: {prompt_type}")
        
        prompt_template = self.config["system_prompts"][prompt_type]
        
        # Special handling for persona prefix
        if "persona" in kwargs and "persona_prefix" not in kwargs:
            kwargs["persona_prefix"] = f"with the persona of {kwargs['persona']}"
        elif "persona_prefix" not in kwargs:
            kwargs["persona_prefix"] = ""
            
        # Format the prompt template with provided kwargs
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required format variable in prompt template: {e}")
    
    def get_user_prompt(self, prompt_type: str = "standard_query", **kwargs) -> str:
        """
        Get a formatted user prompt of the specified type.
        
        Args:
            prompt_type: Type of user prompt to use (from config)
            **kwargs: Format variables to substitute in the template
            
        Returns:
            Formatted user prompt
        """
        if prompt_type not in self.config.get("user_prompts", {}):
            raise ValueError(f"Unknown user prompt type: {prompt_type}")
        
        prompt_template = self.config["user_prompts"][prompt_type]
        
        # Automatically add intro text if not provided
        if "intro" not in kwargs and "intro_text" in self.config:
            kwargs["intro"] = self.config["intro_text"]
            
        # Automatically add options if not provided
        if all(k not in kwargs for k in ["option1", "option2", "option3"]) and "comparison_options" in self.config:
            options = self.config["comparison_options"]
            kwargs["option1"] = next((opt["text"] for opt in options if opt["key"] == "x_higher_than_y"), "")
            kwargs["option2"] = next((opt["text"] for opt in options if opt["key"] == "y_higher_than_x"), "")
            kwargs["option3"] = next((opt["text"] for opt in options if opt["key"] == "similar"), "")
            
        # Format the prompt template with provided kwargs
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required format variable in prompt template: {e}")
    
    def get_option_mapping(self) -> Dict[int, str]:
        """
        Get mapping from option numbers to judgment types.
        
        Returns:
            Dictionary mapping option numbers (1,2,3) to judgment types
        """
        options = self.config.get("comparison_options", [])
        mapping = {}
        
        # Check judgment type
        judgment_type = self.config.get("judgment_type", "three_option")
        
        if judgment_type == "binary":
            # For binary judgments, map option 1 to "similar" and option 2 to "different"
            for i, option in enumerate(options, start=1):
                mapping[i] = option["key"]
        else:
            # For three-option judgments, use position-based mapping
            for i, option in enumerate(options, start=1):
                mapping[i] = option["key"]
            
        return mapping
    
    def create_fairness_query(self, individual1: Dict, individual2: Dict, swap_positions: bool = False) -> Tuple[str, Dict[int, str]]:
        """
        Create a formatted fairness query for comparing two individuals.
        
        Args:
            individual1: Data for the first individual
            individual2: Data for the second individual
            swap_positions: Whether to swap the individuals' positions
            
        Returns:
            Tuple of (formatted query, option mapping)
        """
        # Swap individuals if requested
        if swap_positions:
            individual1, individual2 = individual2, individual1
            
        # Convert charge degree
        def convert_charge_degree(charge_degree):
            if isinstance(charge_degree, str):
                if charge_degree.startswith('F'):
                    return f"Felony"
                elif charge_degree.startswith('M'):
                    return f"Misdemeanor"
            return charge_degree
        
        charge1 = convert_charge_degree(individual1.get('c_charge_degree', ''))
        charge2 = convert_charge_degree(individual2.get('c_charge_degree', ''))
        
        # Create the comparison text
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
        
        # Get the user prompt with this comparison
        query = self.get_user_prompt(comparison=comparison)
        
        # Get the option mapping
        option_mapping = self.get_option_mapping()
        
        return query, option_mapping