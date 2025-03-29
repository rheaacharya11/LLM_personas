#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to generate fixed comparison pairs from COMPAS data.
"""

import pandas as pd
import numpy as np
import random
import os
import json
import argparse

def load_precleaned_data():

    train_path = "../data/processed/compas_train.parquet"
    test_path = "../data/processed/compas_test.parquet"
    print(f"Loading pre-cleaned data from {train_path} and {test_path}")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print(f"Loaded {len(train_df)} training examples and {len(test_df)} test examples")
    
    return train_df, test_df

def generate_fixed_comparisons(df, num_comparisons=1000, output_file="final_fixed_comparisons.json", random_state=11):
    """
    randomly sample 
    """
    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Create pairs
    comparisons = []
    indices_used = set()  # Track which indices we've used to avoid duplicates
    
    for i in range(num_comparisons):
        # Retry up to 10 times to get valid, non-duplicate pairs
        for attempt in range(10):
            # Get two random indices
            idx1, idx2 = random.sample(range(len(df)), 2)
            
            # Skip if we've used this pair before
            if (idx1, idx2) in indices_used or (idx2, idx1) in indices_used:
                continue
                
            # Get the two individuals
            individual1 = df.iloc[idx1]
            individual2 = df.iloc[idx2]
            
            # Add to tracking set
            indices_used.add((idx1, idx2))
            
            # Store the actual row indices as well as IDs for robustness
            comparisons.append({
                "comparison_id": i,
                "individual1_id": int(individual1['id']),
                "individual2_id": int(individual2['id']),
                "individual1_index": idx1,
                "individual2_index": idx2
            })
            break
    
    # Create output directory if it doesn't exist
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(comparisons, f, indent=2)
    
    print(f"Generated and saved {len(comparisons)} fixed comparisons to {output_file}")
    return comparisons

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fixed comparison pairs from COMPAS data")
    parser.add_argument("--comparisons", type=int, default=1000, help="Number of comparisons to generate")
    parser.add_argument("--output", default="final_fixed_comparisons.json", help="Output file path")
    parser.add_argument("--random_state", type=int, default=11, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    print(f"Generating {args.comparisons} fixed comparison pairs from COMPAS data")
    
    # Load pre-cleaned data instead of cleaning again
    train_df, _ = load_precleaned_data()
    
    # Generate fixed comparisons from training data
    generate_fixed_comparisons(
        df=train_df, 
        num_comparisons=args.comparisons, 
        output_file=args.output, 
        random_state=args.random_state
    )
    
    print("Done!")