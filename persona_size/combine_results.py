#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to combine all the chunked output files into a single CSV for analysis.
Run this after all SLURM jobs have completed.
"""

import os
import pandas as pd
import glob
import argparse
from datetime import datetime

def combine_results(input_pattern, output_file):
    """
    Combine all CSV files matching the input pattern into a single output file.
    
    Args:
        input_pattern: Glob pattern for input files (e.g., "results/chunked_outputs/fixed_*.csv")
        output_file: Path to combined output file
    """
    print(f"Searching for files matching pattern: {input_pattern}")
    
    # Find all matching files
    matching_files = glob.glob(input_pattern)
    if not matching_files:
        print(f"No files found matching pattern: {input_pattern}")
        return
    
    print(f"Found {len(matching_files)} files to combine")
    
    # Initialize an empty DataFrame to store the combined results
    combined_df = pd.DataFrame()
    
    # Process each file
    for i, file_path in enumerate(matching_files):
        try:
            print(f"Processing file {i+1}/{len(matching_files)}: {file_path}")
            df = pd.read_csv(file_path)
            print(f"  - Contains {len(df)} rows")
            
            # Append to the combined DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"  - Error processing file: {e}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the combined DataFrame
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nCombined results summary:")
    print(f"  - Total input files: {len(matching_files)}")
    print(f"  - Total rows in combined file: {len(combined_df)}")
    print(f"  - Number of unique personas: {combined_df['persona_id'].nunique()}")
    print(f"  - Number of unique comparisons: {combined_df['comparison_id'].nunique()}")
    print(f"  - Combined file saved to: {output_file}")
    
    # Check for completeness
    total_personas = combined_df['persona_id'].nunique()
    total_comparisons = combined_df['comparison_id'].nunique()
    expected_rows = total_personas * total_comparisons
    actual_rows = len(combined_df)
    
    if actual_rows < expected_rows:
        print(f"\nWARNING: Combined file appears incomplete")
        print(f"  - Expected {expected_rows} rows (personas: {total_personas} × comparisons: {total_comparisons})")
        print(f"  - Actual rows: {actual_rows}")
        print(f"  - Missing {expected_rows - actual_rows} rows")
        
        # Find missing combinations
        all_combinations = set((p, c) for p in range(total_personas) for c in range(total_comparisons))
        existing_combinations = set(zip(combined_df['persona_id'], combined_df['comparison_id']))
        missing_combinations = all_combinations - existing_combinations
        
        print(f"  - Number of missing combinations: {len(missing_combinations)}")
        
        if len(missing_combinations) < 100:
            print("  - Missing combinations (persona_id, comparison_id):")
            for combo in sorted(list(missing_combinations)[:100]):
                print(f"    - {combo}")
        else:
            print("  - First 100 missing combinations (persona_id, comparison_id):")
            for combo in sorted(list(missing_combinations)[:100]):
                print(f"    - {combo}")
    else:
        print(f"\nSuccess! Combined file is complete with all expected combinations.")

def analyze_results(file_path):
    """
    Perform basic analysis on the combined results file.
    
    Args:
        file_path: Path to the combined results CSV file
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"\nAnalyzing results from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Basic statistics
    total_evaluations = len(df)
    num_personas = df['persona_id'].nunique()
    num_comparisons = df['comparison_id'].nunique()
    
    print(f"Basic Statistics:")
    print(f"  - Total evaluations: {total_evaluations}")
    print(f"  - Number of personas: {num_personas}")
    print(f"  - Number of comparisons: {num_comparisons}")
    
    # Choice distribution
    choice_counts = df['choice_type'].value_counts()
    print(f"\nChoice Distribution:")
    for choice, count in choice_counts.items():
        print(f"  - {choice}: {count} ({count/total_evaluations*100:.1f}%)")
    
    # Agreement analysis
    print(f"\nAgreement Analysis:")
    
    # For each comparison, how many personas agree?
    comparison_agreement = {}
    for comp_id in df['comparison_id'].unique():
        comp_df = df[df['comparison_id'] == comp_id]
        most_common_choice = comp_df['choice_type'].value_counts().idxmax()
        agreement_rate = comp_df[comp_df['choice_type'] == most_common_choice].shape[0] / comp_df.shape[0]
        comparison_agreement[comp_id] = agreement_rate
    
    avg_agreement = sum(comparison_agreement.values()) / len(comparison_agreement)
    print(f"  - Average agreement rate across comparisons: {avg_agreement*100:.1f}%")
    
    # Identify comparisons with highest and lowest agreement
    sorted_agreement = sorted(comparison_agreement.items(), key=lambda x: x[1])
    
    print(f"  - 5 comparisons with lowest agreement:")
    for comp_id, agreement in sorted_agreement[:5]:
        print(f"    - Comparison {comp_id}: {agreement*100:.1f}% agreement")
    
    print(f"  - 5 comparisons with highest agreement:")
    for comp_id, agreement in sorted_agreement[-5:]:
        print(f"    - Comparison {comp_id}: {agreement*100:.1f}% agreement")
    
    # Per-persona analysis
    print(f"\nPer-Persona Analysis:")
    
    # Calculate how often each persona agrees with the majority
    persona_agreement_with_majority = {}
    
    for persona_id in df['persona_id'].unique():
        agreements = 0
        total = 0
        
        for comp_id in df['comparison_id'].unique():
            # Get majority choice for this comparison
            comp_df = df[df['comparison_id'] == comp_id]
            majority_choice = comp_df['choice_type'].value_counts().idxmax()
            
            # Get this persona's choice
            persona_choice = df[(df['persona_id'] == persona_id) & (df['comparison_id'] == comp_id)]['choice_type'].values
            
            if len(persona_choice) > 0 and persona_choice[0] == majority_choice:
                agreements += 1
            
            total += 1
        
        agreement_rate = agreements / total if total > 0 else 0
        persona_agreement_with_majority[persona_id] = agreement_rate
    
    avg_persona_agreement = sum(persona_agreement_with_majority.values()) / len(persona_agreement_with_majority)
    print(f"  - Average persona agreement with majority: {avg_persona_agreement*100:.1f}%")
    
    # Identify personas with lowest and highest agreement with majority
    sorted_persona_agreement = sorted(persona_agreement_with_majority.items(), key=lambda x: x[1])
    
    print(f"  - 5 personas with lowest agreement with majority:")
    for persona_id, agreement in sorted_persona_agreement[:5]:
        print(f"    - Persona {persona_id}: {agreement*100:.1f}% agreement with majority")
    
    print(f"  - 5 personas with highest agreement with majority:")
    for persona_id, agreement in sorted_persona_agreement[-5:]:
        print(f"    - Persona {persona_id}: {agreement*100:.1f}% agreement with majority")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine and analyze parallel experiment results")
    parser.add_argument("--input_pattern", default="results/persona_size/chunked_outputs/*.csv", 
                        help="Glob pattern for input CSV files")
    parser.add_argument("--output", default="results/persona_size/combined_results.csv", 
                        help="Path for combined output file")
    parser.add_argument("--analyze", action="store_true", 
                        help="Perform basic analysis after combining")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only perform analysis on an existing combined file")
    parser.add_argument("--input_file", default=None,
                        help="Path to a combined file for analysis (used with --analyze_only)")

    
    args = parser.parse_args()
    
    print(f"Starting results combination at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    combine_results(args.input_pattern, args.output)
    if args.analyze_only:
        if not args.input_file or not os.path.exists(args.input_file):
            print(f"Error: With --analyze_only, you must specify a valid --input_file")
            sys.exit(1)
        
        print(f"Analyzing existing combined file: {args.input_file}")
        analyze_results(args.input_file)
    else:
        # Existing combining code
        combine_results(args.input_pattern, args.output)
        
        if args.analyze:
            analyze_results(args.output)
  
    
    print(f"Process complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")