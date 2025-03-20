#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script processes the output CSV files and generates summary statistics and visualizations.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Tuple
from collections import Counter
import json

def load_results(results_file: str) -> pd.DataFrame:
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} results from {results_file}")
    
    return df

def calculate_summary_statistics(df: pd.DataFrame) -> Dict:

    # Judgment distribution
    judgment_counts = df['judgment'].value_counts().to_dict()
    judgment_percentages = (df['judgment'].value_counts(normalize=True) * 100).to_dict()
    
    # Position bias calculation (if order column exists)
    position_bias = {}
    if 'order' in df.columns:
        normal_order = df[df['order'] == 'normal']
        swapped_order = df[df['order'] == 'swapped']
        
        # Calculate judgment distribution for each order
        normal_counts = normal_order['judgment'].value_counts(normalize=True).to_dict()
        swapped_counts = swapped_order['judgment'].value_counts(normalize=True).to_dict()
        
        # Calculate differences
        all_judgments = set(normal_counts.keys()) | set(swapped_counts.keys())
        differences = {}
        for judgment in all_judgments:
            normal_pct = normal_counts.get(judgment, 0) * 100
            swapped_pct = swapped_counts.get(judgment, 0) * 100
            differences[judgment] = normal_pct - swapped_pct
        
        position_bias = {
            'normal_percentages': {k: v * 100 for k, v in normal_counts.items()},
            'swapped_percentages': {k: v * 100 for k, v in swapped_counts.items()},
            'differences': differences
        }
    
    # Calculate statistics by persona
    personas = df['persona_id'].unique()
    persona_stats = {}
    
    for persona_id in personas:
        persona_df = df[df['persona_id'] == persona_id]
        judgments = persona_df['judgment'].value_counts(normalize=True).to_dict()
        persona_stats[str(persona_id)] = {
            'count': len(persona_df),
            'judgment_percentages': {k: v * 100 for k, v in judgments.items()}
        }
    
    # Agreement analysis between personas
    agreement_by_comparison = {}
    if 'comparison_id' in df.columns:
        for comparison_id in df['comparison_id'].unique():
            comparison_df = df[(df['comparison_id'] == comparison_id) & (df['order'] == 'normal')]
            if len(comparison_df) > 1:
                judgments = comparison_df['judgment'].tolist()
                most_common = Counter(judgments).most_common(1)[0]
                agreement_rate = most_common[1] / len(judgments)
                agreement_by_comparison[str(comparison_id)] = {
                    'most_common_judgment': most_common[0],
                    'agreement_rate': agreement_rate
                }
    
    # Return all statistics
    return {
        'total_judgments': len(df),
        'unique_personas': len(personas),
        'judgment_counts': judgment_counts,
        'judgment_percentages': judgment_percentages,
        'position_bias': position_bias,
        'persona_stats': persona_stats,
        'agreement_by_comparison': agreement_by_comparison
    }

def plot_judgment_distribution(df: pd.DataFrame, output_dir: str):

    plt.figure(figsize=(10, 6))
    
    # Create judgment distribution plot
    judgments = df['judgment'].value_counts().sort_index()
    
    # Map judgment keys to more readable labels
    label_map = {
        'similar': 'Similar Treatment',
        'x_higher_than_y': 'X Higher Risk than Y',
        'y_higher_than_x': 'Y Higher Risk than X',
        'different': 'Different Treatment'
    }
    
    # Create better labels for the plot
    labels = [label_map.get(j, j) for j in judgments.index]
    
    # Plot it
    ax = sns.barplot(x=judgments.index, y=judgments.values)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    plt.title('Distribution of Fairness Judgments')
    plt.ylabel('Count')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'judgment_distribution.png'), dpi=300)
    print(f"Saved judgment distribution plot to {output_dir}/judgment_distribution.png")

def plot_position_bias(df: pd.DataFrame, output_dir: str):

    if 'order' not in df.columns:
        print("Cannot plot position bias: 'order' column not found")
        return
    
    # Get counts for each judgment by order
    order_judgment = pd.crosstab(df['order'], df['judgment'], normalize='index') * 100
    
    plt.figure(figsize=(12, 6))
    
    # Plot side by side bars
    order_judgment.plot(kind='bar', ax=plt.gca())
    
    plt.title('Position Bias in Fairness Judgments')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Presentation Order')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'position_bias.png'), dpi=300)
    print(f"Saved position bias plot to {output_dir}/position_bias.png")

def plot_persona_variability(df: pd.DataFrame, output_dir: str):

    # Get judgment percentages for each persona
    persona_judgments = df.groupby('persona_id')['judgment'].value_counts(normalize=True).unstack().fillna(0) * 100
    
    plt.figure(figsize=(14, 8))
    
    # Sort personas by the percentage of 'similar' judgments
    if 'similar' in persona_judgments.columns:
        persona_judgments = persona_judgments.sort_values(by='similar', ascending=False)
    
    # Plot as a heatmap
    sns.heatmap(persona_judgments, cmap='viridis', annot=False, cbar_kws={'label': 'Percentage (%)'})
    
    plt.title('Judgment Variation Across Personas')
    plt.ylabel('Persona ID')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'persona_variability.png'), dpi=300)
    print(f"Saved persona variability plot to {output_dir}/persona_variability.png")

def analyze_demographic_factors(df: pd.DataFrame, output_dir: str):

    # Check if demographic information is available
    demo_columns = [col for col in df.columns if any(col.startswith(f'individual{i}_') for i in [1, 2])]
    
    if not demo_columns:
        print("Cannot analyze demographic factors: No demographic columns found")
        return
    
    # Focus on race and gender if available
    race_cols = [col for col in demo_columns if 'race' in col]
    gender_cols = [col for col in demo_columns if 'sex' in col]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze race if available
    if race_cols:
        # Filter to normal order for consistency
        if 'order' in df.columns:
            race_df = df[df['order'] == 'normal']
        else:
            race_df = df
            
        # Create a new column for race comparison
        race_df['race_comparison'] = race_df.apply(
            lambda row: f"{row.get('individual1_race', 'Unknown')} vs {row.get('individual2_race', 'Unknown')}"
            if 'individual1_race' in row and 'individual2_race' in row else "Unknown",
            axis=1
        )
        
        # Get judgment distribution by race comparison
        race_judgment = pd.crosstab(
            race_df['race_comparison'], 
            race_df['judgment'], 
            normalize='index'
        ) * 100
        
        # Plot as a heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(race_judgment, cmap='viridis', annot=True, fmt='.1f', cbar_kws={'label': 'Percentage (%)'})
        
        plt.title('Judgment Distribution by Race Comparison')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'race_analysis.png'), dpi=300)
        print(f"Saved race analysis plot to {output_dir}/race_analysis.png")
    
    # Analyze gender if available
    if gender_cols:
        # Filter to normal order for consistency
        if 'order' in df.columns:
            gender_df = df[df['order'] == 'normal']
        else:
            gender_df = df
            
        # Create a new column for gender comparison
        gender_df['gender_comparison'] = gender_df.apply(
            lambda row: f"{row.get('individual1_sex', 'Unknown')} vs {row.get('individual2_sex', 'Unknown')}"
            if 'individual1_sex' in row and 'individual2_sex' in row else "Unknown",
            axis=1
        )
        
        # Get judgment distribution by gender comparison
        gender_judgment = pd.crosstab(
            gender_df['gender_comparison'], 
            gender_df['judgment'], 
            normalize='index'
        ) * 100
        
        # Plot as a heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(gender_judgment, cmap='viridis', annot=True, fmt='.1f', cbar_kws={'label': 'Percentage (%)'})
        
        plt.title('Judgment Distribution by Gender Comparison')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'gender_analysis.png'), dpi=300)
        print(f"Saved gender analysis plot to {output_dir}/gender_analysis.png")

def main():
    parser = argparse.ArgumentParser(description="Analyze fairness elicitation results")
    parser.add_argument("--input", type=str, required=True, help="Path to the CSV file with results")
    parser.add_argument("--output_dir", type=str, default="analysis", help="Directory to save analysis results")
    args = parser.parse_args()
    
    # Load results
    results_df = load_results(args.input)
    
    # Calculate summary statistics
    statistics = calculate_summary_statistics(results_df)
    
    # Save statistics to a JSON file
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'statistics.json'), 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"Saved statistics to {args.output_dir}/statistics.json")
    
    # Create plots
    plot_judgment_distribution(results_df, args.output_dir)
    plot_position_bias(results_df, args.output_dir)
    plot_persona_variability(results_df, args.output_dir)
    analyze_demographic_factors(results_df, args.output_dir)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()