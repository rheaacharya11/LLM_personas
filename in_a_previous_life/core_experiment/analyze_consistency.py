import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_combine_results(input_pattern):
    """Load and combine all CSV files matching the pattern."""
    print(f"Loading files matching pattern: {input_pattern}")
    files = glob.glob(input_pattern)
    
    if not files:
        print(f"No files found matching pattern: {input_pattern}")
        return None
    
    print(f"Found {len(files)} files to combine")
    
    # Combine all files
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataframe has {len(combined_df)} rows")
    
    return combined_df

def analyze_consistency(df):
    """Analyze consistency in judgments between normal and swapped order."""
    # Create a unique identifier for each comparison and persona combination
    df['pair_id'] = df['persona_id'].astype(str) + '_' + df['comparison_id'].astype(str)
    
    # Find all pairs where we have both normal and swapped orders
    pair_counts = df.groupby('pair_id')['order'].nunique()
    complete_pairs = pair_counts[pair_counts == 2].index.tolist()
    
    print(f"Found {len(complete_pairs)} complete pairs with both normal and swapped judgments")
    
    # Filter to pairs with both orders
    complete_df = df[df['pair_id'].isin(complete_pairs)].copy()
    
    # For each pair, determine if judgments are consistent
    results = []
    
    for pair_id in complete_pairs:
        pair_data = complete_df[complete_df['pair_id'] == pair_id].sort_values('order')
        
        if len(pair_data) != 2:
            continue  # Skip if we don't have exactly 2 entries
            
        normal_judgment = pair_data.iloc[0]['judgment']
        swapped_judgment = pair_data.iloc[1]['judgment']
        
        # Check consistency
        is_consistent = False
        
        # Case 1: Both judgments are "similar"
        if normal_judgment == 'similar' and swapped_judgment == 'similar':
            is_consistent = True
            consistency_type = "both_similar"
        
        # Case 2: x_higher_than_y in normal order, y_higher_than_x in swapped order
        elif normal_judgment == 'x_higher_than_y' and swapped_judgment == 'y_higher_than_x':
            is_consistent = True
            consistency_type = "properly_flipped"
        
        # Case 3: y_higher_than_x in normal order, x_higher_than_y in swapped order
        elif normal_judgment == 'y_higher_than_x' and swapped_judgment == 'x_higher_than_y':
            is_consistent = True
            consistency_type = "properly_flipped"
        
        # All other cases are inconsistent
        else:
            is_consistent = False
            consistency_type = "inconsistent"
        
        # Store results
        results.append({
            'pair_id': pair_id,
            'persona_id': int(pair_id.split('_')[0]),
            'comparison_id': int(pair_id.split('_')[1]),
            'normal_judgment': normal_judgment,
            'swapped_judgment': swapped_judgment,
            'is_consistent': is_consistent,
            'consistency_type': consistency_type
        })
    
    # Convert results to dataframe
    results_df = pd.DataFrame(results)
    
    # Calculate overall consistency rate
    consistency_rate = results_df['is_consistent'].mean() * 100
    print(f"Overall consistency rate: {consistency_rate:.2f}%")
    
    # Calculate consistency by type
    consistency_types = results_df['consistency_type'].value_counts(normalize=True) * 100
    print("\nConsistency types:")
    for type_name, percentage in consistency_types.items():
        print(f"  - {type_name}: {percentage:.2f}%")
    
    return results_df, consistency_rate, consistency_types

def analyze_by_demographics(df, results_df):
    """Analyze consistency rates by demographic factors."""
    # Merge results with demographic information
    # We can use the first occurrence of each pair (normal order)
    demographics = df.drop_duplicates('pair_id')[['pair_id', 'individual1_sex', 'individual2_sex', 
                                                 'individual1_race', 'individual2_race',
                                                 'individual1_age', 'individual2_age',
                                                 'individual1_priors_count', 'individual2_priors_count']]
    
    analysis_df = results_df.merge(demographics, on='pair_id', how='left')
    
    # Create binary demographic variables
    analysis_df['same_sex'] = analysis_df['individual1_sex'] == analysis_df['individual2_sex']
    analysis_df['same_race'] = analysis_df['individual1_race'] == analysis_df['individual2_race']
    analysis_df['age_diff'] = abs(analysis_df['individual1_age'] - analysis_df['individual2_age'])
    analysis_df['age_diff_group'] = pd.cut(analysis_df['age_diff'], 
                                           bins=[0, 5, 10, 20, 100], 
                                           labels=['0-5', '6-10', '11-20', '20+'])
    analysis_df['priors_diff'] = abs(analysis_df['individual1_priors_count'] - analysis_df['individual2_priors_count'])
    analysis_df['priors_diff_group'] = pd.cut(analysis_df['priors_diff'], 
                                             bins=[-1, 0, 1, 5, 100], 
                                             labels=['None', '1', '2-5', '5+'])
    
    # Calculate consistency by demographic factors
    demo_factors = ['same_sex', 'same_race', 'age_diff_group', 'priors_diff_group']
    demo_results = {}
    
    for factor in demo_factors:
        consistency_by_factor = analysis_df.groupby(factor)['is_consistent'].mean() * 100
        demo_results[factor] = consistency_by_factor
        
        print(f"\nConsistency by {factor}:")
        for value, rate in consistency_by_factor.items():
            print(f"  - {value}: {rate:.2f}%")
    
    return demo_results, analysis_df

def analyze_by_persona(results_df):
    """Analyze consistency rates by persona."""
    # Calculate consistency rates by persona
    persona_consistency = results_df.groupby('persona_id')['is_consistent'].agg(['mean', 'count'])
    persona_consistency['mean_pct'] = persona_consistency['mean'] * 100
    
    # Sort by consistency rate
    persona_consistency = persona_consistency.sort_values('mean_pct', ascending=False)
    
    # Get basic statistics
    mean_consistency = persona_consistency['mean_pct'].mean()
    median_consistency = persona_consistency['mean_pct'].median()
    std_consistency = persona_consistency['mean_pct'].std()
    
    print(f"\nPersona consistency stats:")
    print(f"  - Mean: {mean_consistency:.2f}%")
    print(f"  - Median: {median_consistency:.2f}%")
    print(f"  - Standard Deviation: {std_consistency:.2f}%")
    
    # Get top and bottom 5 personas
    print("\nTop 5 most consistent personas:")
    for idx, row in persona_consistency.head(5).iterrows():
        print(f"  - Persona {idx}: {row['mean_pct']:.2f}% (sample size: {row['count']})")
    
    print("\nLeast 5 consistent personas:")
    for idx, row in persona_consistency.tail(5).iterrows():
        print(f"  - Persona {idx}: {row['mean_pct']:.2f}% (sample size: {row['count']})")
    
    return persona_consistency

def plot_results(results_df, demo_results, persona_consistency, output_dir="consistency_results"):
    """Generate plots for the analysis results."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Overall consistency types
    plt.figure(figsize=(10, 6))
    consistency_types = results_df['consistency_type'].value_counts(normalize=True) * 100
    sns.barplot(x=consistency_types.index, y=consistency_types.values)
    plt.title('Consistency Types Distribution')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/consistency_types.png", dpi=300)
    
    # Plot 2: Consistency by demographic factors
    for factor, values in demo_results.items():
        plt.figure(figsize=(10, 6))
        sns.barplot(x=values.index, y=values.values)
        plt.title(f'Consistency by {factor}')
        plt.ylabel('Consistency Rate (%)')
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/consistency_by_{factor}.png", dpi=300)
    
    # Plot 3: Distribution of persona consistency rates
    plt.figure(figsize=(12, 6))
    sns.histplot(persona_consistency['mean_pct'], bins=20)
    plt.axvline(persona_consistency['mean_pct'].mean(), color='r', linestyle='--', 
                label=f'Mean: {persona_consistency["mean_pct"].mean():.2f}%')
    plt.title('Distribution of Persona Consistency Rates')
    plt.xlabel('Consistency Rate (%)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/persona_consistency_distribution.png", dpi=300)
    
    # Plot 4: Scatter plot of persona consistency vs. sample size
    plt.figure(figsize=(10, 6))
    plt.scatter(persona_consistency['count'], persona_consistency['mean_pct'], alpha=0.5)
    plt.title('Persona Consistency vs. Sample Size')
    plt.xlabel('Number of Comparison Pairs')
    plt.ylabel('Consistency Rate (%)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/persona_consistency_vs_sample_size.png", dpi=300)
    
    print(f"\nPlots saved to {output_dir}/ directory")

def save_results(df, results_df, persona_consistency, output_dir="core_experiment/consistency_results"):
    """Save analysis results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save combined data
    df.to_csv(f"{output_dir}/combined_data.csv", index=False)
    
    # Save pair-level consistency results
    results_df.to_csv(f"{output_dir}/pair_consistency.csv", index=False)
    
    # Save persona-level consistency results
    persona_consistency.reset_index().to_csv(f"{output_dir}/persona_consistency.csv", index=False)
    
    print(f"\nData files saved to {output_dir}/ directory")

def main():
    # 1. Load and combine results
    input_pattern = "results/chunked_outputs/double_query_study_p*.csv"
    df = load_and_combine_results(input_pattern)
    
    if df is None:
        print("Exiting due to no data found.")
        return
    
    # 2. Analyze consistency
    results_df, consistency_rate, consistency_types = analyze_consistency(df)
    
    # 3. Analyze by demographics
    demo_results, analysis_df = analyze_by_demographics(df, results_df)
    
    # 4. Analyze by persona
    persona_consistency = analyze_by_persona(results_df)
    
    # 5. Generate plots
    plot_results(results_df, demo_results, persona_consistency)
    
    # 6. Save results
    save_results(df, results_df, persona_consistency)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()