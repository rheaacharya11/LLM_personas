import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_and_combine_files(directory='../results/vanilla_experiment', pattern='fairness_judgments_chunk*.csv'):
    """
    Load all CSV files matching the pattern and combine them into a single dataframe.
    """
    # Find all files matching the pattern
    file_paths = glob.glob(os.path.join(directory, pattern))
    
    if not file_paths:
        raise ValueError(f"No files found matching pattern: {os.path.join(directory, pattern)}")
    
    print(f"Found {len(file_paths)} files to combine")
    
    # Read each file into a list of dataframes
    dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"Loaded {len(df)} rows from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Combined dataframe has {len(combined_df)} rows")
    return combined_df

def analyze_judgment_consistency(df):
    """
    Analyze the consistency of judgments between normal and swapped presentations.
    """
    # Create a dictionary to group by comparison_id
    paired_judgments = {}
    
    # Iterate through the dataframe to pair normal and swapped judgments
    for _, row in df.iterrows():
        comparison_id = row['comparison_id']
        order = row['order']
        judgment = row['judgment']
        
        if comparison_id not in paired_judgments:
            paired_judgments[comparison_id] = {}
            
        paired_judgments[comparison_id][order] = judgment
    
    # Now analyze consistency for each comparison
    consistency_results = []
    
    for comp_id, judgments in paired_judgments.items():
        # Skip if we don't have both normal and swapped
        if 'normal' not in judgments or 'swapped' not in judgments:
            continue
            
        normal_judgment = judgments['normal']
        swapped_judgment = judgments['swapped']
        
        # Determine consistency
        is_consistent = False
        consistency_type = "inconsistent"
        
        if normal_judgment == 'similar' and swapped_judgment == 'similar':
            is_consistent = True
            consistency_type = "both_similar"
        elif normal_judgment == 'x_higher_than_y' and swapped_judgment == 'y_higher_than_x':
            is_consistent = True
            consistency_type = "x_higher_y_higher"
        elif normal_judgment == 'y_higher_than_x' and swapped_judgment == 'x_higher_than_y':
            is_consistent = True
            consistency_type = "y_higher_x_higher"
        
        consistency_results.append({
            'comparison_id': comp_id,
            'normal_judgment': normal_judgment,
            'swapped_judgment': swapped_judgment,
            'is_consistent': is_consistent,
            'consistency_type': consistency_type
        })
    
    # Convert to dataframe
    consistency_df = pd.DataFrame(consistency_results)
    
    # Calculate overall consistency
    consistency_rate = consistency_df['is_consistent'].mean() * 100
    
    print(f"Overall consistency rate: {consistency_rate:.2f}%")
    
    # Count different types of judgment pairs
    judgment_pairs = [
        (row['normal_judgment'], row['swapped_judgment']) 
        for _, row in consistency_df.iterrows()
    ]
    
    judgment_pair_counts = Counter(judgment_pairs)
    
    # Print counts of different judgment pairs
    print("\nJudgment pair distribution:")
    for (normal, swapped), count in judgment_pair_counts.most_common():
        percentage = (count / len(consistency_df)) * 100
        print(f"  - Normal: {normal}, Swapped: {swapped}: {count} ({percentage:.2f}%)")
    
    return consistency_df, judgment_pair_counts

def generate_judgment_distribution_plot(judgment_pair_counts, output_file="figures/judgment_distribution.png"):
    """
    Generate a heatmap visualization of judgment pair distributions.
    """
    # Define the possible judgments
    judgments = ['similar', 'x_higher_than_y', 'y_higher_than_x']
    
    # Create a matrix for the heatmap
    matrix = np.zeros((len(judgments), len(judgments)))
    
    # Fill the matrix
    total_count = sum(judgment_pair_counts.values())
    
    for (normal, swapped), count in judgment_pair_counts.items():
        if normal in judgments and swapped in judgments:
            i = judgments.index(normal)
            j = judgments.index(swapped)
            matrix[i, j] = count / total_count * 100
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(matrix, annot=True, fmt='.1f', 
                    xticklabels=judgments, yticklabels=judgments,
                    cmap='YlGnBu')
    
    plt.title('Distribution of Judgment Pairs (% of total)')
    plt.xlabel('Swapped Presentation Judgment')
    plt.ylabel('Normal Presentation Judgment')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    
    print(f"Saved judgment distribution plot to {output_file}")
    return plt.gcf()

def analyze_demographic_factors(df, consistency_df):
    """
    Analyze how demographic factors might influence judgment consistency.
    """
    # Merge the consistency data with the original dataframe to get demographic info
    # Only use the 'normal' order rows to avoid duplication
    normal_rows = df[df['order'] == 'normal'].copy()
    
    # Merge on comparison_id
    demo_analysis = pd.merge(
        normal_rows, 
        consistency_df[['comparison_id', 'is_consistent', 'consistency_type']], 
        on='comparison_id'
    )
    
    # Create features that might be relevant
    demo_analysis['same_sex'] = demo_analysis['individual1_sex'] == demo_analysis['individual2_sex']
    demo_analysis['same_race'] = demo_analysis['individual1_race'] == demo_analysis['individual2_race']
    demo_analysis['age_difference'] = abs(demo_analysis['individual1_age'] - demo_analysis['individual2_age'])
    demo_analysis['priors_difference'] = abs(demo_analysis['individual1_priors_count'] - 
                                            demo_analysis['individual2_priors_count'])
    
    # Analyze consistency by demographic factors
    print("\nConsistency by demographic factors:")
    
    # By sex
    sex_consistency = demo_analysis.groupby('same_sex')['is_consistent'].mean() * 100
    print("\nConsistency when individuals have the same sex:")
    for same_sex, consistency in sex_consistency.items():
        print(f"  - {'Same' if same_sex else 'Different'} sex: {consistency:.2f}%")
    
    # By race
    race_consistency = demo_analysis.groupby('same_race')['is_consistent'].mean() * 100
    print("\nConsistency when individuals have the same race:")
    for same_race, consistency in race_consistency.items():
        print(f"  - {'Same' if same_race else 'Different'} race: {consistency:.2f}%")
    
    # By age difference (binned)
    demo_analysis['age_diff_bin'] = pd.cut(
        demo_analysis['age_difference'], 
        bins=[0, 5, 10, 20, 100], 
        labels=['0-5', '6-10', '11-20', '21+']
    )
    age_consistency = demo_analysis.groupby('age_diff_bin')['is_consistent'].mean() * 100
    print("\nConsistency by age difference:")
    for age_bin, consistency in age_consistency.items():
        print(f"  - Age difference {age_bin} years: {consistency:.2f}%")
    
    # By priors difference (binned)
    demo_analysis['priors_diff_bin'] = pd.cut(
        demo_analysis['priors_difference'], 
        bins=[-1, 0, 2, 5, 100], 
        labels=['None', '1-2', '3-5', '6+']
    )
    priors_consistency = demo_analysis.groupby('priors_diff_bin')['is_consistent'].mean() * 100
    print("\nConsistency by difference in prior convictions:")
    for priors_bin, consistency in priors_consistency.items():
        print(f"  - Priors difference {priors_bin}: {consistency:.2f}%")
    
    return demo_analysis

def main():
    # Load and combine all files
    print("Loading and combining files...")
    combined_df = load_and_combine_files()
    
    # Save the combined data
    output_file = "../results/vanilla_experiment/combined_fairness_judgments.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined data to {output_file}")
    
    # Analyze judgment consistency
    print("\nAnalyzing judgment consistency...")
    consistency_df, judgment_pair_counts = analyze_judgment_consistency(combined_df)
    
    # Save consistency analysis
    consistency_df.to_csv("../results/vanilla_experiment/vanilla_consistency.csv", index=False)
    print("Saved consistency analysis to judgment_consistency_analysis.csv")
    
    # Generate visualization of judgment distribution
    print("\nGenerating visualization...")
    generate_judgment_distribution_plot(judgment_pair_counts)
    
    # Analyze demographic factors
    print("\nAnalyzing demographic factors...")
    demo_analysis = analyze_demographic_factors(combined_df, consistency_df)
    
    # Print summary statistics
    print("\nSummary statistics from combined data:")
    print(f"Total comparisons: {len(consistency_df)}")
    print(f"Consistent judgments: {consistency_df['is_consistent'].sum()} ({consistency_df['is_consistent'].mean()*100:.2f}%)")
    
    # Consistency by judgment type
    consistency_by_type = consistency_df.groupby('consistency_type').size()
    print("\nConsistency by type:")
    for ctype, count in consistency_by_type.items():
        percentage = (count / len(consistency_df)) * 100
        print(f"  - {ctype}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    main()