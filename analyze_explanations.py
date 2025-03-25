import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import re
import glob
import os
import time
import json
import nltk
import multiprocessing as mp
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.data.path.append('/n/home04/racharya/nltk_data')
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

TOL_COLORS = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']

def extract_age(persona_string):
    age_match = re.search(r"age: (\d+)", persona_string)
    return int(age_match.group(1)) if age_match else None

def get_valid_persona_ids():
    df = pd.read_parquet("data/unique_personas.parquet")
    df['age'] = df['persona'].apply(extract_age)
    return set((df[df['age'] >= 18].index + 1).astype(str))

def load_all_persona_data(folder_path="results/fixed_personas_binary"):
    start_time = time.time()
    file_pattern = os.path.join(folder_path, "B_fairness_judgments_*_*.csv")
    file_list = glob.glob(file_pattern)
    file_list.sort()

    print(f"Found {len(file_list)} files to process")
    all_data = pd.DataFrame()
    total_files = len(file_list)
    processed_files = 0
    total_personas = 0
    total_judgments = 0

    for file_path in file_list:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            personas_in_file = df['persona_id'].nunique()
            judgments_in_file = len(df)
            all_data = pd.concat([all_data, df], ignore_index=True)
            total_personas += personas_in_file
            total_judgments += judgments_in_file
            processed_files += 1
            if processed_files % 10 == 0 or processed_files == total_files:
                print(f"Processed {processed_files}/{total_files} files | "
                      f"Total personas: {total_personas} | "
                      f"Total judgments: {total_judgments}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    if 'comparison_id' not in all_data.columns:
        if 'pair_id' in all_data.columns:
            print("Using pair_id as comparison_id")
            all_data['comparison_id'] = all_data['pair_id']
        else:
            print("Creating comparison_id from individual IDs")
            all_data['comparison_id'] = all_data.apply(
                lambda row: f"{min(row['individual1_id'], row['individual2_id'])}_{max(row['individual1_id'], row['individual2_id'])}",
                axis=1
            )

    # Apply age filtering
    print("Filtering to personas aged 18+...")
    valid_persona_ids = get_valid_persona_ids()
    all_data = all_data[all_data['persona_id'].astype(str).isin(valid_persona_ids)]

    end_time = time.time()
    print(f"Data loading complete in {end_time - start_time:.2f} seconds")
    print(f"Total dataset: {len(all_data)} judgments from {all_data['persona_id'].nunique()} personas on {all_data['comparison_id'].nunique()} comparisons")
    return all_data

def plot_clusters_with_extras(results, default_judges, expert_judges, output_file):
    plt.figure(figsize=(12, 10))
    palette = TOL_COLORS[:results['cluster'].nunique()]

    for cluster_id in results['cluster'].unique():
        cluster_data = results[results['cluster'] == cluster_id]
        plt.scatter(cluster_data['tsne1'], cluster_data['tsne2'], label=f"Cluster {cluster_id}",
                    alpha=0.7, s=50, color=palette[cluster_id % len(palette)])

    if default_judges is not None:
        plt.scatter(default_judges['tsne1'], default_judges['tsne2'], marker='*', s=200,
                    color='black', label='Default Judges')

    if expert_judges is not None:
        plt.scatter(expert_judges['tsne1'], expert_judges['tsne2'], marker='*', s=200,
                    color='red', label='Expert Judges')

    plt.legend()
    plt.title('Persona Clusters with Experts and Defaults')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def create_pair_matrix(df):
    """
    Create a matrix where:
    - Each row is a persona
    - Each column is a comparison pair
    - Values are 1 for 'similar' judgment, 0 for 'different'
    """
    # Ensure we have a consistent comparison ID
    if 'comparison_id' not in df.columns:
        print("Creating comparison_id from pair information")
        df['comparison_id'] = df.apply(
            lambda row: f"{min(row['individual1_id'], row['individual2_id'])}_{max(row['individual1_id'], row['individual2_id'])}", 
            axis=1
        )
    
    # Convert judgment to binary
    df['judgment_binary'] = (df['judgment'] == 'similar').astype(int)
    
    # Create the pivot table
    pair_matrix = pd.pivot_table(
        df, 
        values='judgment_binary',
        index='persona_id',
        columns='comparison_id',
        fill_value=0
    )
    
    print(f"Created pair matrix with {pair_matrix.shape[0]} personas and {pair_matrix.shape[1]} pairs")
    return pair_matrix

def analyze_pair_consensus(pair_matrix):
    """
    Identify which comparison pairs have the most consensus or disagreement
    """
    # For each pair, calculate the percentage of 'similar' judgments
    consensus_scores = pair_matrix.mean().sort_values()
    
    # Pairs with scores near 0 or 1 have high consensus
    # Pairs with scores near 0.5 have high disagreement
    high_consensus_similar = consensus_scores.nlargest(10)
    high_consensus_different = consensus_scores.nsmallest(10)
    high_disagreement = consensus_scores[(consensus_scores > 0.4) & (consensus_scores < 0.6)].head(10)
    
    return high_consensus_similar, high_consensus_different, high_disagreement

def create_keyword_features(df, sample_size=None):
    """
    Analyze fairness principles in the reasoning text
    """
    # Define fairness lexicon
    fairness_lexicon = {
        'equality': [  # Same rules or resources for everyone
            'equal', 'same', 'equivalent', 'uniform', 'alike', 'even', 'parity', 'symmetry'
        ],
        
        'equity': [  # Adjusted treatment based on individual need or circumstance
            'need', 'require', 'disadvantage', 'vulnerable', 'accommodate', 'support', 'barrier', 'access'
        ],
        
        'merit': [  # Rewards based on effort, ability, or contribution
            'deserve', 'earn', 'merit', 'qualified', 'achievement', 'effort', 'hardworking', 'capable', 'responsibility'
        ],
        
        'procedural_justice': [  # Fairness through consistent and impartial processes
            'procedure', 'process', 'systematic', 'rules', 'consistent', 'impartial', 'unbiased', 'transparent', 'objective'
        ],
        
        'identity_fairness': [  # Fairness with regard to group-based identity or historical marginalization
            'race', 'gender', 'ethnicity', 'minority', 'identity', 'background', 'diversity', 'representation', 'inclusion'
        ]
    }
    
    # Sample personas if needed
    if sample_size and df['persona_id'].nunique() > sample_size:
        print(f"Sampling {sample_size} personas for keyword analysis")
        persona_ids = df['persona_id'].unique()
        sampled_personas = np.random.choice(persona_ids, size=sample_size, replace=False)
        df_sample = df[df['persona_id'].isin(sampled_personas)]
    else:
        df_sample = df
    
    # Create profiles DataFrame
    profiles = pd.DataFrame(index=df_sample['persona_id'].unique())
    
    # Track progress
    total_personas = len(profiles)
    processed = 0
    
    # Process each persona's explanations
    for persona_id in profiles.index:
        persona_data = df_sample[df_sample['persona_id'] == persona_id]
        
        # Count fairness keywords
        persona_counts = {category: 0 for category in fairness_lexicon.keys()}
        
        # Combine all reasoning text for this persona
        combined_text = ' '.join(persona_data['reasoning'].dropna())
        
        # Count words in each category
        for category, keywords in fairness_lexicon.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, combined_text.lower())
                persona_counts[category] += len(matches)
        
        # Get total words for normalization
        words = nltk.word_tokenize(combined_text.lower())
        total_words = len([w for w in words if w.isalpha()])
        
        # Normalize by total words
        if total_words > 0:
            for category in persona_counts:
                profiles.loc[persona_id, f'{category}_ratio'] = persona_counts[category] / total_words * 100
        
        # Add judgment statistics
        profiles.loc[persona_id, 'similar_ratio'] = persona_data['judgment'].eq('similar').mean() * 100
        
        # Update progress
        processed += 1
        if processed % 100 == 0 or processed == total_personas:
            print(f"Processed {processed}/{total_personas} personas for keyword analysis")
    
    return profiles

def cluster_personas(pair_matrix, profiles=None, n_clusters=5, max_personas=10000):
    """
    Cluster personas based on their judgment patterns and fairness principles
    """
    # Sample if the dataset is too large
    if len(pair_matrix) > max_personas:
        print(f"Sampling {max_personas} personas for clustering (from {len(pair_matrix)})")
        sampled_indices = np.random.choice(pair_matrix.index, size=max_personas, replace=False)
        pair_features = pair_matrix.loc[sampled_indices]
        if profiles is not None:
            profile_features = profiles.loc[profiles.index.intersection(sampled_indices)]
        else:
            profile_features = None
    else:
        pair_features = pair_matrix
        profile_features = profiles
    
    # Normalize pair features
    pair_features_norm = (pair_features - pair_features.mean()) / pair_features.std()
    
    # Combine with profile features if available
    if profile_features is not None and not profile_features.empty:
        # Ensure we have matching indices
        matching_indices = pair_features_norm.index.intersection(profile_features.index)
        
        # Normalize profile features
        profile_features_norm = (profile_features.loc[matching_indices] - 
                                 profile_features.loc[matching_indices].mean()) / \
                                profile_features.loc[matching_indices].std()
        
        # Combine using column binding
        combined_features = pd.concat([pair_features_norm.loc[matching_indices], 
                                      profile_features_norm], axis=1)
    else:
        combined_features = pair_features_norm
    
    # Handle any remaining NaN values
    combined_features = combined_features.fillna(0)
    
    print(f"Running K-means clustering with {n_clusters} clusters on {len(combined_features)} personas")
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    combined_features.columns = combined_features.columns.astype(str)

    clusters = kmeans.fit_predict(combined_features)
    
    # Add cluster assignments
    results = pd.DataFrame(index=combined_features.index)
    results['cluster'] = clusters
    
    print("Generating t-SNE visualization...")
    
    # Use t-SNE for visualization (sample further if needed)
    tsne_sample_size = min(5000, len(combined_features))
    if len(combined_features) > tsne_sample_size:
        tsne_indices = np.random.choice(combined_features.index, size=tsne_sample_size, replace=False)
        tsne_features = combined_features.loc[tsne_indices]
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_features)-1))
        tsne_results = tsne.fit_transform(tsne_features)
        
        # Create a temporary DataFrame for the t-SNE results
        tsne_df = pd.DataFrame(index=tsne_indices)
        tsne_df['tsne1'] = tsne_results[:, 0]
        tsne_df['tsne2'] = tsne_results[:, 1]
        
        # Join with the full results
        results = results.join(tsne_df, how='left')
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_features)-1))
        tsne_results = tsne.fit_transform(combined_features)
        results['tsne1'] = tsne_results[:, 0]
        results['tsne2'] = tsne_results[:, 1]
    
    print("Clustering complete")
    return results

def plot_clusters(results, output_file='persona_clusters.png'):
    """
    Plot the clustering results using t-SNE visualization
    """
    # Filter to only include points with t-SNE coordinates
    valid_results = results.dropna(subset=['tsne1', 'tsne2'])
    
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot
    scatter = plt.scatter(
        valid_results['tsne1'], 
        valid_results['tsne2'], 
        c=valid_results['cluster'], 
        cmap='viridis', 
        alpha=0.7,
        s=50
    )
    
    plt.colorbar(scatter, label='Cluster')
    plt.title('Persona Clusters Based on Fairness Judgments', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Cluster visualization saved to {output_file}")

def analyze_clusters(results, pair_matrix, profiles=None):
    """
    Analyze the characteristics of each cluster
    """
    cluster_stats = results['cluster'].value_counts().sort_index()
    print("\nCluster sizes:")
    for cluster, count in cluster_stats.items():
        print(f"Cluster {cluster}: {count} personas ({count/len(results)*100:.1f}%)")
    
    # Analyze judgment patterns for each cluster
    cluster_judgments = {}
    cluster_profiles = pd.DataFrame()
    cluster_distinctive_pairs = pd.DataFrame()
    
    for cluster_id in results['cluster'].unique():
        # Get personas in this cluster
        cluster_personas = results[results['cluster'] == cluster_id].index
        
        # Get judgments for these personas
        if len(cluster_personas) > 0:
            cluster_pair_data = pair_matrix.loc[pair_matrix.index.intersection(cluster_personas)]
            if len(cluster_pair_data) > 0:
                # Calculate mean judgment for each pair
                cluster_mean = cluster_pair_data.mean()
                cluster_judgments[cluster_id] = cluster_mean
                
                # Compare to other clusters
                other_personas = pair_matrix.index.difference(cluster_personas)
                if len(other_personas) > 0:
                    other_pair_data = pair_matrix.loc[other_personas]
                    other_mean = other_pair_data.mean()
                    
                    # Find distinctive pairs (largest difference in judgment)
                    diff = (cluster_mean - other_mean).abs()
                    distinctive_pairs = diff.nlargest(10)
                    
                    # Add to the distinctive pairs DataFrame
                    cluster_distinctive_pairs[f'Cluster_{cluster_id}'] = distinctive_pairs
                
        # Calculate profile features if available
        if profiles is not None and not profiles.empty:
            cluster_profile_data = profiles.loc[profiles.index.intersection(cluster_personas)]
            if len(cluster_profile_data) > 0:
                cluster_profiles[f'Cluster_{cluster_id}'] = cluster_profile_data.mean()
    
    # Transpose for easier reading
    if not cluster_profiles.empty:
        cluster_profiles = cluster_profiles.T
    
    return cluster_judgments, cluster_profiles, cluster_distinctive_pairs

def analyze_top_distinctive_pairs(df, cluster_distinctive_pairs, results):
    """
    Analyzes the top distinctive comparison pairs for each cluster
    """
    print("\nAnalyzing top distinctive comparison pairs for each cluster...")
    
    # Dictionary to store analysis results
    pair_analysis = {}
    
    for cluster_col in cluster_distinctive_pairs.columns:
        cluster_id = cluster_col.split('_')[-1]
        print(f"\nCluster {cluster_id} distinctive pairs:")
        
        # Get the top 5 pairs for this cluster
        top_pairs = cluster_distinctive_pairs[cluster_col].nlargest(5)
        
        for comparison_id, score in top_pairs.items():
            # Get personas in this cluster
            cluster_personas = results[results['cluster'] == int(cluster_id)].index
            
            # Get judgments for this comparison from this cluster
            cluster_judgments = df[(df['persona_id'].isin(cluster_personas)) & 
                                (df['comparison_id'] == comparison_id)]
            
            # Calculate percentage of "similar" judgments
            if len(cluster_judgments) > 0:
                similar_pct = (cluster_judgments['judgment'] == 'similar').mean() * 100
                print(f"  Pair {comparison_id}: {similar_pct:.1f}% said similar (distinctiveness: {score:.3f})")
                
                # Sample some reasoning from this cluster for this pair
                if len(cluster_judgments) > 0:
                    sample_similar = cluster_judgments[cluster_judgments['judgment'] == 'similar']['reasoning'].sample(min(3, len(cluster_judgments[cluster_judgments['judgment'] == 'similar']))).tolist()
                    sample_different = cluster_judgments[cluster_judgments['judgment'] == 'different']['reasoning'].sample(min(3, len(cluster_judgments[cluster_judgments['judgment'] == 'different']))).tolist()
                    
                    pair_analysis[f"Cluster_{cluster_id}_{comparison_id}"] = {
                        'similar_percent': similar_pct,
                        'distinctiveness': score,
                        'sample_similar': sample_similar,
                        'sample_different': sample_different
                    }
    
    return pair_analysis
def extract_common_phrases_mp(texts, min_count=10):
    def tokenize(text):
        tokens = word_tokenize(text.lower())
        return [w for w in tokens if w.isalpha() and w not in stopwords.words('english')]

    with mp.Pool(mp.cpu_count()) as pool:
        token_lists = pool.map(tokenize, texts)

    all_tokens = [token for sublist in token_lists for token in sublist]
    word_freq = Counter(all_tokens)
    bigrams = Counter(tuple(x[i:i+2]) for x in token_lists for i in range(len(x)-1))
    trigrams = Counter(tuple(x[i:i+3]) for x in token_lists for i in range(len(x)-2))

    return {
        'common_words': word_freq.most_common(30),
        'common_bigrams': bigrams.most_common(30),
        'common_trigrams': trigrams.most_common(20)
    }


def load_expert_default_tsne(tsne_model, features, file_path):
    df = pd.read_csv(file_path)
    pair_matrix = create_pair_matrix(df)
    keyword_profiles = create_keyword_features(df)
    features_norm = (pair_matrix - pair_matrix.mean()) / pair_matrix.std()
    profiles_norm = (keyword_profiles - keyword_profiles.mean()) / keyword_profiles.std()
    combined = pd.concat([features_norm, profiles_norm], axis=1).fillna(0)
    combined.columns = combined.columns.astype(str)

    # Create safe t-SNE model copy with adjusted perplexity
    effective_perplexity = min(tsne_model.perplexity, max(1, combined.shape[0] - 1))
    tsne_model_small = TSNE(n_components=2, random_state=42, perplexity=effective_perplexity)
    tsne_embeds = tsne_model_small.fit_transform(combined)
    return pd.DataFrame(tsne_embeds, columns=['tsne1', 'tsne2'])
def run_fairness_analysis(folder_path, output_dir="fairness_analysis_results"):
    """
    Run the complete fairness analysis pipeline and save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Starting Fairness Analysis on {folder_path} ===")
    
    # Step 1: Load all data
    print("\nStep 1: Loading data from all CSV files...")
    all_data = load_all_persona_data(folder_path)
    
    # Save the combined data (optional, can be large)
    # all_data.to_csv(os.path.join(output_dir, "combined_data.csv"), index=False)
    
    # Step 2: Create pair matrix (personas x comparisons)
    print("\nStep 2: Creating persona-pair judgment matrix...")
    pair_matrix = create_pair_matrix(all_data)
    
    # Save the pair matrix
    pair_matrix.to_csv(os.path.join(output_dir, "pair_matrix.csv"))
    
    # Step 3: Analyze pair consensus
    print("\nStep 3: Analyzing pair consensus...")
    high_similar, high_different, high_disagreement = analyze_pair_consensus(pair_matrix)
    
    # Save consensus results
    consensus_df = pd.DataFrame({
        'high_similar': high_similar,
        'high_different': high_different,
        'high_disagreement': high_disagreement
    })
    consensus_df.to_csv(os.path.join(output_dir, "pair_consensus.csv"))
    
    # Step 4: Extract keyword features
    print("\nStep 4: Extracting keyword features...")
    # Sample 5000 personas for keyword analysis if there are more than 5000
    keyword_sample_size = min(5000, all_data['persona_id'].nunique())
    keyword_profiles = create_keyword_features(all_data, sample_size=keyword_sample_size)
    
    # Save keyword profiles
    keyword_profiles.to_csv(os.path.join(output_dir, "keyword_profiles.csv"))
    
    # Step 5: Cluster personas
    print("\nStep 5: Clustering personas...")
    n_clusters = 5  # Can be adjusted
    cluster_results = cluster_personas(pair_matrix, keyword_profiles, n_clusters=n_clusters)
    
    # Save cluster assignments
    cluster_results.to_csv(os.path.join(output_dir, "cluster_assignments.csv"))
    
    # Step 6: Visualize clusters
    print("\nStep 6: Visualizing clusters...")
    from sklearn.manifold import TSNE

    print("\nLoading expert/default judge positions...")
    tsne_model = TSNE(n_components=2, random_state=11, perplexity=30)

    default_judges = load_expert_default_tsne(tsne_model, cluster_results, "results/fixed_default_binary/A_fairness_judgments_1_10.csv")
    expert_judges = load_expert_default_tsne(tsne_model, cluster_results, "results/fixed_expert_binary/A_fairness_judgments_1_10.csv")

    plot_clusters_with_extras(cluster_results, default_judges, expert_judges,output_file=os.path.join(output_dir, "persona_clusters.png"))
    
    # Step 7: Analyze clusters
    print("\nStep 7: Analyzing cluster characteristics...")
    cluster_judgments, cluster_profiles, cluster_distinctive_pairs = analyze_clusters(
        cluster_results, pair_matrix, keyword_profiles
    )
    
    # Save cluster analysis results
    for cluster_id, judgments in cluster_judgments.items():
        judgments.to_csv(os.path.join(output_dir, f"cluster_{cluster_id}_judgments.csv"))
    
    cluster_profiles.to_csv(os.path.join(output_dir, "cluster_profiles.csv"))
    cluster_distinctive_pairs.to_csv(os.path.join(output_dir, "cluster_distinctive_pairs.csv"))
    
    # Step 8: Analyze top distinctive pairs
    print("\nStep 8: Analyzing top distinctive pairs for each cluster...")
    pair_analysis = analyze_top_distinctive_pairs(all_data, cluster_distinctive_pairs, cluster_results)
    
    # Save pair analysis as JSON
    with open(os.path.join(output_dir, "pair_analysis.json"), 'w') as f:
        # Convert any non-serializable objects
        for key, value in pair_analysis.items():
            for k, v in value.items():
                if isinstance(v, np.float64):
                    pair_analysis[key][k] = float(v)
        json.dump(pair_analysis, f, indent=2)
    
    # Step 9: Extract common phrases
    print("\nStep 9: Extracting common phrases from all reasoning text...")
    phrase_analysis = extract_common_phrases_mp(all_data)
    
    # Save phrase analysis
    with open(os.path.join(output_dir, "common_phrases.json"), 'w') as f:
        # Convert non-serializable objects
        serializable_phrases = {
            'common_words': [(w, int(c)) for w, c in phrase_analysis['common_words']],
            'common_bigrams': [([w1, w2], int(c)) for (w1, w2), c in phrase_analysis['common_bigrams']],
            'common_trigrams': [([w1, w2, w3], int(c)) for (w1, w2, w3), c in phrase_analysis['common_trigrams']]
        }
        json.dump(serializable_phrases, f, indent=2)
    
    # Generate summary report
    summary = {
        'total_personas': all_data['persona_id'].nunique(),
        'total_comparisons': all_data['comparison_id'].nunique(),
        'total_judgments': len(all_data),
        'similar_judgments_percent': (all_data['judgment'] == 'similar').mean() * 100,
        'clusters': {
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_results['cluster'].value_counts().sort_index().to_dict()
        },
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, "analysis_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Analysis complete! Results saved to {output_dir} ===")
    return summary

# Execute the analysis if this script is run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze fairness judgments across personas")
    parser.add_argument("--folder", type=str, default="results/fixed_personas_binary",
                        help="Folder containing fairness judgment CSV files")
    parser.add_argument("--output", type=str, default="fairness_analysis_results",
                        help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    summary = run_fairness_analysis(args.folder, args.output)
    print("\nAnalysis Summary:")
    print(f"- Analyzed {summary['total_personas']} personas")
    print(f"- {summary['total_judgments']} judgments on {summary['total_comparisons']} comparisons")
    print(f"- {summary['similar_judgments_percent']:.1f}% of judgments were 'similar'")
    print(f"- Identified {summary['clusters']['n_clusters']} distinct persona clusters")
    
    # Print cluster sizes
    print("\nCluster sizes:")
    for cluster_id, size in summary['clusters']['cluster_sizes'].items():
        print(f"  Cluster {cluster_id}: {size} personas ({size/summary['total_personas']*100:.1f}%)")