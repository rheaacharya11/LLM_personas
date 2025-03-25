import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from collections import defaultdict
import re
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm

RACE_GROUPS = {
    "asian": ["asian", "chinese", "filipino", "japanese", "korean", "vietnamese", "asian indian", "cambodian", "laotian", "thai", "burmese", "pakistani", "other asian"],
    "black": ["black or african american", "black", "african american"],
    "white": ["white", "white alone"],
    "native": ["american indian", "native hawaiian", "alaska native", "navajo", "cherokee", "other specified american indian"],
    "hispanic": ["mexican", "latino", "hispanic"],
    "other": ["some other race", "two or more races", "other", "other race alone", "other asian alone", "other combinations"]
}
# Define paths
RESULTS_DIR = "../results"
RESULTS_DIR_default = "../results/fixed_personas_binary/"
RESULTS_DIR_expert = "../results/fixed_personas_binary/"
OUTPUT_DIR = "consistency_analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
def extract_persona_attributes(persona_str):
    """Extract structured fields from a persona string."""
    def get_match(pattern):
        match = re.search(pattern, persona_str)
        return match.group(1).strip().lower() if match else None

    return {
        'age': get_match(r"age: (\d+)"),
        'sex': get_match(r"sex: ([^\n,]+)"),
        'race': get_match(r"race: ([^\n,]+)"),
        'ancestry': get_match(r"ancestry: ([^\n,]+)"),
        'birthplace': get_match(r"place of birth: ([^\n,]+)"),
        'occupation': get_match(r"occupation category: ([^\n,]+)"),
        'lifestyle': get_match(r"lifestyle: ([^\n,]+)"),
        'quirks': get_match(r"defining quirks: ([^\n,]+)"),
        'personal_time': get_match(r"personal time: ([^\n,]+)"),
        'personality': get_match(r"big five scores: ([^\n,]+)"),
        'political_views': get_match(r"political views: ([^\n,]+)"),
        'religion': get_match(r"religion: ([^\n,]+)")
    }

def load_persona_demographics(parquet_path):
    """Load and parse persona demographics from a .parquet file."""
    try:
        df = pd.read_parquet(parquet_path)
        if 'persona' not in df.columns:
            print("No 'persona' column found in parquet.")
            return pd.DataFrame()

        print(f"Parsing demographics for {len(df)} personas...")

        demo_data = []
        for idx, row in df.iterrows():
            attributes = extract_persona_attributes(row['persona'])
            attributes['persona_id'] = idx + 1  # Adjust if judgment matrix starts at 1
            demo_data.append(attributes)

        demographics_df = pd.DataFrame(demo_data)
        demographics_df['persona_id'] = demographics_df['persona_id'].astype(int)

        # Convert age to numeric
        demographics_df['age'] = pd.to_numeric(demographics_df['age'], errors='coerce')

        print(f"Successfully extracted structured demographics for {len(demographics_df)} personas.")
        print("Columns in demographics:", demographics_df.columns.tolist())

        return demographics_df

    except Exception as e:
        print(f"Error loading/parsing demographics: {e}")
        return pd.DataFrame()

def map_race_category(raw_race):
    raw_race = raw_race.lower()
    for group, keywords in RACE_GROUPS.items():
        if any(keyword in raw_race for keyword in keywords):
            return group
    return "other"

def load_all_judgments():
    """Load persona judgments from all runs"""
    all_judgments = defaultdict(lambda: defaultdict(dict))
    pattern = r'(B|C)?_?fairness_judgments_(\d+)_(\d+)\.csv'
    dir_path = os.path.join(RESULTS_DIR, f"fixed_personas_binary")
    files = glob.glob(os.path.join(dir_path, "*fairness_judgments_*.csv"))

    print(f"Found {len(files)} judgment files")
    
    for file in sorted(files):
        match = re.search(pattern, os.path.basename(file))
        if not match:
            continue
            
        run = match.group(1)
        start_persona = int(match.group(2))
        end_persona = int(match.group(3))
        
        df = pd.read_csv(file)
        
        for persona_id in range(start_persona, end_persona + 1):
            persona_df = df[df['persona_id'] == persona_id]
            
            for i in range(0, len(persona_df), 2):
                if i+1 >= len(persona_df):
                    break
                    
                row1 = persona_df.iloc[i]
                row2 = persona_df.iloc[i+1]
                
                id1 = int(row1['individual1_id'])
                id2 = int(row1['individual2_id'])
                pair_key = (id1, id2)
                
                judgment1 = row1['judgment']  # forward
                judgment2 = row2['judgment']  # reverse
                
                all_judgments[run][persona_id][pair_key] = (judgment1, judgment2)
    
    print(f"Loaded judgments for {len(all_judgments)} runs")
    return all_judgments

def load_llm_judgments():
    """Load LLM judgments from 'default' and 'expert' runs"""
    llm_judgments = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for llm_type in ['default', 'expert']:
        dir_path = os.path.join(RESULTS_DIR, f"fixed_{llm_type}_binary")
        files = glob.glob(os.path.join(dir_path, "A_fairness_judgments_*.csv"))
        
        print(f"[{llm_type.upper()}] Found {len(files)} files")

        for file in sorted(files):
            # Determine run number based on filename
            match = re.search(r'_(\d+)_(\d+)\.csv$', file)
            run_num = int(match.group(1)) if match else 0

            df = pd.read_csv(file)

            for i in range(0, len(df), 2):
                if i+1 >= len(df):
                    break

                row1 = df.iloc[i]
                row2 = df.iloc[i+1]

                id1 = int(row1['individual1_id'])
                id2 = int(row1['individual2_id'])
                persona_id = id1 % 10  # Assuming id1 and id2 belong to same persona group

                pair_key = (id1, id2)
                judgment1 = row1['judgment']
                judgment2 = row2['judgment']

                llm_judgments[llm_type][persona_id][run_num][pair_key] = (judgment1, judgment2)

    return llm_judgments
def compute_llm_consistency(llm_judgments):
    llm_consistency_scores = {}

    for llm_type in llm_judgments:
        runs = list(llm_judgments[llm_type].keys())
        total = 0
        total_score = 0

        for i, run1 in enumerate(runs):
            for run2 in runs[i+1:]:
                shared_pairs = set(llm_judgments[llm_type][run1].keys()) & set(llm_judgments[llm_type][run2].keys())
                for pair in shared_pairs:
                    j1 = llm_judgments[llm_type][run1][pair]
                    j2 = llm_judgments[llm_type][run2][pair]
                    score = pairwise_semantic_similarity(j1, j2)
                    total_score += score
                    total += 1

        if total > 0:
            llm_consistency_scores[llm_type] = total_score / total * 100  # Percentage

    return llm_consistency_scores
def analyze_persona_consistency(all_judgments, llm_judgments=None):
    """Analyze consistency of judgments across personas"""
    consistency_data = []
    all_runs = list(all_judgments.keys())
    
    # Find personas that appear in all runs
    common_personas = set(all_judgments[all_runs[0]].keys())
    for run in all_runs[1:]:
        common_personas &= set(all_judgments[run].keys())
    
    print(f"Found {len(common_personas)} personas that appear in all runs")
    
    for persona_id in sorted(common_personas):
        # Find common pairs across all runs
        common_pairs = set(all_judgments[all_runs[0]][persona_id].keys())
        for run in all_runs[1:]:
            common_pairs &= set(all_judgments[run][persona_id].keys())
        
        if not common_pairs:
            continue
            
        semantic_consistency_total = 0
        total_comparisons = 0

        for i, run1 in enumerate(all_runs):
            for run2 in all_runs[i+1:]:
                for pair in common_pairs:
                    j1 = all_judgments[run1][persona_id][pair]
                    j2 = all_judgments[run2][persona_id][pair]

                    score = pairwise_semantic_similarity(j1, j2)
                    semantic_consistency_total += score
                    total_comparisons += 1
                
        semantic_consistency = semantic_consistency_total / total_comparisons * 100
        consistency_data.append({
            'persona_id': persona_id,
            'semantic_consistency': semantic_consistency,
            'total_comparisons': total_comparisons
        })
    df = pd.DataFrame(consistency_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "persona_consistency.csv"), index=False)
    llm_consistency_scores = compute_llm_consistency(llm_judgments)
    # Create summary visualizations
    plt.figure(figsize=(10, 6))
    sns.histplot(df['semantic_consistency'], bins=20, kde=True)
    plt.title('Distribution of Semantic Consistency Across Personas')
    plt.xlabel('Semantic Consistency (%)')
    plt.ylabel('Count')
    for llm_type, score in llm_consistency_scores.items():
        plt.axvline(score, linestyle='--', linewidth=2, label=f'{llm_type} LLM')

    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "semantic_consistency_histogram.png"), dpi=300, bbox_inches='tight')

    print(f"Saved persona consistency analysis to {OUTPUT_DIR}")
    return df

def analyze_pair_consistency(all_judgments):
    """Analyze consistency of judgments for each pair"""
    pair_data = []
    
    # Get all unique pairs
    all_pairs = set()
    for run in all_judgments:
        for persona_id in all_judgments[run]:
            all_pairs.update(all_judgments[run][persona_id].keys())
    
    print(f"Found {len(all_pairs)} unique constraint pairs")
    
    # For each pair, compute agreement statistics
    for pair in all_pairs:
        forward_similar = 0
        forward_different = 0
        reverse_similar = 0
        reverse_different = 0
        total_judgments = 0
        
        for run in all_judgments:
            for persona_id in all_judgments[run]:
                if pair in all_judgments[run][persona_id]:
                    judgments = all_judgments[run][persona_id][pair]
                    
                    # Forward direction
                    if judgments[0] == 'similar':
                        forward_similar += 1
                    else:
                        forward_different += 1
                    
                    # Reverse direction
                    if judgments[1] == 'similar':
                        reverse_similar += 1
                    else:
                        reverse_different += 1
                    
                    total_judgments += 1
        
        if total_judgments > 0:
            # Calculate agreement percentages
            forward_total = forward_similar + forward_different
            reverse_total = reverse_similar + reverse_different
            
            forward_agreement = max(forward_similar, forward_different) / forward_total * 100
            forward_similar_pct = forward_similar / forward_total * 100
            
            reverse_agreement = max(reverse_similar, reverse_different) / reverse_total * 100
            reverse_similar_pct = reverse_similar / reverse_total * 100
            
            # Overall agreement
            overall_similar = forward_similar + reverse_similar
            overall_different = forward_different + reverse_different
            overall_total = overall_similar + overall_different
            overall_agreement = max(overall_similar, overall_different) / overall_total * 100
            
            pair_data.append({
                'pair': str(pair),
                'id1': pair[0],
                'id2': pair[1],
                'forward_agreement': forward_agreement,
                'reverse_agreement': reverse_agreement,
                'overall_agreement': overall_agreement,
                'forward_similar_pct': forward_similar_pct,
                'reverse_similar_pct': reverse_similar_pct,
                'total_judgments': total_judgments,
                'forward_majority': 'similar' if forward_similar > forward_different else 'different',
                'reverse_majority': 'similar' if reverse_similar > reverse_different else 'different'
            })
    
    df = pd.DataFrame(pair_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "pair_consistency.csv"), index=False)
    
    # Create summary visualizations
    plt.figure(figsize=(10, 6))
    sns.histplot(df['overall_agreement'], bins=20, kde=True)
    plt.title('Distribution of Overall Agreement Across Pairs')
    plt.xlabel('Agreement (%)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "pair_agreement_distribution.png"), dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='forward_similar_pct', y='reverse_similar_pct')
    plt.title('Forward vs. Reverse "Similar" Judgment Percentage')
    plt.xlabel('Forward Similar (%)')
    plt.ylabel('Reverse Similar (%)')
    plt.plot([0, 100], [0, 100], 'r--')  # Diagonal line
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "forward_vs_reverse_similar.png"), dpi=300, bbox_inches='tight')
    
    print(f"Saved pair consistency analysis to {OUTPUT_DIR}")
    return df
def flatten_llm_judgments(nested_judgments):
    flat_judgments = defaultdict(lambda: defaultdict(dict))  # llm_type -> run -> pair
    for llm_type in nested_judgments:
        for persona_id in nested_judgments[llm_type]:
            for run in nested_judgments[llm_type][persona_id]:
                for pair, judgments in nested_judgments[llm_type][persona_id][run].items():
                    flat_judgments[llm_type][run][pair] = judgments
    return flat_judgments
def compare_llm_to_personas(llm_judgments, all_judgments):
    """Compare LLM judgments to persona judgments"""
    comparison_data = []
    
    # Aggregate persona judgments to get majority opinion
    persona_consensus = defaultdict(lambda: defaultdict(int))
    
    for run in all_judgments:
        for persona_id in all_judgments[run]:
            for pair, judgments in all_judgments[run][persona_id].items():
                if judgments[0] == 'similar':
                    persona_consensus[pair]['forward_similar'] += 1
                else:
                    persona_consensus[pair]['forward_different'] += 1
                    
                if judgments[1] == 'similar':
                    persona_consensus[pair]['reverse_similar'] += 1
                else:
                    persona_consensus[pair]['reverse_different'] += 1
                    
                persona_consensus[pair]['total'] += 1
    
    # Get majority judgments for each pair
    persona_majority = {}
    for pair, counts in persona_consensus.items():
        forward_majority = 'similar' if counts['forward_similar'] >= counts['forward_different'] else 'different'
        reverse_majority = 'similar' if counts['reverse_similar'] >= counts['reverse_different'] else 'different'
        
        forward_agreement = max(counts['forward_similar'], counts['forward_different']) / counts['total'] * 100
        reverse_agreement = max(counts['reverse_similar'], counts['reverse_different']) / counts['total'] * 100
        
        persona_majority[pair] = {
            'forward_majority': forward_majority,
            'reverse_majority': reverse_majority,
            'forward_agreement': forward_agreement,
            'reverse_agreement': reverse_agreement,
            'total_judgments': counts['total']
        }
    
    # Compare LLM judgments to persona majority
    for llm_type in llm_judgments:
        runs = list(llm_judgments[llm_type].keys())
        
        # Compute LLM majority judgments
        llm_majority = {}
        for pair in persona_majority:
            forward_judgments = []
            reverse_judgments = []
            
            for run in runs:
                if pair in llm_judgments[llm_type][run]:
                    judgments = llm_judgments[llm_type][run][pair]
                    forward_judgments.append(judgments[0])
                    reverse_judgments.append(judgments[1])
            
            if forward_judgments:
                forward_similar = forward_judgments.count('similar')
                forward_different = forward_judgments.count('different')
                forward_majority = 'similar' if forward_similar >= forward_different else 'different'
                
                reverse_similar = reverse_judgments.count('similar')
                reverse_different = reverse_judgments.count('different')
                reverse_majority = 'similar' if reverse_similar >= reverse_different else 'different'
                
                llm_majority[pair] = {
                    'forward_majority': forward_majority,
                    'reverse_majority': reverse_majority
                }
        
        # Compare LLM judgments to persona judgments
        for pair in persona_majority:
            if pair in llm_majority:
                persona_forward = persona_majority[pair]['forward_majority']
                persona_reverse = persona_majority[pair]['reverse_majority']
                
                llm_forward = llm_majority[pair]['forward_majority']
                llm_reverse = llm_majority[pair]['reverse_majority']
                
                forward_match = persona_forward == llm_forward
                reverse_match = persona_reverse == llm_reverse
                
                comparison_data.append({
                    'llm_type': llm_type,
                    'pair': str(pair),
                    'forward_match': forward_match,
                    'reverse_match': reverse_match,
                    'persona_forward_majority': persona_forward,
                    'persona_reverse_majority': persona_reverse,
                    'llm_forward_majority': llm_forward,
                    'llm_reverse_majority': llm_reverse,
                    'persona_forward_agreement': persona_majority[pair]['forward_agreement'],
                    'persona_reverse_agreement': persona_majority[pair]['reverse_agreement']
                })
    
    df = pd.DataFrame(comparison_data)
    print(f"LLM comparison rows: {len(df)}")
    print(f"LLM comparison columns: {df.columns.tolist()}")
    df.to_csv(os.path.join(OUTPUT_DIR, "llm_persona_comparison.csv"), index=False)
    
    # Create summary visualizations
    match_by_llm = df.groupby('llm_type').agg({
        'forward_match': 'mean',
        'reverse_match': 'mean'
    }).reset_index()
    
    match_by_llm['forward_match'] *= 100
    match_by_llm['reverse_match'] *= 100
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(match_by_llm))
    width = 0.35
    
    plt.bar(x - width/2, match_by_llm['forward_match'], width, label='Forward')
    plt.bar(x + width/2, match_by_llm['reverse_match'], width, label='Reverse')
    
    plt.xlabel('LLM Type')
    plt.ylabel('Match with Persona Majority (%)')
    plt.title('LLM Match with Persona Majority Judgments')
    plt.xticks(x, match_by_llm['llm_type'])
    plt.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "llm_persona_match.png"), dpi=300, bbox_inches='tight')
    
    print(f"Saved LLM-persona comparison to {OUTPUT_DIR}")
    return df

def create_judgment_matrix(all_judgments, run='A'):
    """Create a binary matrix of judgments for clustering analysis"""
    # Get all personas and pairs from this run
    personas = sorted(list(all_judgments[run].keys()))
    all_pairs = set()
    for persona_id in personas:
        all_pairs.update(all_judgments[run][persona_id].keys())
    pairs = sorted(list(all_pairs))
    
    print(f"Creating judgment matrix with {len(personas)} personas and {len(pairs)} pairs")
    
    # Create matrix
    forward_matrix = np.zeros((len(personas), len(pairs)))
    reverse_matrix = np.zeros((len(personas), len(pairs)))
    
    # Fill matrix with judgments (1 for similar, 0 for different)
    for i, persona_id in enumerate(personas):
        for j, pair in enumerate(pairs):
            if pair in all_judgments[run][persona_id]:
                judgments = all_judgments[run][persona_id][pair]
                forward_matrix[i, j] = 1 if judgments[0] == 'similar' else 0
                reverse_matrix[i, j] = 1 if judgments[1] == 'similar' else 0
    
    # Combine matrices
    combined_matrix = np.hstack((forward_matrix, reverse_matrix))
    
    return combined_matrix, personas, pairs


def cluster_personas(all_judgments, run='A', llm_judgments=None):
    """Cluster personas based on their fairness judgments"""
    print(f"Clustering personas based on run {run}...")
    
    # Create judgment matrix
    judgment_matrix, personas, pairs = create_judgment_matrix(all_judgments, run)
    
    # Load demographic data if available
    demographics = load_persona_demographics('../data/unique_personas.parquet')
    
    # Apply UMAP for dimensionality reduction (better preserves global structure than t-SNE)
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=15)
        embedding = reducer.fit_transform(judgment_matrix)
    except ImportError:
        print("UMAP not available, falling back to t-SNE")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(personas)//2), 
                   init='pca', learning_rate='auto')
        embedding = tsne.fit_transform(judgment_matrix)
    
    # Apply HDBSCAN for clustering (more robust than DBSCAN)
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, prediction_data=True)
        labels = clusterer.fit_predict(embedding)
    except ImportError:
        print("HDBSCAN not available, falling back to DBSCAN")
        scaler = StandardScaler()
        scaled_embedding = scaler.fit_transform(embedding)
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        labels = clusterer.fit_predict(scaled_embedding)
    
    # Get cluster labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters, {list(labels).count(-1)} outliers")
    
    # Create dataframe with results
    cluster_df = pd.DataFrame({
        'persona_id': personas,
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'cluster': labels
    })
    
    # Find characteristic judgments for each cluster
    cluster_profiles = []
    for cluster in sorted(set(labels)):
        if cluster == -1:  # Skip noise points
            continue
            
        # Get indices of personas in this cluster
        cluster_indices = np.where(labels == cluster)[0]
        
        # Find distinctive judgments
        distinctive_judgments = []
        for j in range(judgment_matrix.shape[1]//2):  # Only look at forward direction
            # Calculate agreement within cluster
            cluster_agreement = np.mean(judgment_matrix[cluster_indices, j])
            # Calculate agreement in other clusters
            other_indices = np.where(labels != cluster)[0]
            if len(other_indices) > 0:
                other_agreement = np.mean(judgment_matrix[other_indices, j])
                # If there's a significant difference
                if abs(cluster_agreement - other_agreement) > 0.3:
                    pair = pairs[j]
                    judgment = 'similar' if cluster_agreement > 0.5 else 'different'
                    distinctive_judgments.append((pair, judgment, abs(cluster_agreement - other_agreement)))
        
        # Sort by distinctiveness
        distinctive_judgments.sort(key=lambda x: x[2], reverse=True)
        
        cluster_profiles.append({
            'cluster': cluster,
            'size': len(cluster_indices),
            'distinctive_judgments': distinctive_judgments[:5]  # Top 5 most distinctive
        })
    
    # Merge demographic data if available
    if demographics is not None:
        # Convert persona_id in cluster_df to int (if not already)
        cluster_df['persona_id'] = cluster_df['persona_id'].astype(int)
        # Merge with demographics data
        cluster_df = cluster_df.merge(demographics, how='left', on='persona_id')

        cluster_df['race_group'] = cluster_df['race'].fillna('other').apply(map_race_category)

    
    # Project LLM judgments into the same space
    llm_embeddings = {}
    if llm_judgments is not None:
        for llm_type in llm_judgments:
            for run_id in llm_judgments[llm_type]:
                # Create LLM judgment vector
                llm_matrix = np.zeros((1, len(pairs)*2))
                
                for j, pair in enumerate(pairs):
                    if pair in llm_judgments[llm_type][run_id]:
                        judgments = llm_judgments[llm_type][run_id][pair]
                        llm_matrix[0, j] = 1 if judgments[0] == 'similar' else 0
                        llm_matrix[0, j+len(pairs)] = 1 if judgments[1] == 'similar' else 0
                
                # Project LLM into the same space
                if 'UMAP' in str(type(reducer)):
                    llm_embedding = reducer.transform(llm_matrix)
                else:
                    llm_embedding = tsne.transform(llm_matrix)
                
                llm_embeddings[(llm_type, run_id)] = llm_embedding
    
    # Create high-quality visualizations
    create_cluster_visualization(cluster_df, llm_embeddings, demographics, run)
    
    # Create heatmap of persona-LLM similarity
    if llm_judgments is not None:
        create_persona_llm_similarity(judgment_matrix, personas, llm_judgments, pairs, run)
    
    # Save cluster data
    cluster_df.to_csv(os.path.join(OUTPUT_DIR, f"persona_clusters_run_{run}.csv"), index=False)
    
    # Save cluster profiles
    with open(os.path.join(OUTPUT_DIR, f"cluster_profiles_run_{run}.txt"), 'w') as f:
        for profile in cluster_profiles:
            f.write(f"Cluster {profile['cluster']} (n={profile['size']}):\n")
            f.write("  Distinctive judgments:\n")
            for pair, judgment, diff in profile['distinctive_judgments']:
                f.write(f"    {pair}: {judgment} (diff: {diff:.2f})\n")
            f.write("\n")
    
    print(f"Saved clustering results to {OUTPUT_DIR}")
    return cluster_df, cluster_profiles

def create_cluster_visualization(cluster_df, llm_embeddings, demographics, run):
    """Create a beautiful visualization of the persona clusters with demographic information"""
    # Set up a clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Define color mapping if demographics are available
    if demographics is not None and 'age' in cluster_df.columns:
        # Use age for color
        age_cmap = plt.cm.viridis
        ages = cluster_df['age'].fillna(-1)
        age_norm = plt.Normalize(ages[ages >= 0].min(), ages[ages >= 0].max())
        
        # Use sex for marker shape
        markers = {'M': 'o', 'F': '^', 'Other': 's', 'Unknown': 'x'}
        
        # Use race for size
        sizes = {'White': 40, 'Black': 40, 'Asian': 40, 'Hispanic': 40, 
                'Other': 40, 'Unknown': 20}
        
        # Plot each demographic group
        '''
        if 'sex' in cluster_df.columns and 'race' in cluster_df.columns:
            sexes = cluster_df['sex'].dropna().unique()
            races = cluster_df['race'].dropna().unique()

            for sex in sexes:
                for race_value in races:
                    subset = cluster_df[(cluster_df['sex'] == sex) & (cluster_df['race'] == race_value)]
                    if not subset.empty:
                        plt.scatter(subset['x'], subset['y'],
                                    label=f"{sex}, {race_value}",
                                    alpha=0.6,
                                    s=40)
        '''
        if 'race_group' in cluster_df.columns:
            unique_races = sorted(cluster_df['race_group'].dropna().unique())
            palette = sns.color_palette("hls", len(unique_races))  # or another colormap

            for i, race in enumerate(unique_races):
                race_points = cluster_df[cluster_df['race_group'] == race]
                plt.scatter(race_points['x'], race_points['y'],
                            label=race.title(),
                            color=palette[i],
                            s=40,
                            alpha=0.7)
    
    # Plot LLM points
    for (llm_type, run_id), embedding in llm_embeddings.items():
        plt.scatter(embedding[0, 0], embedding[0, 1], 
                   marker='*', s=300, 
                   edgecolor='black', linewidth=1.5,
                   label=f"{llm_type} Run {run_id}")
    
    # Add labels and legend
    plt.title(f'Persona Clusters Based on Fairness Judgments (Run {run})', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    
    # Handle the legend - if too many items, put it outside the plot
    if demographics is not None and 'race' in cluster_df.columns and 'sex' in cluster_df.columns:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if len(by_label) > 10:
            plt.legend(by_label.values(), by_label.keys(), 
                      loc='center left', bbox_to_anchor=(1, 0.5), 
                      fontsize=10, frameon=True)
        else:
            plt.legend(by_label.values(), by_label.keys(), 
                      loc='best', fontsize=10, frameon=True)
    else:
        plt.legend(loc='best', fontsize=10, frameon=True)
    
   
    
    # Improve layout
    plt.tight_layout()
    
    # Save high-resolution figure
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"persona_clusters_run_{run}.png"), 
               dpi=300, bbox_inches='tight')
    plt.colorbar().remove()  # remove existing colorbar if one is there

    plt.close()

def create_persona_llm_similarity(judgment_matrix, personas, llm_judgments, pairs, run):
    """Create a visualization showing similarity between personas and LLMs"""
    # Calculate similarity between each persona and each LLM
    similarity_data = []
    LLM_COLORS = {
        'default': '#DDCC77',  # yellow
        'expert': '#AA4499'    # PINK
    }
    for llm_type in llm_judgments:
        for llm_run in llm_judgments[llm_type]:
            # Create LLM judgment vector
            llm_vector = np.zeros(len(pairs)*2)
            
            for j, pair in enumerate(pairs):
                if pair in llm_judgments[llm_type][llm_run]:
                    judgments = llm_judgments[llm_type][llm_run][pair]
                    llm_vector[j] = 1 if judgments[0] == 'similar' else 0
                    llm_vector[j+len(pairs)] = 1 if judgments[1] == 'similar' else 0
            
            # Calculate similarity with each persona
            for i, persona_id in enumerate(personas):
                persona_vector = judgment_matrix[i]
                
                # Calculate Jaccard similarity
                intersection = np.sum(np.logical_and(persona_vector == 1, llm_vector == 1))
                union = np.sum(np.logical_or(persona_vector == 1, llm_vector == 1))
                jaccard = intersection / union if union > 0 else 0
                
                # Calculate agreement percentage (simpler measure)
                agreement = np.mean(persona_vector == llm_vector)
                
                similarity_data.append({
                    'persona_id': persona_id,
                    'llm_type': llm_type,
                    'llm_run': llm_run,
                    'jaccard_similarity': jaccard,
                    'agreement_pct': agreement * 100
                })
    
    similarity_df = pd.DataFrame(similarity_data)
    
    # Save similarity data
    similarity_df.to_csv(os.path.join(OUTPUT_DIR, f"persona_llm_similarity_run_{run}.csv"), index=False)
    
    # Create visualizations
    # 1. Top 20 most similar personas to each LLM
    plt.figure(figsize=(14, 10))
    
    # For each LLM type
    for llm_type in similarity_df['llm_type'].unique():
        llm_data = similarity_df[similarity_df['llm_type'] == llm_type]
        
        # Average across runs
        avg_similarity = llm_data.groupby('persona_id')['agreement_pct'].mean().reset_index()
        
        # Get top 20
        top20 = avg_similarity.sort_values('agreement_pct', ascending=False).head(20)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top20, x='agreement_pct', y='persona_id', palette='viridis')
        plt.title(f'Top 20 Personas Most Similar to {llm_type}', fontsize=14)
        plt.xlabel('Agreement (%)', fontsize=12)
        plt.ylabel('Persona ID', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"top20_similar_personas_{llm_type}_run_{run}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Distribution of similarity scores
    plt.figure(figsize=(12, 8))
    
    for llm_type in similarity_df['llm_type'].unique():
        llm_data = similarity_df[similarity_df['llm_type'] == llm_type]
        
        # Average across runs
        avg_similarity = llm_data.groupby('persona_id')['agreement_pct'].mean()
        
        sns.kdeplot(avg_similarity, label=llm_type, fill=True, alpha=0.3, color=LLM_COLORS.get(llm_type, None))

    
    plt.title('Distribution of Persona-LLM Similarity Scores', fontsize=14)
    plt.xlabel('Agreement (%)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"persona_llm_similarity_distribution_run_{run}.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved persona-LLM similarity analysis to {OUTPUT_DIR}")
    return similarity_df
def pairwise_semantic_similarity(j1, j2):
    """
    Compute semantic similarity between two judgment tuples.
    Each is a tuple like ('similar', 'different').

    Returns a float between 0 and 1.
    """
    if j1 == j2:
        return 1.0
    if j1 == tuple(reversed(j2)):
        return 1.0
    if set(j1) == set(j2):  # one is ('similar', 'different') and so is the other
        return 1.0
    if j1[0] == j2[0]:
        return 0.5
    return 0.0


    return 1.0 - abs(pair_score(judgment1) - pair_score(judgment2))
def main():
    """Run the consistency analysis pipeline"""
    print("Starting fairness judgment consistency analysis...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    
    # Load data
    print("\nLoading persona judgments...")
    all_judgments = load_all_judgments()
    
    print("\nLoading LLM judgments...")
    print("\nLoading LLM judgments...")
    llm_judgments_nested = load_llm_judgments()
    llm_judgments = flatten_llm_judgments(llm_judgments_nested)
    
    # Analyze persona consistency
    print("\nAnalyzing persona consistency...")
    persona_consistency = analyze_persona_consistency(all_judgments, llm_judgments)
    
    # Analyze pair consistency
    print("\nAnalyzing pair consistency...")
    pair_consistency = analyze_pair_consistency(all_judgments)
    
    # Compare LLM to personas
    print("\nComparing LLM judgments to persona judgments...")
    llm_comparison = compare_llm_to_personas(llm_judgments, all_judgments)
    
    # Cluster personas based on judgments
    print("\nClustering personas based on fairness judgments...")
    for run in all_judgments.keys():
        cluster_df, cluster_profiles = cluster_personas(all_judgments, run, llm_judgments)
    
    print("\nConsistency analysis complete!")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()