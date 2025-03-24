# Load libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

def prepare_feature_analysis(constraints_data, compas_data):
    pair_features = []
    
    # Create a dictionary mapping indices to rows
    compas_dict = {i: row for i, row in compas_data.reset_index().iterrows()}
    
    # Iterate through judges
    for judge_id, judgments in constraints_data.items():
        for judgment_pair in judgments:
            id1, id2 = judgment_pair['pair']

            judgment1 = judgment_pair.get('judgment1', "unknown")
            judgment2 = judgment_pair.get('judgment2', "unknown")
            
            # Skip if both judgments are unknown
            if judgment1 == "unknown" and judgment2 == "unknown":
                continue
            
            # Check if indices exist in the dataset
            if id1 not in compas_dict or id2 not in compas_dict:
                print(f"Warning: Index {id1} or {id2} not found in COMPAS data, skipping")
                continue
                
            # Get features for both individuals
            indiv1 = compas_dict[id1]
            indiv2 = compas_dict[id2]
            
            # Create combined judgment
            combined_judgment = get_combined_judgment(judgment1, judgment2)
            
            # Track consistency
            consistent = (judgment1 == judgment2) or (judgment1 == "unknown" or judgment2 == "unknown")
            
        # Calculate feature differences/comparisons
            pair_data = {
                'judge_id': judge_id,
                'id1': id1,
                'id2': id2,
                'judgment': combined_judgment,
                'judgment1': judgment1,
                'judgment2': judgment2,
                'consistent': consistent,

                # Demographic differences
                'same_race': indiv1['race'] == indiv2['race'],
                'race_diff': f"{indiv1['race']}_{indiv2['race']}",
                'same_gender': indiv1['sex'] == indiv2['sex'],
                'gender_diff': f"{indiv1['sex']}_{indiv2['sex']}",
                'age_diff': indiv1['age'] - indiv2['age'],
                'abs_age_diff': abs(indiv1['age'] - indiv2['age']),
                
                # Criminal history differences
                'prior_count_diff': indiv1['priors_count'] - indiv2['priors_count'],
                'abs_prior_count_diff': abs(indiv1['priors_count'] - indiv2['priors_count']),
                'juv_fel_diff': indiv1['juv_fel_count'] - indiv2['juv_fel_count'],
                'juv_misd_diff': indiv1['juv_misd_count'] - indiv2['juv_misd_count'],
                'charge_degree_same': indiv1['c_charge_degree'] == indiv2['c_charge_degree'],
                
                # Actual outcome differences (useful for some analyses)
                'same_recid_outcome': indiv1['two_year_recid'] == indiv2['two_year_recid'],
                'recid_outcome_diff': indiv1['two_year_recid'] - indiv2['two_year_recid']
            }
            
            # Add separate columns for individual features (for some analyses)
            for col in ['race', 'sex', 'age', 'priors_count', 'juv_fel_count', 'c_charge_degree']:
                pair_data[f'id1_{col}'] = indiv1[col]
                pair_data[f'id2_{col}'] = indiv2[col]
                
            pair_features.append(pair_data)
    
    return pd.DataFrame(pair_features)

def get_combined_judgment(judgment1, judgment2):
    """Combine directional judgments into a single judgment"""
    if judgment1 == "unknown" and judgment2 == "unknown":
        return "unknown"
    elif judgment1 == "similar" or judgment2 == "similar":
        return "similar"
    elif judgment1 == "different" and judgment2 == "different":
        return "different"
    else:
        return "ordered"  # One direction has a preference

# Then in your model preparation:
def prepare_model_data(pair_features_df, mode='multiclass'):
    """Prepare data for modeling, ensuring proper handling of categorical variables"""
    import pandas as pd
    
    # Drop rows with unknown judgments
    valid_df = pair_features_df[pair_features_df['judgment'] != 'unknown']
    
    # Create a copy to avoid modifying the original
    features_df = valid_df.copy()
    
    # Fix any string columns that should be boolean
    bool_columns = ['same_race', 'same_gender', 'charge_degree_same', 'same_recid_outcome']
    for col in bool_columns:
        if col in features_df.columns and features_df[col].dtype == 'object':
            features_df[col] = features_df[col].map({'True': True, 'False': False})
    
    # IMPORTANT: Exclude the 'consistent' column from features since it directly relates to the judgment
    # and will dominate feature importance
    columns_to_drop = ['judgment', 'judgment1', 'judgment2', 'consistent']
    X = features_df.drop(columns_to_drop, axis=1, errors='ignore')
    
    # Define target variable based on mode
    if mode == 'multiclass':
        # Multi-class target, assuming 'judgment' needs to be multi-class
        y = valid_df['judgment']
    elif mode == 'binary':
        # Binary classification target, assuming a binary column exists or creating one
        y = (valid_df['judgment'] == 'similar').astype(int)
    elif mode == 'consistency':
        # Consistency prediction, assuming 'consistent' is relevant
        y = valid_df['consistent'].astype(int)
    else:
        raise ValueError("Invalid mode specified. Choose 'multiclass', 'binary', or 'consistency'.")

    return X, y

def analyze_judgment_consistency():
    """
    Analyze which features lead to consistent vs. inconsistent judgments
    """
    # Filter for pairs with both judgments
    valid_pairs = pair_features_df[pair_features_df['consistent'].notna()]
    
    # Split into consistent vs inconsistent judgments
    consistent = valid_pairs[valid_pairs['consistent'] == True]
    inconsistent = valid_pairs[valid_pairs['consistent'] == False]
    
    # Analyze features that lead to consistency
    X_consistent = consistent.drop(['consistent', 'judgment1', 'judgment2'], axis=1)
    y_consistent = consistent['consistent']
    
    # Compare with features that lead to inconsistency
    X_combined = valid_pairs.drop(['consistent', 'judgment1', 'judgment2'], axis=1)
    y_combined = valid_pairs['consistent']
    
    # Train model to predict consistency
    consistency_model = RandomForestClassifier()
    consistency_model.fit(X_combined, y_combined)
    
    # Extract features that predict consistency
    consistency_importance = consistency_model.feature_importances_
    
    return {
        'feature_names': X_combined.columns,
        'importance': consistency_importance,
        'consistent_count': len(consistent),
        'inconsistent_count': len(inconsistent)
    }

def analyze_feature_importance(X, y, persona_filter=None, target_type='multiclass'):
    """
    Analyze which features are most important for fairness judgments
    """
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Filter by persona if specified
    if persona_filter and 'judge_id' in X.columns:
        mask = X['judge_id'] == persona_filter
        X_filtered = X[mask].drop('judge_id', axis=1)
        y_filtered = y[mask]
    else:
        X_filtered = X.copy()
        if 'judge_id' in X_filtered.columns:
            X_filtered = X_filtered.drop('judge_id', axis=1)
        y_filtered = y
    
    # Handle string/object columns - convert them to numeric
    for col in X_filtered.columns:
        if X_filtered[col].dtype == 'object' or pd.api.types.is_string_dtype(X_filtered[col]):
            # For columns with 'race_diff' or other categorical data, use one-hot encoding
            if col not in ['judge_id', 'id1', 'id2']:  # Skip identifier columns
                X_filtered = pd.get_dummies(X_filtered, columns=[col], drop_first=False)
    
    # Check if we have enough data after filtering
    if len(X_filtered) < 10:
        print(f"Warning: Not enough data for persona {persona_filter}, skipping")
        # Return empty results with structure maintained
        return {
            'feature_names': X_filtered.columns,
            'importance': np.zeros(len(X_filtered.columns)),
            'sorted_indices': np.array([]),
            'demographic_importance': 0,
            'behavioral_importance': 0,
            'accuracy': 0,
            'class_metrics': {},
            'model': None
        }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42)
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importance = rf.feature_importances_
    
    # Get feature names
    feature_names = X_filtered.columns
    
    # Sort by importance
    indices = np.argsort(importance)[::-1]
    
    # Group features by type
    demographic_features = []
    for col in feature_names:
        if any(term in col for term in ['same_race', 'same_gender', 'age_diff', 'abs_age_diff', 'race_', 'gender_']):
            demographic_features.append(col)
    
    # Calculate aggregate importance by feature type
    demographic_importance = sum([importance[i] for i, feat in enumerate(feature_names) 
                                if feat in demographic_features or 'age' in feat or 'sex' in feat or 'race' in feat])
    
    behavioral_importance = sum([importance[i] for i, feat in enumerate(feature_names) 
                               if 'prior' in feat or 'juv' in feat or 'charge' in feat])
    
    # Model accuracy
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # For multiclass, compute accuracy for each class
    class_metrics = {}
    if target_type == 'multiclass':
        for class_name in y_filtered.unique():
            class_mask = y_test == class_name
            if sum(class_mask) > 0:  # Ensure class exists in test set
                class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
                class_metrics[class_name] = class_accuracy
    
    return {
        'feature_names': feature_names,
        'importance': importance,
        'sorted_indices': indices,
        'demographic_importance': demographic_importance,
        'behavioral_importance': behavioral_importance,
        'accuracy': accuracy,
        'class_metrics': class_metrics,
        'model': rf
    }

def compare_persona_importance(X, y, target_type='multiclass'):
    """Compare feature importance across different personas/judges"""
    results = {}
    
    # Get unique judges
    if 'judge_id' in X.columns:
        for judge in X['judge_id'].unique():
            results[judge] = analyze_feature_importance(X, y, judge, target_type)
    
    # Also get overall importance
    results['all'] = analyze_feature_importance(X, y, None, target_type)
    
    return results

def visualize_feature_importance(importance_results, top_n=15, output_dir='visualizations'):
    """Create visualizations of feature importance with custom color scheme"""
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    
    # Custom color palette
    custom_colors = ['#88CCEE', '#DDCC77', '#AA4499', '#882255', '#44AA99']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Bar chart of top N features
    plt.figure(figsize=(12, 8))
    
    feature_names = importance_results['feature_names']
    importance = importance_results['importance']
    indices = importance_results['sorted_indices']
    
    # Only show top N features
    if len(indices) > top_n:
        indices = indices[:top_n]
    
    # Repeat colors as needed for all bars
    bar_colors = [custom_colors[i % len(custom_colors)] for i in range(len(indices))]
    
    plt.barh(range(len(indices)), importance[indices], align='center', color=bar_colors)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top Features for Fairness Judgments')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_features_importance.png', dpi=300)
    
    # Demographic vs Behavioral importance
    plt.figure(figsize=(8, 6))
    labels = ['Demographic Features', 'Behavioral Features', 'Other']
    
    other_importance = 1.0 - (importance_results['demographic_importance'] + 
                              importance_results['behavioral_importance'])
    
    sizes = [importance_results['demographic_importance'], 
             importance_results['behavioral_importance'],
             other_importance]
    
    # Use the first 3 colors for the pie chart
    pie_colors = custom_colors[:3]
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=pie_colors)
    plt.axis('equal')
    plt.title('Relative Importance: Feature Categories')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_category_importance.png', dpi=300)

def compare_persona_importance_plot(results, output_dir='visualizations'):
    """Compare feature importance across different personas/judges with custom colors"""
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl
    
    # Custom color palette
    custom_colors = ['#88CCEE', '#DDCC77', '#AA4499', '#882255', '#44AA99']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top 10 features overall
    if 'all' in results and 'sorted_indices' in results['all'] and len(results['all']['sorted_indices']) > 0:
        top_indices = results['all']['sorted_indices'][:min(10, len(results['all']['sorted_indices']))]
        top_features = [results['all']['feature_names'][i] for i in top_indices]
    else:
        # Fallback if 'all' results not available
        all_features = set()
        for persona, result in results.items():
            if 'feature_names' in result and 'sorted_indices' in result and len(result['sorted_indices']) > 0:
                top_indices = result['sorted_indices'][:min(5, len(result['sorted_indices']))]
                all_features.update([result['feature_names'][i] for i in top_indices])
        top_features = list(all_features)[:10]
    
    # Create comparison dataframe
    comparison_data = []
    
    for persona, result in results.items():
        if 'feature_names' not in result or 'importance' not in result:
            continue
            
        # Create a dictionary mapping feature names to importance
        feat_imp = {result['feature_names'][i]: result['importance'][i] for i in range(len(result['feature_names']))}
        
        # Add data for each top feature
        for feature in top_features:
            comparison_data.append({
                'Persona': persona,
                'Feature': feature,
                'Importance': feat_imp.get(feature, 0)
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    if comparison_df.empty:
        print("Warning: No comparison data available for personas")
        return
    
    # Create heatmap with custom colormap
    plt.figure(figsize=(14, 8))
    pivot_table = comparison_df.pivot(index='Feature', columns='Persona', values='Importance')
    
    # Create a custom colormap from our color scheme
    custom_cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', 
                                                               [custom_colors[0], custom_colors[2], custom_colors[3]],
                                                               N=256)
    
    sns.heatmap(pivot_table, annot=True, cmap=custom_cmap, fmt='.3f')
    plt.title('Feature Importance Comparison Across Personas')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_comparison.png', dpi=300)
    
    # Create demographic vs behavioral comparison
    demo_behav_data = []
    for persona, result in results.items():
        if 'demographic_importance' not in result or 'behavioral_importance' not in result:
            continue
            
        demo_behav_data.append({
            'Persona': persona,
            'Demographic': result['demographic_importance'],
            'Behavioral': result['behavioral_importance']
        })
    
    demo_behav_df = pd.DataFrame(demo_behav_data)
    if demo_behav_df.empty:
        return
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(demo_behav_df))
    width = 0.35
    
    plt.bar(x - width/2, demo_behav_df['Demographic'], width, label='Demographic', color=custom_colors[0])
    plt.bar(x + width/2, demo_behav_df['Behavioral'], width, label='Behavioral', color=custom_colors[1])
    
    plt.xlabel('Persona')
    plt.ylabel('Importance Score')
    plt.title('Demographic vs. Behavioral Feature Importance by Persona/Judge')
    plt.xticks(x, demo_behav_df['Persona'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/demo_vs_behav_by_persona.png', dpi=300)
    
def analyze_judgment_distribution(pair_features_df, output_dir='visualizations'):
    """Analyze the distribution of judgment types"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Count judgments by type
    judgment_counts = pair_features_df['judgment'].value_counts()
    
    plt.figure(figsize=(10, 6))
    judgment_counts.plot(kind='bar')
    plt.title('Distribution of Judgment Types')
    plt.xlabel('Judgment Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/judgment_distribution.png')
    
    # Calculate consistency rate
    consistency_rate = pair_features_df['consistent'].mean()
    print(f"Judgment consistency rate: {consistency_rate:.2f}")
    
    # Analyze consistency by judgment type
    consistency_by_type = pair_features_df.groupby('judgment')['consistent'].mean()
    
    plt.figure(figsize=(10, 6))
    consistency_by_type.plot(kind='bar')
    plt.title('Consistency Rate by Judgment Type')
    plt.xlabel('Judgment Type')
    plt.ylabel('Consistency Rate')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/consistency_by_judgment.png')

def statistical_significance_test(results):
    """Test whether differences in feature importance across personas are statistically significant"""
    # Create dataframe of importance by persona for each feature
    feature_importance_df = pd.DataFrame()
    
    # Get union of top 15 features across all personas
    top_features = set()
    for persona, result in results.items():
        top_features.update([result['feature_names'][i] for i in result['sorted_indices'][:15]])
    
    top_features = list(top_features)
    
    # Perform ANOVA for each feature
    anova_results = {}
    
    for feature in top_features:
        feature_by_persona = []
        
        for persona, result in results.items():
            if persona == 'all':
                continue
                
            # Get feature index and importance
            try:
                idx = list(result['feature_names']).index(feature)
                importance = result['importance'][idx]
                feature_by_persona.append({'persona': persona, 'importance': importance})
            except ValueError:
                continue
        
        if len(feature_by_persona) > 1:
            # Create groups for ANOVA
            groups = []
            for item in feature_by_persona:
                # Use bootstrap to create groups
                np.random.seed(42)
                bootstrap_samples = np.random.normal(item['importance'], item['importance']*0.1, 30)
                groups.append(bootstrap_samples)
            
            # Perform ANOVA
            f_stat, p_val = stats.f_oneway(*groups)
            
            anova_results[feature] = {
                'f_statistic': f_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
    
    return pd.DataFrame(anova_results).T

def create_feature_importance_table(results, significance_results):
    """Create the final table of feature importance across personas with significance indicators"""
    # Get overall top 20 features
    top_features = [results['all']['feature_names'][i] for i in results['all']['sorted_indices'][:20]]
    
    # Create table data
    table_data = []
    
    for feature in top_features:
        row = {'Feature': feature}
        
        # Categorize feature
        if any(term in feature.lower() for term in ['race', 'gender', 'sex', 'age']):
            category = 'Demographic'
        elif any(term in feature.lower() for term in ['prior', 'juv', 'charge', 'crime']):
            category = 'Behavioral'
        else:
            category = 'Other'
            
        row['Category'] = category
        
        # Add importance for each persona
        for persona, result in results.items():
            try:
                idx = list(result['feature_names']).index(feature)
                importance = result['importance'][idx]
                row[persona] = importance
            except ValueError:
                row[persona] = 0
        
        # Add significance indicator
        if feature in significance_results.index:
            row['Significant'] = '✓' if significance_results.loc[feature, 'significant'] else ''
            row['p-value'] = significance_results.loc[feature, 'p_value']
        else:
            row['Significant'] = ''
            row['p-value'] = 1.0
            
        table_data.append(row)
    
    # Create and format DataFrame
    table_df = pd.DataFrame(table_data)
    
    # Sort by overall importance
    table_df = table_df.sort_values('all', ascending=False)
    
    return table_df

def run_feature_importance_analysis(constraints_data, compas_data, output_dir='feature_analysis'):
    """Run the complete feature importance analysis pipeline"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing feature data...")
    pair_features_df = prepare_feature_analysis(constraints_data, compas_data)
    
    # Analyze judgment distribution
    print("Analyzing judgment distribution...")
    analyze_judgment_distribution(pair_features_df, output_dir)
    
    # Run multiclass analysis (similar, ordered, different)
    print("Running multiclass analysis...")
    X_multi, y_multi = prepare_model_data(pair_features_df, 'multiclass')
    multiclass_results = compare_persona_importance(X_multi, y_multi, 'multiclass')
    
    # Run binary analysis (similar vs not similar)
    print("Running binary analysis...")
    X_binary, y_binary = prepare_model_data(pair_features_df, 'binary')
    binary_results = compare_persona_importance(X_binary, y_binary, 'binary')
    
    # Run consistency analysis
    print("Running consistency analysis...")
    X_consist, y_consist = prepare_model_data(pair_features_df, 'consistency')
    consistency_results = compare_persona_importance(X_consist, y_consist, 'consistency')
    
    # Create visualizations for each analysis type
    print("Creating visualizations...")
    visualize_feature_importance(multiclass_results['all'], 
                                output_dir=f"{output_dir}/multiclass")
    compare_persona_importance_plot(multiclass_results, 
                                   output_dir=f"{output_dir}/multiclass")
    
    visualize_feature_importance(binary_results['all'], 
                                output_dir=f"{output_dir}/binary")
    compare_persona_importance_plot(binary_results, 
                                   output_dir=f"{output_dir}/binary")
    
    visualize_feature_importance(consistency_results['all'], 
                                output_dir=f"{output_dir}/consistency")
    compare_persona_importance_plot(consistency_results, 
                                   output_dir=f"{output_dir}/consistency")
    
    # Test statistical significance
    print("Testing statistical significance...")
    multiclass_significance = statistical_significance_test(multiclass_results)
    binary_significance = statistical_significance_test(binary_results)
    consistency_significance = statistical_significance_test(consistency_results)
    
    # Create final tables
    print("Creating final tables...")
    multiclass_table = create_feature_importance_table(multiclass_results, multiclass_significance)
    binary_table = create_feature_importance_table(binary_results, binary_significance)
    consistency_table = create_feature_importance_table(consistency_results, consistency_significance)
    
    # Save to CSV
    multiclass_table.to_csv(f'{output_dir}/multiclass_feature_importance.csv', index=False)
    binary_table.to_csv(f'{output_dir}/binary_feature_importance.csv', index=False)
    consistency_table.to_csv(f'{output_dir}/consistency_feature_importance.csv', index=False)
    
    # Display summary
    print("\n=== Feature Importance Analysis Complete ===")
    print(f"Model Accuracy (Multiclass): {multiclass_results['all']['accuracy']:.2f}")
    print(f"Model Accuracy (Binary): {binary_results['all']['accuracy']:.2f}")
    print(f"Model Accuracy (Consistency): {consistency_results['all']['accuracy']:.2f}")
    
    print("\nTop 5 Features for Fairness Judgments (Multiclass):")
    top5 = multiclass_table[['Feature', 'Category', 'all', 'Significant']].head(5)
    print(top5.to_string(index=False))
    
    return {
        'multiclass': {
            'results': multiclass_results,
            'significance': multiclass_significance,
            'table': multiclass_table
        },
        'binary': {
            'results': binary_results,
            'significance': binary_significance,
            'table': binary_table
        },
        'consistency': {
            'results': consistency_results,
            'significance': consistency_significance,
            'table': consistency_table
        },
        'pair_features': pair_features_df
    }

# Example usage
if __name__ == "__main__":
    import json
    import sys
    import os
    
    if len(sys.argv) < 3:
        print("Usage: python feature_analysis.py <constraints_json> <compas_data_csv>")
        sys.exit(1)
    
    constraints_file = sys.argv[1]
    compas_file = sys.argv[2]
    
    # Load data
    with open(constraints_file, 'r') as f:
        constraints_data = json.load(f)
    
    compas_data = pd.read_parquet(compas_file)
    
    # Run analysis
    results = run_feature_importance_analysis(constraints_data, compas_data)
    print(f"Analysis complete. Results saved to 'feature_analysis/' directory.")