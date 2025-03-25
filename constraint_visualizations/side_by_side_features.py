import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import functions from the existing feature_importance.py
# Instead of importing, we now include needed functions directly in this script

# This function is from feature_importance.py - now included directly here
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

# This analyze_feature_importance is from feature_importance.py
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
    if len(X_filtered) < 12:
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

def compare_feature_importance_from_files(json_file1, json_file2, compas_file, 
                                         output_dir='comparison_visualizations',
                                         label1='Dataset 1', label2='Dataset 2',
                                         top_n=12, mode='multiclass'):
    """
    Create side-by-side comparison of feature importances from two different JSON files
    
    Parameters:
    -----------
    json_file1, json_file2 : str
        Paths to JSON files containing judgment data
    compas_file : str
        Path to COMPAS dataset (parquet format)
    output_dir : str
        Directory to save output visualizations
    label1, label2 : str
        Labels for the two datasets in the visualization
    top_n : int
        Number of top features to display
    mode : str
        Analysis mode ('multiclass', 'binary', 'consistency', 'similar_ordered')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    with open(json_file1, 'r') as f:
        constraints_data1 = json.load(f)
    
    with open(json_file2, 'r') as f:
        constraints_data2 = json.load(f)
    
    compas_data = pd.read_parquet(compas_file)
    
    # Process first dataset
    print(f"Processing {label1}...")
    pair_features1 = prepare_feature_analysis(constraints_data1, compas_data)
    
    # Process second dataset
    print(f"Processing {label2}...")
    pair_features2 = prepare_feature_analysis(constraints_data2, compas_data)
    
    # Process with prepare_model_data that now handles 'similar_ordered' mode
    X1, y1 = prepare_model_data(pair_features1, mode)
    X2, y2 = prepare_model_data(pair_features2, mode)
    
    # Use 'binary' analysis type for similar_ordered since it's a binary classification
    analysis_mode = 'binary' if mode == 'similar_ordered' else mode
    results1 = analyze_feature_importance(X1, y1, None, analysis_mode)
    results2 = analyze_feature_importance(X2, y2, None, analysis_mode)
    
    # Create comparison visualization
    print("Creating comparison visualization...")
    create_side_by_side_comparison(results1, results2, label1, label2, top_n, output_dir, mode)
    
    return results1, results2

def prepare_model_data(pair_features, mode='multiclass'):
    """
    Prepare data for modeling based on the specified mode
    
    Parameters:
    -----------
    pair_features : pd.DataFrame
        DataFrame with pair-wise features from prepare_feature_analysis
    mode : str
        Analysis mode, one of:
        - 'multiclass': predict full judgment categories (similar, different, ordered)
        - 'binary': predict similar vs not-similar
        - 'consistency': predict whether judgments are consistent 
        - 'similar_ordered': binary classification of similar vs ordered judgments
        
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    """
    # Copy the dataframe to avoid modifying the original
    X = pair_features.copy()
    
    # Handle different prediction modes
    if mode == 'multiclass':
        # Full multiclass prediction (similar, different, ordered)
        # Filter out unknown judgments
        X = X[X['judgment'] != 'unknown']
        y = X['judgment']
        
    elif mode == 'binary':
        # Binary classification: similar vs not-similar
        # Filter out unknown judgments
        X = X[X['judgment'] != 'unknown']
        # Create binary target: 1 for similar, 0 for anything else
        y = (X['judgment'] == 'similar').astype(int)
        
    elif mode == 'consistency':
        # Predict whether judgments are consistent
        # Only use entries where both judgments exist
        X = X[(X['judgment1'] != 'unknown') & (X['judgment2'] != 'unknown')]
        y = X['consistent']
        
    elif mode == 'similar_ordered':
        # Binary classification: similar vs ordered
        # Filter to only include similar or ordered judgments
        X = X[(X['judgment'] == 'similar') | (X['judgment'] == 'ordered')]
        # Create binary target: 1 for similar, 0 for ordered
        y = (X['judgment'] == 'similar').astype(int)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Drop target columns and id columns from feature matrix
    columns_to_drop = ['judgment', 'judgment1', 'judgment2', 'consistent', 'id1', 'id2']
    feature_columns = X.columns.difference(columns_to_drop)
    X = X[feature_columns]
    
    return X, y

def create_side_by_side_comparison(results1, results2, label1, label2, top_n=12, 
                                 output_dir='comparison_visualizations', mode='multiclass'):
    """Create side-by-side bar chart comparing feature importances"""
    # Get union of top features from both datasets
    top_indices1 = results1['sorted_indices'][:min(top_n*2, len(results1['sorted_indices']))]
    top_indices2 = results2['sorted_indices'][:min(top_n*2, len(results2['sorted_indices']))]
    
    top_features1 = [results1['feature_names'][i] for i in top_indices1]
    top_features2 = [results2['feature_names'][i] for i in top_indices2]
    
    # Get union of top features
    all_top_features = list(set(top_features1) | set(top_features2))
    
    # Filter out individual variables (keep only difference variables)
    filtered_features = [f for f in all_top_features if not f.startswith('id1_') and not f.startswith('id2_')]
    all_top_features = filtered_features
    
    # Find all race-related features
    race_features1 = [f for f in results1['feature_names'] if 'race' in f.lower()]
    race_features2 = [f for f in results2['feature_names'] if 'race' in f.lower()]
    race_features = list(set(race_features1) | set(race_features2))
    
    # Combine all race_diff features into a single meta-feature
    race_diff_features1 = [f for f in results1['feature_names'] if 'race_diff_' in f]
    race_diff_features2 = [f for f in results2['feature_names'] if 'race_diff_' in f]
    
    # Calculate combined importance for race_diff in dataset 1
    combined_race_diff_imp1 = 0
    for feature in race_diff_features1:
        idx = list(results1['feature_names']).index(feature)
        combined_race_diff_imp1 += results1['importance'][idx]
    
    # Calculate combined importance for race_diff in dataset 2
    combined_race_diff_imp2 = 0
    for feature in race_diff_features2:
        idx = list(results2['feature_names']).index(feature)
        combined_race_diff_imp2 += results2['importance'][idx]
    
    # Print the combined importance
    print(f"Combined race_diff importance: {combined_race_diff_imp1:.4f} (dataset1), {combined_race_diff_imp2:.4f} (dataset2)")
    
    # Create a synthetic feature for the combined race_diff
    if len(race_diff_features1) > 0 or len(race_diff_features2) > 0:
        all_top_features.append("race_diff_combined")
    
    # Get the most important individual race feature (as backup)
    top_race_feature = None
    top_race_importance = 0
    
    for feature in race_features:
        # Skip the race_diff features since we're combining them
        if 'race_diff_' in feature:
            continue
            
        imp1 = 0
        if feature in results1['feature_names']:
            idx = list(results1['feature_names']).index(feature)
            imp1 = results1['importance'][idx]
            
        imp2 = 0
        if feature in results2['feature_names']:
            idx = list(results2['feature_names']).index(feature)
            imp2 = results2['importance'][idx]
            
        avg_imp = (imp1 + imp2) / 2
        if avg_imp > top_race_importance:
            top_race_importance = avg_imp
            top_race_feature = feature
            
    # Add top individual race feature to all_top_features if not already there
    if top_race_feature and top_race_feature not in all_top_features:
        all_top_features.append(top_race_feature)
    
    # Sort by average importance
    feature_avg_importance = {}
    for feature in all_top_features:
        # Special handling for our synthetic race_diff_combined feature
        if feature == "race_diff_combined":
            imp1 = combined_race_diff_imp1
            imp2 = combined_race_diff_imp2
        else:
            imp1 = 0
            if feature in results1['feature_names']:
                idx = list(results1['feature_names']).index(feature)
                imp1 = results1['importance'][idx]
                
            imp2 = 0
            if feature in results2['feature_names']:
                idx = list(results2['feature_names']).index(feature)
                imp2 = results2['importance'][idx]
            
        feature_avg_importance[feature] = (imp1 + imp2) / 2
    
    # Sort features by average importance
    sorted_features = sorted(all_top_features, 
                            key=lambda x: feature_avg_importance[x], 
                            reverse=True)[:top_n]
                            
    # Print race-related features for debugging
    print(f"Found {len(race_features)} race-related features:")
    for feature in race_features:
        imp1 = 0
        if feature in results1['feature_names']:
            idx = list(results1['feature_names']).index(feature)
            imp1 = results1['importance'][idx]
            
        imp2 = 0
        if feature in results2['feature_names']:
            idx = list(results2['feature_names']).index(feature)
            imp2 = results2['importance'][idx]
            
        print(f"  {feature}: {imp1:.4f} (dataset1), {imp2:.4f} (dataset2)")
    
    # Get importance values for each feature
    imp1 = []
    imp2 = []
    
    for feature in sorted_features:
        # Special handling for our synthetic race_diff_combined feature
        if feature == "race_diff_combined":
            imp1.append(combined_race_diff_imp1)
            imp2.append(combined_race_diff_imp2)
        else:
            if feature in results1['feature_names']:
                idx = list(results1['feature_names']).index(feature)
                imp1.append(results1['importance'][idx])
            else:
                imp1.append(0)
                
            if feature in results2['feature_names']:
                idx = list(results2['feature_names']).index(feature)
                imp2.append(results2['importance'][idx])
            else:
                imp2.append(0)
    
    # Create bar chart
    fig, ax = plt.figure(figsize=(14, 10)), plt.subplot(111)
    x = np.arange(len(sorted_features))
    width = 0.35
    
    # Custom colors
    color1 = '#88CCEE'
    color2 = '#DDCC77'
    
    # Create bars
    bars1 = ax.barh(x - width/2, imp1, width, label=label1, color=color1)
    bars2 = ax.barh(x + width/2, imp2, width, label=label2, color=color2)
    
    # Add labels and legend
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Feature Importance Comparison: {label1} vs {label2} ({mode} mode)')
    ax.set_yticks(x)
    ax.set_yticklabels(sorted_features)
    ax.legend()
    
    # Add values on bars
    for i, v in enumerate(imp1):
        ax.text(v + 0.01, i - width/2, f'{v:.3f}', 
                color='black', fontweight='bold', va='center')
    
    for i, v in enumerate(imp2):
        ax.text(v + 0.01, i + width/2, f'{v:.3f}', 
                color='black', fontweight='bold', va='center')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_comparison_{mode}.png', dpi=300)
    plt.close()
    
    # Create a table of the comparison results
    comparison_table = pd.DataFrame({
        'Feature': sorted_features,
        f'{label1}_importance': imp1,
        f'{label2}_importance': imp2,
        'Difference': [abs(imp1[i] - imp2[i]) for i in range(len(imp1))],
        'Percent_Difference': [abs(imp1[i] - imp2[i])/((imp1[i] + imp2[i])/2)*100 if (imp1[i] + imp2[i]) > 0 else 0 
                              for i in range(len(imp1))]
    })
    
    # Save table to CSV
    comparison_table.to_csv(f'{output_dir}/feature_importance_comparison_{mode}.csv', index=False)
    
    return comparison_table

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python compare_feature_importance.py <json_file1> <json_file2> <compas_file> [label1] [label2]")
        sys.exit(1)
    
    json_file1 = sys.argv[1]
    json_file2 = sys.argv[2]
    compas_file = sys.argv[3]
    
    label1 = sys.argv[4] if len(sys.argv) > 4 else "Dataset 1"
    label2 = sys.argv[5] if len(sys.argv) > 5 else "Dataset 2"
    
    # Run comparison for all modes including the custom similar_ordered mode
    for mode in ['multiclass', 'binary', 'consistency', 'similar_ordered']:
        print(f"\nRunning analysis for mode: {mode}")
        results1, results2 = compare_feature_importance_from_files(
            json_file1, json_file2, compas_file,
            label1=label1, label2=label2, mode=mode
        )
    
    print("\nComparison complete. Results saved to 'comparison_visualizations/' directory.")