import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set

def load_constraints(filepath: str) -> Dict[Tuple[int, int], List[int]]:
    """
    Load constraints from JSON file, outputting a dictionary!
    
    Returns:
        Dictionary mapping (i, j) pair tuples to list of judge IDs who said these individuals
        should be treated equally
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert string tuple keys to actual tuples
    constraints = {}
    for key, judges in data.items():
        # Parse the key format - handle multiple possible formats
        if key.startswith('(') and key.endswith(')'):
            clean_key = key.strip('(').strip(')').split(',')
        elif key.startswith('"(') and key.endswith(')"'):
            clean_key = key.strip('"(').strip(')"').split(', ')
        else:
            # Try to handle any other format
            key = key.replace('(', '').replace(')', '').replace('"', '')
            clean_key = key.split(',')
        
        # Convert to integers
        try:
            i = int(clean_key[0].strip())
            j = int(clean_key[1].strip())
            constraints[(i, j)] = judges
        except (ValueError, IndexError) as e:
            print(f"Error parsing key {key}: {e}")
    
    print(f"Loaded {len(constraints)} constraint pairs")
    return constraints

def compute_constraint_weights(constraints: Dict[Tuple[int, int], List[int]], 
                           judges_per_pair: int = 10, 
                           total_judges: int = 1000) -> Dict[Tuple[int, int], float]:
    """
    Compute the weight for each constraint based on fraction of judges who selected it.
        
    Returns:
        Dictionary mapping (i, j) pairs to weights
    """
    # Compute weights
    weights = {}
    for pair, judges in constraints.items():
        weights[pair] = len(judges) / judges_per_pair
    
    return weights

def load_training_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from parquet file, handling categorical features.
    
    Args:
        filepath: Path to the parquet file containing training data
        
    Returns:
        X: Feature matrix
        y: Target vector
    """
    # Load the data
    df = pd.read_parquet(filepath)
    
    id_to_index = {}
    if 'id' in df.columns:
        for idx, id_val in enumerate(df['id']):
            id_to_index[int(id_val)] = idx
        # Remove ID column for feature processing
        feature_df = df.drop(columns=['id'])
    else:
        # Use row index as ID if no ID column exists
        for idx in range(len(df)):
            id_to_index[idx] = idx
        feature_df = df
    
    # Assume last column is the target
    X_df = feature_df.iloc[:, :-1]
    y = feature_df.iloc[:, -1].values
    
    # Handle categorical features
    categorical_columns = X_df.select_dtypes(include=['object', 'category']).columns
    numerical_columns = X_df.select_dtypes(include=['number']).columns
    
    # Apply one-hot encoding to categorical columns
    if not categorical_columns.empty:
        X_categorical = pd.get_dummies(X_df[categorical_columns], drop_first=True)
        X_numerical = X_df[numerical_columns]
        X_processed = pd.concat([X_numerical, X_categorical], axis=1)
    else:
        X_processed = X_df
    
    X = X_processed.values
    
    print(f"Processed data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"ID mapping contains {len(id_to_index)} entries")
    
    return X, y, id_to_index

def get_constraint_pairs(constraints: Dict[Tuple[int, int], List[int]], subset_size: int = None) -> Set[Tuple[int, int]]:
    """
    Get all unique constraint pairs or a random subset.
    """
    pairs = set(constraints.keys())
    
    if subset_size is not None and subset_size < len(pairs):
        pairs = set(np.random.choice(list(pairs), size=subset_size, replace=False))
    
    return pairs
