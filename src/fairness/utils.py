import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union

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

def load_training_data(filepath: str, 
                      test_filepath: Optional[str] = None, 
                      id_column: Optional[str] = 'id',
                      target_column: Optional[str] = 'two_year_recid') -> Union[Tuple[np.ndarray, np.ndarray, Dict[int, int]], 
                                                                               Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]]:
    """
    Load training data from parquet file, handling categorical features.
    
    Args:
        filepath: Path to the parquet file containing training data
        
    Returns:
        X: Feature matrix
        y: Target vector
    """
    # Load the data
  # Load training data
    df_train = pd.read_parquet(filepath)
    
    # Create ID to index mapping
    id_to_index = {}
    if id_column and id_column in df_train.columns:
        # If specific ID column is provided, use it
        for idx, id_val in enumerate(df_train[id_column]):
            try:
                id_to_index[int(id_val)] = idx
            except (ValueError, TypeError):
                # Handle non-integer IDs
                id_to_index[id_val] = idx
        # Keep ID column for processing but will remove before final feature matrix
        feature_df = df_train.copy()
    else:
        # Use row index as ID if no ID column exists
        for idx in range(len(df_train)):
            id_to_index[idx] = idx
        feature_df = df_train.copy()
    
    # Extract target vector
    if target_column and target_column in df_train.columns:
        y_train = df_train[target_column].values
        # Keep all columns except the target as potential features
        feature_cols = [col for col in df_train.columns if col != target_column]
        features_df = df_train[feature_cols]
    else:
        # Fallback to last column if specific target column not found
        print(f"Warning: '{target_column}' column not found, using last column as target")
        y_train = df_train.iloc[:, -1].values
        features_df = df_train.iloc[:, :-1]
    
    # Explicitly convert y to integer type
    y_train = y_train.astype(int)
    
    # Handle categorical features with one-hot encoding
    categorical_columns = features_df.select_dtypes(include=['object', 'category']).columns
    numerical_columns = features_df.select_dtypes(include=['number']).columns
    
    # Process test data if provided
    if test_filepath:
        df_test = pd.read_parquet(test_filepath)
        
        # Extract target vector for test data
        if target_column and target_column in df_test.columns:
            y_test = df_test[target_column].values
            # Keep all columns except the target as potential features for consistency
            test_features_df = df_test[[col for col in feature_cols if col in df_test.columns]]
        else:
            print(f"Warning: '{target_column}' column not found in test data, using last column as target")
            y_test = df_test.iloc[:, -1].values
            test_features_df = df_test.iloc[:, :-1]
        
        y_test = y_test.astype(int)
        
        # Apply consistent one-hot encoding to both train and test sets
        if not categorical_columns.empty:
            # For categorical features, we need to ensure consistent encoding
            combined_df = pd.concat([features_df, test_features_df], axis=0)
            combined_dummies = pd.get_dummies(combined_df[categorical_columns], drop_first=True)
            
            # Split back into train and test
            X_train_cat = combined_dummies.iloc[:len(features_df)]
            X_test_cat = combined_dummies.iloc[len(features_df):]
            
            # Combine with numerical features
            if not numerical_columns.empty:
                X_train_num = features_df[numerical_columns]
                X_test_num = test_features_df[numerical_columns]
                
                X_train_processed = pd.concat([X_train_num, X_train_cat], axis=1)
                X_test_processed = pd.concat([X_test_num, X_test_cat], axis=1)
            else:
                X_train_processed = X_train_cat
                X_test_processed = X_test_cat
        else:
            # No categorical features to encode
            X_train_processed = features_df
            X_test_processed = test_features_df
        
        # Remove ID column if it's in the feature matrix
        if id_column and id_column in X_train_processed.columns:
            X_train_processed = X_train_processed.drop(columns=[id_column])
        if id_column and id_column in X_test_processed.columns:
            X_test_processed = X_test_processed.drop(columns=[id_column])
        
        # Convert to numpy arrays
        X_train = X_train_processed.values
        X_test = X_test_processed.values
        
        print(f"Processed training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Processed test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"ID mapping contains {len(id_to_index)} entries")
        
        return X_train, y_train, X_test, y_test, id_to_index
    
    # No test data - process only training data
    if not categorical_columns.empty:
        X_categorical = pd.get_dummies(features_df[categorical_columns], drop_first=True)
        
        if not numerical_columns.empty:
            X_numerical = features_df[numerical_columns]
            X_processed = pd.concat([X_numerical, X_categorical], axis=1)
        else:
            X_processed = X_categorical
    else:
        X_processed = features_df
    
    # Remove ID column if it's in the feature matrix
    if id_column and id_column in X_processed.columns:
        X_processed = X_processed.drop(columns=[id_column])
    
    # Convert to numpy array
    X_train = X_processed.values
    
    print(f"Processed data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"ID mapping contains {len(id_to_index)} entries")
    
    return X_train, y_train, id_to_index

def get_constraint_pairs(constraints: Dict[Tuple[int, int], List[int]], subset_size: int = None) -> Set[Tuple[int, int]]:
    """
    Get all unique constraint pairs or a random subset.
    """
    pairs = set(constraints.keys())
    
    if subset_size is not None and subset_size < len(pairs):
        pairs = set(np.random.choice(list(pairs), size=subset_size, replace=False))
    
    return pairs

def map_constraints_to_indices(constraints: Dict[Tuple[int, int], List[int]], 
                              id_to_index: Dict[int, int]) -> Dict[Tuple[int, int], List[int]]:
    """
    Map constraint pairs from original IDs to array indices.
    
    Args:
        constraints: Dictionary mapping (original_id_i, original_id_j) pairs to list of judge IDs
        id_to_index: Mapping from original IDs to array indices
        
    Returns:
        Dictionary mapping (index_i, index_j) pairs to list of judge IDs
    """
    mapped_constraints = {}
    for (id_i, id_j), judges in constraints.items():
        if id_i in id_to_index and id_j in id_to_index:
            mapped_constraints[(id_to_index[id_i], id_to_index[id_j])] = judges
        else:
            print(f"Warning: Could not map constraint ({id_i}, {id_j}) to indices")
    
    return mapped_constraints