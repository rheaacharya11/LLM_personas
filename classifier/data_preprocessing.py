import pandas as pd
import numpy as np
from typing import Dict, Set, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CompasDataProcessor:
    def __init__(self, compas_file_path: str, stakeholder_file_path: str):
        """
        Initialize the COMPAS data processor.
        
        Args:
            compas_file_path: Path to the COMPAS dataset CSV
            stakeholder_file_path: Path to the persona preferences CSV
        """
        self.compas_df = pd.read_csv(compas_file_path)
        self.stakeholder_df = pd.read_csv(stakeholder_file_path)

    def preprocess_compas_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """
        Preprocess the COMPAS dataset.
        
        Returns:
            X: Feature matrix
            y: Target labels (recidivism)
            feature_names: List of feature names
            individual_ids: List of individual IDs
        """
        # Select relevant features 
        feature_columns = [
            'sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count', 
            'juv_other_count', 'priors_count', 'c_charge_degree'
        ]
        
        # One-hot encode categorical features
        df_processed = pd.get_dummies(
            self.compas_df[feature_columns], 
            columns=['sex', 'race', 'c_charge_degree'],
            drop_first=True
        )
        
        # Store feature names
        feature_names = df_processed.columns.tolist()
        
        # Extract features
        X = df_processed.values
        
        # Extract target (assuming 'two_year_recid' is the column for recidivism)
        y = self.compas_df['two_year_recid'].values
        
        # Store individual IDs (assuming 'id' is the ID column)
        individual_ids = self.compas_df['id'].values.tolist()
        
        # Standardize numerical features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y, feature_names, individual_ids

    def extract_fairness_constraints(self) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], float]]:
        """
        Process stakeholder preferences to extract fairness constraints.
        
        Returns:
            constraint_pairs: Set of pairs (i,j) where stakeholders express fairness constraints
            weights: Dictionary mapping constraint pairs to weights
        """
        # Initialize constraint pairs and weights
        constraint_pairs = set()
        weights = {}
        
        # Create a mapping from individual IDs to indices
        id_to_index = {id_val: idx for idx, id_val in enumerate(self.compas_df['id'].values)}
        
        # Process each row in the stakeholder preferences
        for _, row in self.stakeholder_df.iterrows():
            # Get the individual IDs for this row
            individual1_id = row['individual1_id']
            individual2_id = row['individual2_id']
            
            # Get the corresponding indices
            if individual1_id not in id_to_index or individual2_id not in id_to_index:
                continue  # Skip if either individual is not in the dataset
                
            i = id_to_index[individual1_id]
            j = id_to_index[individual2_id]
            
            # Get the choice type for this preference
            choice_type = row['choice_type']
            
            # Add constraints only for "equal" choice type
            if choice_type == 'equal':
                constraint_pairs.add((i, j))
                constraint_pairs.add((j, i))
                # Initialize weight if not present
                weights[(i, j)] = weights.get((i, j), 0) + 1
                weights[(j, i)] = weights.get((j, i), 0) + 1
            
            # No constraints added for "different" choice type
                
        # Normalize weights by the number of preferences for each pair
        pair_counts = {}
        for i, j in constraint_pairs:
            # Count how many stakeholders expressed a preference for this pair
            pair_key = (min(i, j), max(i, j))  # Canonical order to count unique pairs
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
            
        # Normalize weights
        normalized_weights = {}
        for (i, j), weight in weights.items():
            pair_key = (min(i, j), max(i, j))
            normalized_weights[(i, j)] = weight / pair_counts[pair_key]
            
        return constraint_pairs, normalized_weights

    def parse_stakeholder_preferences(self) -> pd.DataFrame:
        """
        Pre-process the stakeholder preferences by converting choice to choice_type.
        
        Returns:
            Processed stakeholder dataframe with standardized choice_type
        """
        # Make a copy to avoid modifying the original
        stakeholder_df = self.stakeholder_df.copy()
        
        # Map choice to choice_type if not already present
        if 'choice_type' not in stakeholder_df.columns:
            choice_map = {
                "Ok to treat differently, or no opinion": "different",
                "Should be treated exactly the same": "equal"
            }
            stakeholder_df['choice_type'] = stakeholder_df['choice'].map(choice_map)
            
        return stakeholder_df

def run_fairness_elicitation(compas_file_path: str, stakeholder_file_path: str):
    """
    Run the fairness elicitation algorithm on the COMPAS dataset.
    
    Args:
        compas_file_path: Path to the COMPAS dataset CSV
        stakeholder_file_path: Path to the stakeholder preferences CSV
    """
    # Initialize the data processor
    processor = CompasDataProcessor(compas_file_path, stakeholder_file_path)
    
    # Parse stakeholder preferences
    processor.stakeholder_df = processor.parse_stakeholder_preferences()
    
    # Preprocess the COMPAS data
    X, y, feature_names, individual_ids = processor.preprocess_compas_data()
    
    # Extract fairness constraints
    constraint_pairs, weights = processor.extract_fairness_constraints()
    
    print(f"Extracted {len(constraint_pairs)} constraint pairs from stakeholder preferences")
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(X)), test_size=0.2, random_state=42
    )
    
    # Update constraint indices to match the training set indices
    train_id_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(train_idx)}
    
    train_constraint_pairs = set()
    train_weights = {}
    
    for i, j in constraint_pairs:
        if i in train_id_to_new_idx and j in train_id_to_new_idx:
            new_i = train_id_to_new_idx[i]
            new_j = train_id_to_new_idx[j]
            train_constraint_pairs.add((new_i, new_j))
            train_weights[(new_i, new_j)] = weights[(i, j)]
    
    # Import the FairnessElicitation class
    from fairness_elicitation import FairnessElicitation
    
    # Initialize the fairness elicitation algorithm
    fairness = FairnessElicitation(
        X=X_train,
        y=y_train,
        constraint_pairs=train_constraint_pairs,
        weights=train_weights,
        gamma=0.1,  # Adjust as needed
        eta=0.05,   # Adjust as needed
        C_lambda=1.0,
        C_tau=1.0
    )
    
    # Generate a Pareto curve for different gamma values
    gamma_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    pareto_curve = fairness.pareto_curve(gamma_values)
    
    # Print the results
    print("\nPareto curve (accuracy vs. fairness violation):")
    for gamma, (accuracy, violation) in zip(gamma_values, pareto_curve):
        print(f"Gamma: {gamma:.2f}, Accuracy: {accuracy:.4f}, Fairness Violation: {violation:.4f}")
    
    # Run the algorithm with the default gamma
    print("\nRunning fairness elicitation with gamma = 0.1...")
    fairness.gamma = 0.1
    D = fairness.no_regret_dynamics(T=1000)
    
    # Evaluate results
    accuracy = fairness.evaluate_accuracy(D)
    violation, violations_by_pair = fairness.evaluate_fairness_violation(D)
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Overall fairness violation: {violation:.4f}")
    
    # Visualize the most violated constraints
    most_violated = sorted(violations_by_pair.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\nMost violated constraints:")
    for (i, j), violation_val in most_violated:
        if violation_val > 0:
            i_orig = train_idx[i]
            j_orig = train_idx[j]
            i_id = individual_ids[i_orig]
            j_id = individual_ids[j_orig]
            print(f"Constraint ({i_id}, {j_id}): Violation = {violation_val:.4f}, Weight = {train_weights.get((i, j), 0):.4f}")
    
    return fairness, D, X_test, y_test, test_idx