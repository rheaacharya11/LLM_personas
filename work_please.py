import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import os
from typing import Dict, List, Tuple, Set, Any, Union

class FairnessElicitationAlgorithm:
    """
    Implementation of the No-Regret Algorithm for Fairness Elicitation from the paper
    "An Algorithmic Framework for Fairness Elicitation"
    """
    
    def __init__(self, 
                 data_path: str, 
                 constraint_sets_path: str,
                 categorical_features: List[str] = None,
                 target_column: str = 'two_year_recid',
                 time_horizon: int = 1000,
                 C_lambda: float = 10.0,
                 C_tau: float = 10.0):
        """
        Initialize the algorithm with data and parameters.
        
        Args:
            data_path: Path to the data file (parquet format)
            constraint_sets_path: Path to the constraint sets JSON file
            categorical_features: List of categorical features to one-hot encode
            target_column: Name of the target column
            time_horizon: Number of iterations for the algorithm (T)
            C_lambda: Upper bound on lambda parameters (for fairness constraints)
            C_tau: Upper bound on tau parameter (for sum of violations)
        """
        self.data_path = data_path
        self.constraint_sets_path = constraint_sets_path
        self.categorical_features = categorical_features or ['sex', 'race', 'c_charge_degree']
        self.target_column = target_column
        self.time_horizon = time_horizon
        self.C_lambda = C_lambda
        self.C_tau = C_tau
        
        # Load and preprocess data
        self.load_data()
        
        # Load constraint sets
        self.load_constraint_sets()
        
        # Initialize parameters
        self.n = len(self.X)  # Number of samples
        self.d = self.X.shape[1]  # Number of features
        
        # Initialize algorithm state
        self.theta = np.zeros((self.n, self.n))  # For lambda updates
        self.tau = 0  # Initialize tau
        
        # Storage for algorithm trajectory
        self.errors = []
        self.fairness_violations = []
        self.models = []
    
    def load_data(self):
        # Load data
        df = pd.read_parquet(self.data_path)
        
        # Extract target BEFORE any processing
        self.y = df[self.target_column].values
        
        # Save original indices
        self.original_df = df.copy()
        
        # Create a copy for processing
        df_processed = df.copy()
        
        # Identify categorical columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
        print(f"Categorical columns to encode: {categorical_columns}")
        
        # Drop target column from features
        if self.target_column in df_processed.columns:
            df_processed = df_processed.drop(self.target_column, axis=1)
        
        # One-hot encode categorical features
        if categorical_columns:
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            categorical_data = df_processed[categorical_columns].fillna('Unknown')
            encoded_data = encoder.fit_transform(categorical_data)
            
            # Get feature names
            encoded_feature_names = []
            for i, feature in enumerate(categorical_columns):
                categories = encoder.categories_[i][1:]
                encoded_feature_names.extend([f"{feature}_{category}" for category in categories])
            
            # Drop original categorical columns
            df_processed = df_processed.drop(categorical_columns, axis=1)
            
            # Add encoded columns
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=df_processed.index)
            df_processed = pd.concat([df_processed, encoded_df], axis=1)
        
        # Ensure all columns are numeric
        for col in df_processed.columns:
            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Fill NaN values
        df_processed = df_processed.fillna(0)
        
        # Convert to numpy array
        self.X = df_processed.values
        
        print(f"Loaded data with {self.X.shape[0]} samples and {self.X.shape[1]} features")
        print(f"Label array shape: {self.y.shape}")
    def load_constraint_sets(self):
        """Load fairness constraint sets from JSON file"""
        with open(self.constraint_sets_path, 'r') as f:
            constraint_data = json.load(f)
        
        # Initialize constraint sets
        self.constraints = set()
        self.w_ij = {}  # Weight for each constraint (proportion of judges)
        
        # Count total number of judges
        judge_ids = list(constraint_data.keys())
        num_judges = len(judge_ids)
        
        # Each judge was presented 50 pairs but only selected a few
        pairs_per_judge = 50  # Number of pairs presented to each judge
        total_pairs_presented = pairs_per_judge * num_judges
        self.A = total_pairs_presented  # A is the total number of pairs presented
        
        # Process each judge's constraints
        for judge_id, pairs in constraint_data.items():
            for pair in pairs:
                # Convert to tuple for set operations
                constraint = tuple(pair)
                self.constraints.add(constraint)
                
                # Update weights (proportion of judges with this constraint)
                if constraint not in self.w_ij:
                    self.w_ij[constraint] = 1 / num_judges
                else:
                    self.w_ij[constraint] += 1 / num_judges
        
        avg_pairs_selected = sum(len(pairs) for pairs in constraint_data.values()) / num_judges
        print(f"Loaded {len(self.constraints)} unique constraints from {num_judges} judges")
        print(f"Judges selected {avg_pairs_selected:.2f} pairs on average out of {pairs_per_judge} presented")
    
    def cost_sensitive_oracle(self, costs):
        """
        Implement the cost-sensitive classification oracle.
        Returns a model that minimizes the weighted classification cost.
        
        Args:
            costs: Array of shape (n, 2) with costs[i, 0] and costs[i, 1] representing
                  the cost of classifying i as 0 or 1 respectively.
        
        Returns:
            A trained model that minimizes the weighted cost.
        """
        # We'll use logistic regression as our base classifier
        model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000, C=1.0)
        
        # Sample weights are the difference between costs for misclassification
        # Higher weight means more important to classify correctly
        sample_weights = np.abs(costs[:, 1] - costs[:, 0])
        
        # Train the model with sample weights
        model.fit(self.X, self.y, sample_weight=sample_weights)
        
        return model
    
    def compute_prediction_probs(self, model):
        """Compute prediction probabilities for each sample"""
        return model.predict_proba(self.X)[:, 1]
    
    def compute_error(self, model):
        """Compute classification error of the model"""
        y_pred = model.predict(self.X)
        return 1 - accuracy_score(self.y, y_pred)
    
    def compute_fairness_violation(self, probs, alpha_ij, gamma):
        """
        Compute the fairness violation for given probabilities and parameters.
        
        Args:
            probs: Prediction probabilities for all samples.
            alpha_ij: Excess fairness violation terms.
            gamma: Allowed fairness violation buffer.
            
        Returns:
            The maximum weighted fairness violation across all constraints.
        """
        violations = []
        
        for (i, j) in self.constraints:
            try:
                # Use indices directly as row numbers
                idx_i, idx_j = i, j
                
                # Make sure indices are within range
                if idx_i >= len(probs) or idx_j >= len(probs):
                    print(f"Warning: Constraint indices ({i}, {j}) out of range, skipping")
                    continue
                
                # Calculate violation for this pair
                diff = probs[idx_i] - probs[idx_j]
                
                # Only count violations where i is predicted more positively than j
                if diff > (alpha_ij.get((i, j), 0) + gamma):
                    violation = diff - alpha_ij.get((i, j), 0) - gamma
                    weight = self.w_ij.get((i, j), 0)
                    violations.append(weight * violation)
            except Exception as e:
                print(f"Error processing constraint ({i}, {j}): {e}")
                continue
       
        # Return the maximum violation (if any), otherwise 0
        return max(violations) if violations else 0.0
    
    def best_response_primal(self, lambda_t, tau_t, gamma):
        """
        Computes the best response for the primal player (D_t, alpha_t).
        
        Args:
            lambda_t: Current lambda values for all pairs.
            tau_t: Current tau value.
            gamma: Fairness violation buffer.
            
        Returns:
            D_t: A model (classifier).
            alpha_t: Dictionary of alpha values for each constraint pair.
        """
        # Initialize costs for each sample
        costs = np.zeros((len(self.y), 2))
        
        # Set costs based on true labels and lambda values
        for i in range(len(self.y)):
            # Classification costs
            if self.y[i] == 0:
                costs[i, 0] = 0
                costs[i, 1] = 1/len(self.y)
            else:
                costs[i, 0] = 1/len(self.y)
                costs[i, 1] = 0
                
            # Add costs from lambda terms
            for j in range(len(self.y)):
                if (i, j) in lambda_t:
                    costs[i, 1] += lambda_t[(i, j)]
                if (j, i) in lambda_t:
                    costs[i, 1] -= lambda_t[(j, i)]
        
        # Get classifier from the cost-sensitive oracle
        D_t = self.cost_sensitive_oracle(costs)
        
        # Compute alpha values
        alpha_t = {}
        for (i, j) in self.constraints:
            # If tau * w_ij/|A| - lambda_ij ≤ 0 then alpha_ij = 1, otherwise 0
            weight = self.w_ij.get((i, j), 0)
            
            if tau_t * weight / self.A <= lambda_t.get((i, j), 0):
                alpha_t[(i, j)] = 1
            else:
                alpha_t[(i, j)] = 0
                
        return D_t, alpha_t
    
    def update_dual_variables(self, lambda_t, tau_t, D_t, alpha_t, gamma, t):
        # Get prediction probabilities
        probs = self.compute_prediction_probs(D_t)
        
        # Use smaller step size for lambda updates
        mu_lambda = 1 / (self.C_lambda * np.sqrt(np.log(self.n) / self.time_horizon))
        
        # Update only for actual constraints (not all pairs)
        lambda_new = lambda_t.copy()
        
        # Sparse update - only update existing constraints
        for (i, j) in self.constraints:
            # Calculate gradient for this pair
            gradient = probs[i] - probs[j] - alpha_t.get((i, j), 0) - gamma
            
            # Simple multiplicative update instead of full softmax
            lambda_new[(i, j)] = min(
                self.C_lambda, 
                lambda_t.get((i, j), 0) * np.exp(mu_lambda * gradient)
            )
        
        # Update tau using online gradient descent
        mu_tau = self.C_tau / np.sqrt(self.time_horizon)
        
        # Compute gradient for tau (simplified)
        tau_gradient = sum(self.w_ij.get((i, j), 0) * alpha_t.get((i, j), 0) 
                        for (i, j) in self.constraints) / self.A - gamma
        
        # Update tau with projection
        tau_new = max(0, min(self.C_tau, tau_t + mu_tau * tau_gradient))
        
        return lambda_new, tau_new
        
    def run(self, gamma_values=None):
        """
        Run the algorithm for multiple gamma values.
        
        Args:
            gamma_values: List of gamma values to test. If None, uses default values.
            
        Returns:
            Results for different gamma values.
        """
        if gamma_values is None:
            gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        results = {}
        
        for gamma in gamma_values:
            print(f"Running algorithm with gamma = {gamma}")
            
            # Reset algorithm state
            self.theta = np.zeros((self.n, self.n))
            self.tau = 0
            
            # Initialize storages for this gamma
            errors = []
            fairness_violations = []
            D_avg = None
            
            # Initialize lambda and tau
            lambda_t = {pair: 0.0 for pair in self.constraints}
            tau_t = 0.0
            
            # Run the algorithm for T iterations
            for t in range(1, self.time_horizon + 1):
                print(f"Starting iteration {t}")  # Add this line
                
                try:
                    # Best response of primal player
                    print(f"  Computing best response for iteration {t}")
                    D_t, alpha_t = self.best_response_primal(lambda_t, tau_t, gamma)
                    
                    # Compute metrics
                    print(f"  Computing metrics for iteration {t}")
                    error = self.compute_error(D_t)
                    probs = self.compute_prediction_probs(D_t)
                    fairness_violation = self.compute_fairness_violation(probs, alpha_t, gamma)
                    
                    # Update dual variables
                    print(f"  Updating dual variables for iteration {t}")
                    lambda_t, tau_t = self.update_dual_variables(lambda_t, tau_t, D_t, alpha_t, gamma, t)
                    
                    print(f"  Completed iteration {t}: Error = {error:.4f}, Violation = {fairness_violation:.4f}")
                except Exception as e:
                    print(f"Error in iteration {t}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            # Store results for this gamma
            results[gamma] = {
                'errors': errors,
                'fairness_violations': fairness_violations,
                'final_error': errors[-1],
                'final_fairness_violation': fairness_violations[-1]
            }
        
        return results
    
    def plot_trajectory(self, results, gamma_values=None):
        """
        Plot the trajectory of the algorithm for different gamma values.
        
        Args:
            results: Results from running the algorithm.
            gamma_values: List of gamma values to plot. If None, uses all available in results.
        """
        if gamma_values is None:
            gamma_values = list(results.keys())
        
        plt.figure(figsize=(12, 8))
        plt.title("Single-Subject Trajectory for Various γ's")
        plt.xlabel("error(t)")
        plt.ylabel("max violation(t)")
        
        # Add horizontal lines at 0.1 intervals
        for y in np.arange(0, 1.1, 0.1):
            plt.axhline(y=y, color='r', linestyle='-', alpha=0.3)
        
        # Plot trajectory for each gamma
        for gamma in gamma_values:
            if gamma in results:
                errors = results[gamma]['errors']
                violations = results[gamma]['fairness_violations']
                plt.plot(errors, violations, label=f"γ = {gamma}")
        
        plt.legend()
        plt.grid(True)
        plt.savefig("single_subject_trajectory.png")
        plt.show()
    
    def plot_pareto_curves(self, results):
        """
        Plot the Pareto curves for different subjects/gamma values.
        
        Args:
            results: Results from running the algorithm.
        """
        plt.figure(figsize=(12, 8))
        plt.title("Variability of Subject Pareto Curves")
        plt.xlabel("error")
        plt.ylabel("max violation")
        
        # Extract final errors and violations for each gamma
        gammas = list(results.keys())
        errors = [results[gamma]['final_error'] for gamma in gammas]
        violations = [results[gamma]['final_fairness_violation'] for gamma in gammas]
        
        # Plot the Pareto curve
        plt.plot(errors, violations)
        
        plt.grid(True)
        plt.savefig("pareto_curves.png")
        plt.show()

# Usage example:
def main():
    algorithm = FairnessElicitationAlgorithm(
        data_path="data/processed/compas_train.parquet",
        constraint_sets_path="constraint_sets/binary_personas/constraint_sets.json",
        time_horizon=1000,
        C_lambda=10.0,
        C_tau=10.0
    )
    
    # Run the algorithm for different gamma values
    gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = algorithm.run(gamma_values)
    
    # Plot the results
    algorithm.plot_trajectory(results)
    algorithm.plot_pareto_curves(results)

if __name__ == "__main__":
    main()