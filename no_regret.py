import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import pickle
import os
from typing import Dict, List, Tuple, Set, Any, Union
TOL_COLORS = [
    "#117733",  # green
    "#332288",  # navy
    "#44AA99",  # teal
    "#88CCEE",  # light blue
    "#DDCC77",  # sand
    "#CC6677",  # red-pink
    "#AA4499",  # purple
    "#882255",  # wine
    "#661100",  # brown
    "#999933",  # olive
]
# Import River library components
from river import optim
from river import utils

# exponentiated gradient descent
class Hedge:
    def __init__(self, lr=0.1, weight_bound=(0, float('inf'))):
        self.learning_rate = learning_rate
        self.min_bound, self.max_bound = weight_bound
    
    def step(self, weight, gradient):
        # Multiplicative update
        new_weight = weight * np.exp(self.learning_rate * gradient)
        # Apply bounds
        return min(max(new_weight, self.min_bound), self.max_bound)

class FairnessElicitationAlgorithm:
    """
    Implementation of the No-Regret Algorithm for Fairness Elicitation from the paper
    "An Algorithmic Framework for Fairness Elicitation" using River's no-regret optimizers
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
        """
        self.data_path = data_path
        self.constraint_sets_path = constraint_sets_path
        self.categorical_features = categorical_features or ['sex', 'race', 'c_charge_degree']
        self.target_column = target_column
        self.time_horizon = time_horizon
        self.C_lambda = C_lambda
        self.C_tau = C_tau
        self.eta = 0
        
        # Load and preprocess data
        self.load_data()
        
        # Load constraint sets
        self.load_constraint_sets()
        
        # Initialize parameters
        self.n = len(self.X)  # Number of samples
        self.d = self.X.shape[1]  # Number of features
        
        # Initialize River optimizers for no-regret learning
        self.lambda_optimizers = {}
        self.tau_optimizer = None
        
    def load_data(self):
        # Load data 
        df = pd.read_parquet(self.data_path)
        
        # Extract target : 'two_year_recid'
        self.y = df[self.target_column].values
        
        # Save original indices
        self.original_df = df.copy()
        
        # Create a copy for processing
        df_processed = df.copy()
        self.id_col = 'id'  # or whatever the column name is
        self.ids = df[self.id_col].tolist()
        self.id_to_index = {id_: idx for idx, id_ in enumerate(self.ids)}
        self.index_to_id = {idx: id_ for idx, id_ in enumerate(self.ids)}
        # Identify categorical columns
        categorical_columns = [col for col in self.categorical_features if col in df_processed.columns]
        print(f"Categorical columns to encode: {categorical_columns}")
        
        # Drop target column from features
        if self.target_column in df_processed.columns:
            df_processed = df_processed.drop(self.target_column, axis=1)
        print(df_processed[categorical_columns].head())
        # One-hot encode categorical features
        if categorical_columns:
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            # Assuming `encoder` is your OneHotEncoder
            
            categorical_data = df_processed[categorical_columns].fillna('Unknown')
            encoded_data = encoder.fit_transform(categorical_data)
            with open("onehot_encoder.pkl", "wb") as f:
                pickle.dump(encoder, f)

            print("\nCategories Detected by Encoder (first few):")
            for feature, cats in zip(categorical_columns, encoder.categories_):
                print(f"{feature}: {cats[:5]} (Total categories: {len(cats)})")

            
            print("\nEncoded Data Shape:")
            print(encoded_data.shape)
            # Get feature names
            encoded_feature_names = []
            for i, feature in enumerate(categorical_columns):
                categories = encoder.categories_[i][1:]
                encoded_feature_names.extend([f"{feature}_{category}" for category in categories])
            
            with open("encoded_feature_names.pkl", "wb") as f:
                pickle.dump(encoded_feature_names, f)
            print("\nEncoded Feature Names (first few):")
            print(encoded_feature_names[:10])
            # Drop original categorical columns
            df_processed = df_processed.drop(categorical_columns, axis=1)
            
            # Add encoded columns
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=df_processed.index)
            df_processed = pd.concat([df_processed, encoded_df], axis=1)
            print("\nFinal DataFrame after Adding Encoded Columns:")
            print(df_processed.head())
        # Ensure all columns are numeric
        for col in df_processed.columns:
            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Fill NaN values with 0's
        df_processed = df_processed.fillna(0)
        
        # Convert to numpy array
        self.X = df_processed.values
        # Save final training column order
        self.train_columns = df_processed.columns.tolist()

        # Save to file for later use in testing
        with open("train_feature_columns.pkl", "wb") as f:
            pickle.dump(self.train_columns, f)
        
        print(f"Loaded data with {self.X.shape[0]} samples and {self.X.shape[1]} features")
        print(f"Label array shape: {self.y.shape}")

    def load_constraint_sets(self, judge_id=None, custom_constraints=None):
        """
        Load fairness constraint sets from JSON file

        Args:
            judge_id: Optional specific judge ID or a collection (list/set) of judge IDs to use.
                    If None, uses multiple judges.
            custom_constraints: Optional custom constraints to override file-based constraints.
        """

        if custom_constraints is not None:
            self.constraints = set()
            self.w_ij = {}
            
            for constraint in custom_constraints:
                pair = tuple(constraint["pair"])
                pair_swapped = tuple(reversed(pair))
                self.constraints.update([pair, pair_swapped])
                
                for p in [pair, pair_swapped]:
                    self.w_ij[p] = self.w_ij.get(p, 0) + constraint["weight"]  # or normalized weight if needed

            self.A = len(custom_constraints)  # Total number of presented pairs
            print(f"Loaded {len(self.constraints)} unique constraints from custom constraint set")
            return

        with open(self.constraint_sets_path, 'r') as f:
            constraint_data = json.load(f)

        # Print a few entries for inspection
        for key, value in list(constraint_data.items())[:5]:
            print(key, value)

        if isinstance(constraint_data, list):
            print("Number of entries:", len(constraint_data))
        elif isinstance(constraint_data, dict):
            print("Number of entries:", len(constraint_data))
        else:
            print("Unexpected data type.")

        # Initialize constraint sets
        self.constraints = set()
        self.w_ij = {}  # Weight for each constraint

        # If specific judge(s) provided
        if judge_id is not None:
            # If a single judge ID is provided, convert it to a list
            if isinstance(judge_id, (str, int)):
                judge_ids = [str(judge_id)]
            else:
                # Convert all judge IDs to strings for consistency
                judge_ids = [str(j) for j in judge_id]

            total_pairs = 0
            for j in judge_ids:
                if j not in constraint_data:
                    raise ValueError(f"Judge ID {j} not found in constraint data")
                focused_constraints = constraint_data[j]
                total_pairs += len(focused_constraints)
                for constraint in focused_constraints:
                    pair = tuple(constraint['pair'])
                    self.constraints.add(pair)
                    # Multiply by 10000 as before (adjust if needed)
                    self.w_ij[pair] = self.w_ij.get(pair, 0) + constraint['weight'] * 1

            pairs_per_judge = 200  # Adjust based on your experiment
            self.A = pairs_per_judge * len(judge_ids)
            print(f"Loaded {len(self.constraints)} constraints from judges {judge_ids}")
            print(f"Judges selected {total_pairs} pairs out of {pairs_per_judge * len(judge_ids)} presented")
        
        else:
            # Multiple judges case (default behavior)
            import random
            judge_ids = list(constraint_data.keys())
            num_judges = len(judge_ids)
            mean_judges_per_pair = 100
            
            pairs_per_judge = 100  # Number of pairs presented to each judge
            total_pairs_presented = pairs_per_judge * num_judges
            self.A = 1000  # A is the total number of pairs presented (adjust as needed)
            
            for j_id, constraints in constraint_data.items():
                for constraint in constraints:
                    pair = tuple(constraint['pair'])
                    pair_swapped = tuple(reversed(pair))
                    self.constraints.update([pair, pair_swapped])
                    # Weighted average of pairs
                    for p in [pair, pair_swapped]:
                        if p not in self.w_ij:
                            self.w_ij[p] = constraint['weight'] / mean_judges_per_pair
                        else:
                            self.w_ij[p] += constraint['weight'] / mean_judges_per_pair
            
            avg_pairs_selected = sum(len(pairs) for pairs in constraint_data.values()) / num_judges
            print(f"Loaded {len(self.constraints)} unique constraints from {num_judges} judges")
            print(f"Judges selected {avg_pairs_selected:.2f} pairs on average out of {pairs_per_judge} presented")
        '''
        def initialize_optimizers(self):
            """Initialize River's no-regret optimizers for each constraint and tau"""
            
            # For lambda values, use exponentiated gradient descent (Hedge algorithm)
            for pair in self.constraints:
                # Initial learning rate inversely proportional to C_lambda and sqrt(log(n))
                # This matches the theoretical bound in the paper
                lr = 1.0 / (self.C_lambda * np.sqrt(np.log(self.n)))
                
                # Create exponential weights optimizer with clipping bounds [0, C_lambda]
                self.lambda_optimizers[(i, j)] = optim.AdaGrad(lr=lr, eps=1e-8)
            # For tau, use online gradient descent
            # Initial learning rate inversely proportional to C_tau
            self.tau_optimizer = optim.SGD(
                lr=1.0 / (self.C_tau * np.sqrt(self.time_horizon)),
                bounds=(0, self.C_tau)
            )
        '''
    
    def cost_sensitive_oracle(self, costs, C=1.0):
        """
        Implement the cost-sensitive classification oracle.
        Returns a model that minimizes the weighted classification cost.
        
        Args:
            costs: Array of shape (n, 2) with costs[i, 0] and costs[i, 1] representing
                  the cost of classifying i as 0 or 1 respectively.
        
        Returns:
            A trained model that minimizes the weighted cost.
        """
        # Calculate the cost difference (positive means bias toward 0, negative toward 1)
        cost_difference = costs[:, 0] - costs[:, 1]
        
        # Create sample weights based on the absolute cost difference
        sample_weights = np.abs(cost_difference) 
        
        # Create target values based on the sign of cost difference
        # If c₀ > c₁, then we want to predict 1, otherwise 0
        target_values = (cost_difference > 0).astype(int)
        
        # Use a standard classifier with sample weights
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',  # Faster for smaller datasets
            max_iter=1000,
            class_weight=None,  # We handle through sample_weights
            random_state=11
        )
        
        # Train the model with sample weights
        model.fit(self.X, target_values, sample_weight=sample_weights)
        
        return model
    
    def compute_prediction_probs(self, model):
        """Compute prediction probabilities for each sample"""
        # returns second column in array, aka the probability of it being 1
        return model.predict_proba(self.X)[:, 1]
    
    def compute_error(self, model):
        """Compute classification error of the model"""

        y_pred = model.predict(self.X)
        return 1 - accuracy_score(self.y, y_pred)
        # accuracy_score is the proportion of correctly predicted instances
    
    def compute_fairness_violation(self, probs, alpha_ij, gamma):
        total_violation = 0.0
        individual_violations = {}
        max_violation = 0.0
        processed_pairs = set()
        violation_count = 0
        
        for (i, j) in self.constraints:
            pair_key = tuple(sorted([i, j]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            try:
                abs_diff = abs(probs[self.id_to_index[i]] - probs[self.id_to_index[j]])
                alpha_ij_value = max(alpha_ij.get((i, j), 0), alpha_ij.get((j, i), 0))
                violation = max(0, abs_diff - gamma)
                
                if violation > 0:
                    violation_count += 1
                    individual_violations[pair_key] = (abs_diff, violation)
                    max_violation = max(max_violation, violation)
                
                weight = self.w_ij.get((i, j), 0) + self.w_ij.get((j, i), 0)
                weighted_violation = weight * violation
                total_violation += weighted_violation
                    
            except Exception as e:
                print(f"Error processing constraint {pair_key}: {e}")
        
        print(f"Violations exceeding γ={gamma}: {violation_count}/{len(processed_pairs)}")
        
        return total_violation / self.A, max_violation
        
    def best_response_primal(self, lambda_values, tau_value, gamma):
        """
        Computes the best response for the primal player (D_t, alpha_t).
        
        Args:
            lambda_values: Current lambda values for all pairs.
            tau_value: Current tau value.
            gamma: Fairness violation buffer.
            
        Returns:
            D_t: A model (classifier).
            alpha_t: Dictionary of alpha values for each constraint pair.
        """
        # Initialize costs for each sample
        costs = np.zeros((len(self.y), 2))
        
        # Set costs based on true labels and lambda values
        for id_ in self.ids:
            # Classification costs
            idx = self.id_to_index[id_]
            if self.y[idx] == 0:
                costs[self.id_to_index[id_], 0] = 0
                costs[self.id_to_index[id_], 1] = 1/len(self.y)
            else:
                costs[self.id_to_index[id_], 0] = 1/len(self.y)
                costs[self.id_to_index[id_], 1] = 0
                
            # Add costs from lambda terms
            for other_id in self.ids:
                if (id_, other_id) in lambda_values:
                    costs[idx, 1] += lambda_values[(id_, other_id)]
                if (other_id, id_) in lambda_values:
                    costs[idx, 1] -= lambda_values[(other_id, id_)]
        
        # Get classifier from the cost-sensitive oracle
        D_t = self.cost_sensitive_oracle(costs)
        
        # Compute alpha values
        alpha_t = {}
        for (i, j) in self.constraints:
            # If tau * w_ij/|A| - lambda_ij ≤ 0 then alpha_ij = 1, otherwise 0
            # comparing constraint's importance (lambda_ij) to the relaxation threshold
            weight = self.w_ij.get((i, j), 0)
            lambda_ij = lambda_values.get((i, j), 0)
            
            if tau_value * weight / self.A <= lambda_ij:
                alpha_t[(i, j)] = 1.0
            else:
                alpha_t[(i, j)] = 0.0
                
        return D_t, alpha_t
        

    def update_dual_variables(self, lambda_values, tau_value, D_t, alpha_t, gamma, t):
        """
        Improved update mechanism with better exploration and debugging
        """
        # probabilities of the different things being 1
        probs = self.compute_prediction_probs(D_t)
        
        # Track violations for reporting
        violation_count = 0
        max_gradient = 0.0

        # Calculate learning rates as specified in the paper
        mu_lambda = 1.0 / (self.C_lambda * np.sqrt(np.log(self.n) * t))  # Add t to denominator
        mu_tau = self.C_tau / np.sqrt(self.time_horizon * t)  # Add t to denominator
    
        
        # Update lambda values with more aggressive learning and exploration
        lambda_new = {}
        total_gradient_norm = 0.0
        
        # Step 1: Identify all constraint violations
        violations = []
        for (i, j) in self.constraints:
            diff = probs[self.id_to_index[i]] - probs[self.id_to_index[j]]
            gradient = diff - gamma
            curr_lambda = lambda_values.get((i, j), 0)
            beta = 0.9
            new_val = beta * curr_lambda + (1-beta) * max(0.0, min(self.C_lambda, curr_lambda + mu_lambda * gradient))
            lambda_new[(i, j)] = new_val
            if gradient > 0: # gradient = violation
                violations.append(((i, j), gradient, diff))
                violation_count += 1
                total_gradient_norm += gradient**2
        
        tau_gradient = sum(self.w_ij.get((i, j), 0) * alpha_t.get((i, j), 0) 
                       for (i, j) in self.constraints) / self.A - self.eta  # eta is often 0
        tau_new = max(0.0, min(self.C_tau, tau_value + mu_tau * tau_gradient))
    
        if violation_count > 0:
            print(f"Iteration {t}: Violations = {violation_count}")
       
        return lambda_new, tau_new
                
    def average_models(self, models, weights=None):
        """
        Average multiple models into a single model.
        
        For logistic regression, we average the coefficients and intercepts.
        
        Args:
            models: List of models to average.
            weights: Optional weights for averaging (default: equal weights).
            
        Returns:
            An averaged model.
        """
        if not models:
            return None
        
        if weights is None:
            weights = np.ones(len(models)) / len(models)
        
        # Create a new model
        avg_model = LogisticRegression()
        
        # Average coefficients
        avg_coef = np.zeros_like(models[0].coef_)
        avg_intercept = 0.0
        
        for i, model in enumerate(models):
            avg_coef += weights[i] * model.coef_
            avg_intercept += weights[i] * model.intercept_
        
        # Set the averaged coefficients
        avg_model.coef_ = avg_coef
        avg_model.intercept_ = np.array([avg_intercept])
        
        # Set necessary attributes
        avg_model.classes_ = np.array([0, 1])
        
        return avg_model
        
    def run(self, gamma_values=None):
        """
        Run the algorithm for multiple gamma values.
        
        Args:
            gamma_values: List of gamma values to test. If None, uses default values.
            
        Returns:
            Results for different gamma values.
        """
        if gamma_values is None:
            gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        results = {}
        
        for gamma in gamma_values:
            print(f"\nRunning algorithm with gamma = {gamma}")
            
            
            # Initialize storages for this gamma
            errors = []
            fairness_violations = []
            max_violations = []
            models = []
            
            # Initialize lambda and tau
            lambda_t = {pair: 0 for pair in self.constraints}
            tau_t = 0.0
            
            # Run the algorithm for T iterations
            stalled_iterations = 0
            prev_violation = float('inf')
            prev_error = float('inf')
            prev_lambda = {}
            for t in range(1, self.time_horizon + 1):
                # Best response of primal player
                print(f"Iteration {t}: lambda_t sum: {sum(lambda_t.values())}, tau_t: {tau_t}")
                
                D_t, alpha_t = self.best_response_primal(lambda_t, tau_t, gamma)
                
                # Compute metrics
                error = self.compute_error(D_t)
                probs = self.compute_prediction_probs(D_t)
                total_violation, max_violation = self.compute_fairness_violation(probs, alpha_t, gamma)
                
                # Track this model and its performance
                models.append(D_t)
                errors.append(error)
                fairness_violations.append(total_violation)
                max_violations.append(max_violation)
                
                # Check if we're making progress
                if t > 1:
                    error_change = abs(error - prev_error)
                    violation_change = abs(max_violation - prev_violation)
                    
                    if error_change < 1e-5 and violation_change < 1e-5:
                        stalled_iterations += 1
                    else:
                        stalled_iterations = 0
                
                prev_error = error
                prev_violation = max_violation
                
                # Update dual variables using River's no-regret optimizers
                lambda_t, tau_t = self.update_dual_variables(lambda_t, tau_t, D_t, alpha_t, gamma, t)
                
                # After updating dual variables
                print(f"Iteration {t}: Error: {error}, Max violation: {max_violation}")
                lambda_changes = [(k, lambda_t[k] - prev_lambda.get(k, 0)) 
                     for k in lambda_t 
                     if abs(lambda_t[k] - prev_lambda.get(k, 0)) > 0.01]
                print(f"Largest lambda changes: {sorted(lambda_changes, key=lambda x: abs(x[1]), reverse=True)[:5]}")
                
                # Store current lambda for next iteration comparison
                prev_lambda = dict(lambda_t)
                # Early stopping if we've made no progress
                if stalled_iterations > 50:
                    print(f"Early stopping after {t} iterations due to lack of progress")
                    break
            
            # Calculate averaged model
            final_model = self.average_models(models)
            final_error = self.compute_error(final_model)
            final_probs = self.compute_prediction_probs(final_model)
            
            # Create dummy alpha_t for the final model
            dummy_alpha = {pair: 0.0 for pair in self.constraints}
            
            final_violation, final_max_violation = self.compute_fairness_violation(
                final_probs, dummy_alpha, gamma
            )
            
            # Store results for this gamma
            results[gamma] = {
                'models': models,
                'errors': errors,
                'fairness_violations': fairness_violations,
                'max_violations': max_violations,
                'final_model': final_model,
                'final_error': final_error,
                'final_fairness_violation': final_violation,
                'final_max_violation': final_max_violation,
                'lambda_final': lambda_t,
                'tau_final': tau_t
            }
            
            print(f"Completed gamma = {gamma}: Final Error = {final_error:.4f}, Final Max Violation = {final_max_violation:.4f}")
        
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
        plt.title("Algorithm Trajectory for Various γ Values")
        plt.xlabel("Error")
        plt.ylabel("Maximum Fairness Violation")
        
        # Add horizontal lines at 0.1 intervals
        for y in np.arange(0, 1.1, 0.1):
            plt.axhline(y=y, color='r', linestyle='-', alpha=0.3)
        
        # Plot trajectory for each gamma
        for idx, gamma in enumerate(gamma_values):
            color = TOL_COLORS[idx % len(TOL_COLORS)]
            if gamma in results:
                errors = results[gamma]['errors']
                violations = results[gamma]['max_violations']
                plt.plot(errors, violations, label=f"γ = {gamma}", color=color)
                
                # Mark start and end points
                plt.scatter(errors[0], violations[0], color=color, s=50, 
                            marker='o', label=f"Start γ={gamma}" if gamma == gamma_values[0] else "")
                plt.scatter(errors[-1], violations[-1], color=color, s=50, 
                            marker='x', label=f"End γ={gamma}" if gamma == gamma_values[0] else "")
        
        plt.legend()
        plt.grid(True)
        plt.savefig("trajectory_plot.png")
        plt.show()
    
    def plot_pareto_curves(self, results):
        """
        Plot the Pareto curves for different subjects/gamma values.
        
        Args:
            results: Results from running the algorithm.
        """
        plt.figure(figsize=(12, 8))
        plt.title("Pareto Curve: Error vs. Fairness Violation")
        plt.xlabel("Error")
        plt.ylabel("Maximum Fairness Violation")
        
        # Extract final errors and violations for each gamma
        gammas = sorted(list(results.keys()))
        errors = [results[gamma]['final_error'] for gamma in gammas]
        violations = [results[gamma]['final_max_violation'] for gamma in gammas]
        
        # Plot the Pareto curve with points
        plt.plot(errors, violations, 'o-', markersize=8)
        
        # Add gamma labels to points
        for i, gamma in enumerate(gammas):
            plt.annotate(f"γ={gamma}", 
                        (errors[i], violations[i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        plt.grid(True)
        plt.savefig("pareto_curve.png")
        plt.show()

        


# Usage example:
def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run fairness elicitation algorithm")
    parser.add_argument("--judge_id", nargs='+', help="Specific judge ID to use constraints from")
    parser.add_argument("--data_path", type=str, default="train200_subset.parquet", 
                        help="Path to training data")
    parser.add_argument("--constraints_path", type=str, 
                        default="multi_persona_data/final_train.json",
                        help="Path to constraint sets JSON")
    parser.add_argument("--gamma", type=float, default=0.15, help="Gamma value for fairness violation")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations to run")
    
    args = parser.parse_args()
    
    # Initialize the algorithm
    algorithm = FairnessElicitationAlgorithm(
        data_path=args.data_path,
        constraint_sets_path=args.constraints_path,
        time_horizon=args.iterations,
        C_lambda=10.0, 
        C_tau=1.0
    )
    
    # If judge_id provided, use only that judge's constraints
    if args.judge_id:
        algorithm.load_constraint_sets(judge_id=args.judge_id)
    
    # Run the algorithm with specified gamma value
    gamma_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    # ---- BEFORE TRAINING ----
    base_model = LogisticRegression()
    base_model.fit(algorithm.X, algorithm.y)
    base_probs = base_model.predict_proba(algorithm.X)[:, 1]
    results = algorithm.run(gamma_values)
    
    # Plot the results
    algorithm.plot_trajectory(results)
    algorithm.plot_pareto_curves(results)
    # Compute final probabilities for a selected gamma, e.g., gamma = 0.15
    selected_gamma = 0.2
    final_model = results[selected_gamma]['final_model']
    final_probs = algorithm.compute_prediction_probs(final_model)

    # ---- KL Divergence ----
    # Bin probabilities into histograms to approximate PDFs
    bins = np.linspace(0, 1, 21)
    base_hist, _ = np.histogram(base_probs, bins=bins, density=True)
    final_hist, _ = np.histogram(final_probs, bins=bins, density=True)

    # Add small constant to avoid division by zero
    eps = 1e-10
    kl_div = entropy(base_hist + eps, final_hist + eps)

    print(f"\nKL Divergence (Before vs. After Training, γ = {selected_gamma}): {kl_div:.4f}\n")

    # ---- PLOT OVERALL DISTRIBUTIONS ----
    plt.figure(figsize=(10, 6))
    plt.hist(base_probs, bins=20, alpha=0.6, label="Before Training", color="gray", edgecolor="black")
    plt.hist(final_probs, bins=20, alpha=0.6, label=f"After Training (γ = {selected_gamma})", color="skyblue", edgecolor="black")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title("Predicted Probability Distributions (Overall)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("overall_prob_distributions.png")
    plt.show()

   

    
    # Save results to file
    import pickle
    with open(f"results_gamma_{args.gamma}.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to results_gamma_{args.gamma}.pkl")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy  # For KL divergence

def main1():
    # Initialize and load data
    algorithm = YourAlgorithm()  # Replace with actual class
    algorithm.load_data()

    gamma_values = [0.0, 0.1, 0.2, 0.5, 1.0]

    # ---- BEFORE TRAINING ----
    uniform_costs = np.ones((len(algorithm.y), 2))
    base_model = algorithm.cost_sensitive_oracle(uniform_costs)
    base_probs = algorithm.compute_prediction_probs(base_model)  # shape: (n_samples,)

    # ---- AFTER TRAINING ----
    results = algorithm.run(gamma_values)
    selected_gamma = gamma_values[2]
    final_model = results[selected_gamma]['final_model']
    final_probs = algorithm.compute_prediction_probs(final_model)

    # ---- KL Divergence ----
    # Bin probabilities into histograms to approximate PDFs
    bins = np.linspace(0, 1, 21)
    base_hist, _ = np.histogram(base_probs, bins=bins, density=True)
    final_hist, _ = np.histogram(final_probs, bins=bins, density=True)

    # Add small constant to avoid division by zero
    eps = 1e-10
    kl_div = entropy(base_hist + eps, final_hist + eps)

    print(f"\nKL Divergence (Before vs. After Training, γ = {selected_gamma}): {kl_div:.4f}\n")

    # ---- PLOT OVERALL DISTRIBUTIONS ----
    plt.figure(figsize=(10, 6))
    plt.hist(base_probs, bins=20, alpha=0.6, label="Before Training", color="gray", edgecolor="black")
    plt.hist(final_probs, bins=20, alpha=0.6, label=f"After Training (γ = {selected_gamma})", color="skyblue", edgecolor="black")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title("Predicted Probability Distributions (Overall)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("overall_prob_distributions.png")
    plt.show()

    # ---- GROUP-WISE PLOTS ----
    group_labels = algorithm.group_labels  # shape: (n_samples,)
    unique_groups = np.unique(group_labels)

    num_groups = len(unique_groups)
    fig, axs = plt.subplots(1, num_groups, figsize=(6 * num_groups, 5), sharey=True)

    for i, group in enumerate(unique_groups):
        idx = group_labels == group
        base_g = base_probs[idx]
        final_g = final_probs[idx]

        axs[i].hist(base_g, bins=20, alpha=0.6, label="Before", color="gray", edgecolor="black")
        axs[i].hist(final_g, bins=20, alpha=0.6, label="After", color="skyblue", edgecolor="black")
        axs[i].set_title(f"Group: {group}")
        axs[i].set_xlabel("Predicted Probability")
        axs[i].set_ylabel("Density")
        axs[i].legend()
        axs[i].grid(True)

    plt.suptitle("Predicted Probability Distributions by Group")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("groupwise_prob_distributions.png")
    plt.show()

if __name__ == "__main__":
    main()
