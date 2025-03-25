import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import os
from typing import Dict, List, Tuple, Set, Any, Union

# Import River library components
from river import optim
from river import utils

# exponentiated gradient descent
class Hedge:
    def __init__(self, lr=0.1, weight_bound=(0, float('inf'))):
        self.learning_rate = lr  # Fix variable name
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
        
        # Identify categorical columns
        categorical_columns = [col for col in self.categorical_features if col in df_processed.columns]
        print(f"Categorical columns to encode: {categorical_columns}")
        
        # Drop target column from features
        if self.target_column in df_processed.columns:
            df_processed = df_processed.drop(self.target_column, axis=1)
        print(df_processed[categorical_columns].head())
        # One-hot encode categorical features
        if categorical_columns:
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            categorical_data = df_processed[categorical_columns].fillna('Unknown')
            encoded_data = encoder.fit_transform(categorical_data)
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
        
        print(f"Loaded data with {self.X.shape[0]} samples and {self.X.shape[1]} features")
        print(f"Label array shape: {self.y.shape}")

    def load_constraint_sets(self, judge_id=None):
        """
        Load fairness constraint sets from JSON file
        
        Args:
            judge_id: Optional specific judge ID to use. If None, uses multiple judges.
        """
        with open(self.constraint_sets_path, 'r') as f:
            constraint_data = json.load(f)
        for key, value in list(constraint_data.items())[:5]:  # Adjust the number 5 as needed
            print(key, value)

        

        # Check the type of data and count the entries
        if isinstance(constraint_data, list):
            # If it's a list, count the number of elements in the list
            print("Number of entries:", len(constraint_data))
        elif isinstance(constraint_data, dict):
            # If it's a dictionary, count the number of key-value pairs
            print("Number of entries:", len(constraint_data))
        else:
            print("Unexpected data type.")
        # Initialize constraint sets
        self.constraints = set()
        self.w_ij = {}  # Weight for each constraint
        
        # If specific judge requested
        if judge_id is not None:
            # Check if judge exists
            if str(judge_id) not in constraint_data:
                raise ValueError(f"Judge ID {judge_id} not found in constraint data")
            focused_constraints = constraint_data[judge_id]
            for constraint in focused_constraints:  # directly iterate over the list
                pair = tuple(constraint['pair'])  # Extract the pair and convert to tuple
                self.constraints.add(pair)
                self.w_ij[pair] = 2 * constraint['weight']

            pairs_per_judge = 50  # Assuming 50 pairs were presented
            self.A = pairs_per_judge  # A is the total number of pairs presented
                
            print(f"Loaded {len(self.constraints)} constraints from judge {judge_id}")
            print(f"Judge selected {len(focused_constraints)} pairs out of {pairs_per_judge} presented")
        
        else:
            # Multiple judges case (original code)
            import random
            judge_ids = list(constraint_data.keys())
            # selected_judges = random.sample(judge_ids, min(100, len(judge_ids)))
    
            # Filter constraint_data to only include selected judges
            constraint_data = {judge: constraint_data[judge] for judge in judge_ids}
            num_judges = len(judge_ids)
            mean_judges_per_pair = 10
            
            # Each judge was presented 50 pairs but only selected a few
            pairs_per_judge = 50  # Number of pairs presented to each judge
            total_pairs_presented = pairs_per_judge * num_judges
            self.A =  5000  # A is the total number of pairs presented
            
            # Process each judge's constraints
            for j_id, constraints in constraint_data.items():
                for constraint in constraints:
                    pair = tuple(constraint['pair'])  # Extract the pair and convert to tuple
                    pair_swapped = tuple(reversed(pair)) # include the opposite direciton too because equality
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
            C=0.5,
            solver='liblinear',  # Faster for smaller datasets
            max_iter=1000,
            class_weight=None,  # We handle through sample_weights
            random_state=42
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
        """Calculate fairness violations based on similarity constraints"""
        max_violation = 0.0
        total_violation = 0.0
        violation_count = 0
        processed_pairs = set()
        
        for (i, j) in self.constraints:
            # Skip if we've processed this pair (in either direction)
            pair_key = tuple(sorted([i, j]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            # Absolute difference for similarity constraints
            abs_diff = abs(probs[i] - probs[j])
            
            # Get the alpha value for this constraint
            alpha_value = alpha_ij.get((i, j), 0)
            
            # Calculate violation
            violation = max(0, abs_diff - gamma - alpha_value)
            
            if violation > 0:
                violation_count += 1
                max_violation = max(max_violation, violation)
                
                # Weight by importance
                weight = self.w_ij.get((i, j), 0)
                total_violation += weight * violation
        
        if len(processed_pairs) > 0:
            print(f"Found {violation_count} violations out of {len(processed_pairs)} unique constraint pairs")
        
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
                if (i, j) in lambda_values:
                    costs[i, 1] += lambda_values[(i, j)]
                if (j, i) in lambda_values:
                    costs[i, 1] -= lambda_values[(j, i)]
        
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
        """Update dual variables using pure exponentiated gradient (Hedge algorithm)"""
        # Get classification probabilities
        probs = self.compute_prediction_probs(D_t)
        
        # Calculate learning rates based on paper specifications
        mu_lambda = 1.0 / (self.C_lambda * np.sqrt(np.log(self.n)))
        mu_tau = 1.0 / (self.C_tau * np.sqrt(self.time_horizon))
        
        # Update lambda values using exponentiated gradient
        lambda_new = {}
        violation_count = 0
        
        for (i, j) in self.constraints:
            # Calculate gradient = violation
            diff = probs[i] - probs[j]
            gradient = diff - gamma - alpha_t.get((i, j), 0)
            curr_lambda = lambda_values.get((i, j), 0)
            
            # Use exponentiated gradient update
            if gradient > 0:  # There's a violation
                # Multiplicative update
                new_val = curr_lambda * np.exp(mu_lambda * gradient)
                violation_count += 1
            else:
                # Gradual decay for non-violations
                new_val = curr_lambda * 0.99
            
            # Clip to bounds
            new_val = min(self.C_lambda, new_val)
            lambda_new[(i, j)] = new_val
        
        # Update tau using gradient descent
        tau_gradient = sum(self.w_ij.get((i, j), 0) * alpha_t.get((i, j), 0) 
                        for (i, j) in self.constraints) / self.A - self.eta
        tau_new = max(0.0, min(self.C_tau, tau_value + mu_tau * tau_gradient))
        
        print(f"Iteration {t}: Found {violation_count} violations, lambda sum: {sum(lambda_new.values()):.4f}, tau: {tau_new:.4f}")
        
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
        """Run the algorithm for multiple gamma values with proper initialization"""
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
            lambda_t = {pair: 0.01 for pair in self.constraints}  # Small initial value
            tau_t = 0.0
            
            # Start with a strongly biased model
            initial_model = self.initialize_vanilla_model()
            initial_probs = self.compute_prediction_probs(initial_model)
            initial_alpha = {pair: 0.0 for pair in self.constraints}
            
            # Calculate initial metrics
            initial_error = self.compute_error(initial_model)
            initial_violation, initial_max = self.compute_fairness_violation(
                initial_probs, initial_alpha, gamma
            )
            
            # Store initial values
            models.append(initial_model)
            errors.append(initial_error)
            fairness_violations.append(initial_violation)
            max_violations.append(initial_max)
            
            print(f"Initial model - Error: {initial_error:.4f}, Max violation: {initial_max:.4f}")
            
            # Run the algorithm for T iterations
            stalled_iterations = 0
            prev_violation = initial_max
            prev_error = initial_error
            
            for t in range(1, self.time_horizon + 1):
                # Best response of primal player
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
                
                # Update dual variables
                lambda_t, tau_t = self.update_dual_variables(lambda_t, tau_t, D_t, alpha_t, gamma, t)
                
                print(f"Iteration {t}: Error: {error:.4f}, Max violation: {max_violation:.4f}")
                
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
        for gamma in gamma_values:
            if gamma in results:
                errors = results[gamma]['errors']
                violations = results[gamma]['max_violations']
                plt.plot(errors, violations, label=f"γ = {gamma}")
                
                # Mark start and end points
                plt.scatter(errors[0], violations[0], color='green', s=50, 
                            marker='o', label=f"Start γ={gamma}" if gamma == gamma_values[0] else "")
                plt.scatter(errors[-1], violations[-1], color='red', s=50, 
                            marker='x', label=f"End γ={gamma}" if gamma == gamma_values[0] else "")
        
        plt.legend()
        plt.grid(True)
        plt.savefig("trajectory_plot.png")
        plt.show()
    
    def initialize_vanilla_model(self):
        """Create an initial model with good accuracy (~66%) that still violates fairness constraints"""
        
        # First, train a basic model to get good accuracy
        base_model = LogisticRegression(C=1.0, class_weight='balanced')
        base_model.fit(self.X, self.y)
        
        # Check its accuracy
        base_preds = base_model.predict(self.X)
        base_accuracy = sum(base_preds == self.y) / len(self.y)
        print(f"Base model accuracy: {base_accuracy:.4f}")
        
        # Create a new model starting with the accurate base model's coefficients
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        model.coef_ = base_model.coef_.copy()
        model.intercept_ = base_model.intercept_.copy()
        
        # Identify constraint pairs that should be treated similarly
        similar_pairs = list(self.constraints)
        
        if similar_pairs:
            # Find features that appear in many of the similar pairs
            feature_counts = np.zeros(self.X.shape[1])
            
            for (i, j) in similar_pairs:
                # Find features where these individuals differ
                for k in range(self.X.shape[1]):
                    if abs(self.X[i, k] - self.X[j, k]) > 0.01:
                        feature_counts[k] += 1
            
            # Get top differentiating features
            top_features = np.argsort(-feature_counts)[:5]  # Top 5 differentiating features
            print(f"Top differentiating features: {top_features}")
            
            # Amplify these features to create violations without destroying accuracy
            for k in top_features:
                # Amplify existing coefficient to increase disparity
                model.coef_[0, k] *= 5.0
            
            # Check if we're creating violations
            probs = model.predict_proba(self.X)[:, 1]
            violations = 0
            max_diff = 0
            
            for (i, j) in similar_pairs:
                diff = abs(probs[i] - probs[j])
                max_diff = max(max_diff, diff)
                if diff > 0.2:  # Check if there's a meaningful difference
                    violations += 1
            
            print(f"Created model with {violations}/{len(similar_pairs)} violations, max diff: {max_diff:.4f}")
            
            # If still not enough violations, add more bias
            if violations < len(similar_pairs) * 0.5 or max_diff < 0.3:
                print("Adding additional bias to create more violations")
                # Add direct bias to specific sensitive attributes
                # This assumes first few features might be sensitive attributes
                for k in range(min(3, self.X.shape[1])):
                    model.coef_[0, k] += 2.0
        
        # Check final model accuracy
        final_preds = model.predict(self.X)
        final_accuracy = sum(final_preds == self.y) / len(self.y)
        print(f"Final model accuracy: {final_accuracy:.4f}")
        
        return model
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
    parser.add_argument("--judge_id", type=str, help="Specific judge ID to use constraints from")
    parser.add_argument("--data_path", type=str, default="data/processed/compas_train.parquet", 
                        help="Path to training data")
    parser.add_argument("--constraints_path", type=str, 
                        default="constraint_sets/lenient/binary_personas/constraint_sets.json",
                        help="Path to constraint sets JSON")
    parser.add_argument("--gamma", type=float, default=0.3, help="Gamma value for fairness violation")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations to run")
    
    args = parser.parse_args()
    
    # Initialize the algorithm
    algorithm = FairnessElicitationAlgorithm(
        data_path=args.data_path,
        constraint_sets_path=args.constraints_path,
        time_horizon=args.iterations,
        C_lambda=20.0, 
        C_tau=1.0
    )
    
    # If judge_id provided, use only that judge's constraints
    if args.judge_id:
        algorithm.load_constraint_sets(judge_id=args.judge_id)
    
    # Run the algorithm with specified gamma value
    gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = algorithm.run(gamma_values)
    
    # Plot the results
    algorithm.plot_trajectory(results)
    algorithm.plot_pareto_curves(results)
    
    # Save results to file
    import pickle
    with open(f"results_gamma_{args.gamma}.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to results_gamma_{args.gamma}.pkl")

if __name__ == "__main__":
    main()