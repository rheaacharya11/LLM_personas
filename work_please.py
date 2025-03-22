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
        self.avg_lambda = {}
        
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
        categorical_columns = [col for col in self.categorical_features if col in df_processed.columns]
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
        
        import random
        judge_ids = list(constraint_data.keys())
        selected_judges = random.sample(judge_ids, min(100, len(judge_ids)))

        # Filter constraint_data to only include selected judges
        constraint_data = {judge: constraint_data[judge] for judge in selected_judges}
        num_judges = len(selected_judges)
        
        # Each judge was presented 50 pairs but only selected a few
        pairs_per_judge = 50  # Number of pairs presented to each judge
        num_judges = 1000  # Override to match your scale
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
        total_violation = 0.0
        individual_violations = {}
        
        for (i, j) in self.constraints:
            try:
                # Calculate violation for this pair
                diff = probs[i] - probs[j]
                violation = max(0, diff - gamma - alpha_ij.get((i, j), 0))
                
                # Always store the raw violation for each constraint
                weight = self.w_ij.get((i, j), 0)
                weighted_violation = weight * violation
                total_violation += weighted_violation
                
                # Store even small violations for debugging
                if diff > 0:
                    individual_violations[(i, j)] = (diff, violation, weighted_violation)
            except Exception as e:
                print(f"Error processing constraint ({i}, {j}): {e}")
                continue
        
        # Debug information about violations
        if individual_violations:
            # Get the max raw violation 
            max_violation_pair = max(individual_violations.items(), 
                                    key=lambda x: x[1][1])
            max_violation = max_violation_pair[1][1]
            
            # print("Violation Diagnostics:")
            top_violations = sorted(individual_violations.items(), 
                                key=lambda x: x[1][1], reverse=True)[:5]
            # print("Top 5 Violations:")
            for (i, j), (diff, violation, weighted_violation) in top_violations:
                feature_diff = np.abs(self.X[i] - self.X[j])
                # print(f"  Pair ({i},{j}): "
                    # f"Prob Diff = {diff:.4f}, "
                    # f"Violation = {violation:.4f}, "
                    # f"Max Feature Diff = {feature_diff.max():.2f}")
        else:
            max_violation = 0.0
        
        return total_violation / self.A, max_violation
    
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
            lambda_ij = lambda_t.get((i, j), 0)
            
            if tau_t * weight / self.A <= lambda_ij:
                alpha_t[(i, j)] = 1.0
            else:
                alpha_t[(i, j)] = 0.0
                
        return D_t, alpha_t

    def update_dual_variables(self, lambda_t, tau_t, D_t, alpha_t, gamma, t):
        """
        Update the dual variables lambda and tau.
        
        Args:
            lambda_t: Current lambda values.
            tau_t: Current tau value.
            D_t: Current model.
            alpha_t: Current alpha values.
            gamma: Fairness violation buffer.
            t: Current iteration.
            
        Returns:
            lambda_new: Updated lambda values.
            tau_new: Updated tau value.
        """
        # Get prediction probabilities
        probs = self.compute_prediction_probs(D_t)
        
        # Use a much smaller step size to slow convergence
        mu_lambda = 0.001 / (self.C_lambda * np.sqrt(np.log(self.n) * t))
        
        # Simply update lambda directly
        lambda_new = {}
        
        # Track violations for debugging
        violation_count = 0
        max_gradient = 0.0
        
        for (i, j) in self.constraints:
            # Calculate gradient for this pair
            gradient = probs[i] - probs[j] - alpha_t.get((i, j), 0) - gamma
            step_scale = min(1.0, abs(gradient))
            # Count positive gradients (violations)
            if gradient > 0:
                violation_count += 1
                max_gradient = max(max_gradient, gradient)
            
            # Simple additive update with clipping
            if gradient > 0:
                # Increase lambda when constraint is violated
                lambda_new[(i, j)] = min(
                    self.C_lambda,
                    lambda_t.get((i, j), 0) + mu_lambda * gradient * step_scale
                )
            else:
                # Decrease lambda when constraint is satisfied
                lambda_new[(i, j)] = max(
                    0.0,
                    lambda_t.get((i, j), 0) + mu_lambda * gradient
                )
        
        # Update tau using online gradient descent
        mu_tau = self.C_tau / (1 + np.sqrt(t))
        
        # Compute gradient for tau (sum of w_ij * alpha_ij / |A| - eta)
        tau_gradient = 0.0
        for (i, j) in self.constraints:
            weight = self.w_ij.get((i, j), 0)
            alpha_ij = alpha_t.get((i, j), 0)
            tau_gradient += weight * alpha_ij / self.A
        
        # Update tau with projection (eta = 0 for simplicity)
        tau_new = max(0.0, min(self.C_tau, tau_t + mu_tau * tau_gradient))
        
        for pair in self.constraints:
            if pair not in self.avg_lambda:
                self.avg_lambda[pair] = lambda_new.get(pair, 0)
            else:
                self.avg_lambda[pair] = (t * self.avg_lambda[pair] + lambda_new.get(pair, 0)) / (t + 1)

        # Debug information
        if t % 1 == 0 and violation_count > 0:
            print(f"  Found {violation_count} constraint violations, max gradient: {max_gradient:.6f}")
        
        # Every 100 iterations, use the averaged values to stabilize
        if t % 100 == 0 and t > 0:
            print("  Using averaged lambda values to stabilize")
            lambda_new = self.avg_lambda.copy()
        
        # Add extensive logging in update_dual_variables
        # print("Constraint Diagnostics:")
        for (i, j) in self.constraints:
            feature_diff = np.abs(self.X[i] - self.X[j])
            # print(f"Pair ({i},{j}): Max Feature Diff = {feature_diff.max()}, Mean Diff = {feature_diff.mean()}")
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
            gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        results = {}
        
        for gamma in gamma_values:
            print(f"Running algorithm with gamma = {gamma}")
            
            # Reset algorithm state
            self.theta = {}
            self.tau = 0.0
            
            # Initialize storages for this gamma
            errors = []
            fairness_violations = []
            max_violations = []
            models = []
            
            # Initialize lambda and tau
            # Start with small random values to break symmetry
            lambda_t = {pair: self.w_ij.get(pair, 0) * 0.1 for pair in self.constraints}            
            tau_t = 0.01
            
            # Run the algorithm for T iterations
            stalled_iterations = 0
            prev_violation = float('inf')
            prev_error = float('inf')
            
            for t in range(1, self.time_horizon + 1):
                if t % 50 == 0:
                    print(f"Starting iteration {t}")
                
                try:
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
                            
                        # If we've stalled for too many iterations, try to shake things up
                        if stalled_iterations > 20 and t % 50 == 0:
                            print(f"  Algorithm stalled for {stalled_iterations} iterations, adding noise")
                            # Add noise to lambda values
                            for pair in self.constraints:
                                lambda_t[pair] = max(0, lambda_t[pair] + np.random.normal(0, 0.01))
                            stalled_iterations = 0
                    
                    prev_error = error
                    prev_violation = max_violation
                    
                    # Update dual variables
                    lambda_t, tau_t = self.update_dual_variables(lambda_t, tau_t, D_t, alpha_t, gamma, t)
                    
                    if t % 50 == 0:
                        # Find the constraint with the largest violation
                        largest_violation = 0
                        largest_pair = None
                        
                        for (i, j) in self.constraints:
                            violation = max(0, probs[i] - probs[j] - gamma)
                            if violation > largest_violation:
                                largest_violation = violation
                                largest_pair = (i, j)
                        
                        if largest_pair:
                            lambda_val = lambda_t.get(largest_pair, 0)
                            print(f"  Iteration {t}: Error = {error:.4f}, Max Violation = {max_violation:.4f}")
                            print(f"  Largest violation at {largest_pair}: {largest_violation:.6f}, lambda = {lambda_val:.6f}")
                        else:
                            print(f"  Iteration {t}: Error = {error:.4f}, Max Violation = {max_violation:.4f}")
                    
                    # Early stopping if we've made no progress
                    if stalled_iterations > 100:
                        print(f"Early stopping after {t} iterations due to lack of progress")
                        break
                        
                except Exception as e:
                    print(f"Error in iteration {t}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # Calculate averaged model
            final_model = self.average_models(models)
            final_error = self.compute_error(final_model)
            final_probs = self.compute_prediction_probs(final_model)
            
            # Create dummy alpha_t for the final model (we don't need it for violation calculation)
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
    algorithm = FairnessElicitationAlgorithm(
        data_path="data/processed/compas_train.parquet",
        constraint_sets_path="constraint_sets/binary_personas/constraint_sets.json",
        time_horizon=500,  # Reduced iterations 
        C_lambda=1.0,      # Reduced from 5.0
        C_tau=1.0          # Reduced from 5.0
    )
    
    # Run the algorithm for different gamma values
    gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = algorithm.run(gamma_values)
    
    # Plot the results
    algorithm.plot_trajectory(results)
    algorithm.plot_pareto_curves(results)

if __name__ == "__main__":
    main()