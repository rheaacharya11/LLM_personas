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
        """Load fairness constraint sets with weighted average across all reviewers"""
        with open(self.constraint_sets_path, 'r') as f:
            constraint_data = json.load(f)
        
        # Track which pairs were presented to which judges
        self.constraints = set()
        self.w_ij = {}
        pair_reviewers = {}  # Maps pairs to the set of judges who reviewed them
        
        # Process each judge's constraints
        for judge_id, pairs in constraint_data.items():
            # Get all pairs this judge reviewed (both selected and not selected)
            reviewed_pairs = set()
            for constraint in pairs:
                pair = tuple(constraint['pair'])
                reviewed_pairs.add(pair)
                
                # Add to constraints set
                self.constraints.add(pair)
                
                # Add judge to reviewers for this pair
                if pair not in pair_reviewers:
                    pair_reviewers[pair] = set()
                pair_reviewers[pair].add(judge_id)
                
                # Add weight to the sum
                weight = constraint['weight']
                if pair not in self.w_ij:
                    self.w_ij[pair] = weight
                else:
                    self.w_ij[pair] += weight
        
        # Normalize weights by total number of reviewers for each pair
        for pair in self.constraints:
            num_reviewers = len(pair_reviewers[pair])
            self.w_ij[pair] /= num_reviewers
        
        # Print statistics
        total_pairs = len(self.constraints)
        print(f"Loaded {total_pairs} unique constraints")
        print(f"Average number of reviewers per pair: {sum(len(r) for r in pair_reviewers.values())/total_pairs:.2f}")
    
    def load_constraint_sets_for_subject(self, subject_id):
        """Load fairness constraint sets for a specific subject/judge with weights"""
        with open(self.constraint_sets_path, 'r') as f:
            constraint_data = json.load(f)
        
        # Ensure the subject exists in the data
        if str(subject_id) not in constraint_data:
            raise ValueError(f"Subject ID {subject_id} not found in constraint data")
        
        # Initialize constraint sets for this subject only
        self.constraints = set()
        self.w_ij = {}  # For a single subject, weights are based on bidirectionality
        
        # Get constraints for this specific subject
        pairs_data = constraint_data[str(subject_id)]
        
        # Each judge was presented 50 pairs
        pairs_per_judge = 50  # Number of pairs presented to each judge
        self.A = pairs_per_judge  # A is the total number of pairs presented
        
        # Process the subject's constraints
        for pair_data in pairs_data:
            # Extract pair and weight
            if isinstance(pair_data, dict):
                pair = tuple(pair_data["pair"])
                weight = pair_data["weight"]
            else:
                # Handle old format if needed
                pair = tuple(pair_data)
                weight = 1.0
                
            # Add to constraints set
            self.constraints.add(pair)
            self.w_ij[pair] = weight  # Use the bidirectional similarity weight
        
        print(f"Loaded {len(self.constraints)} unique constraints from subject {subject_id}")
        print(f"Subject selected {len(self.constraints)} pairs out of {pairs_per_judge} presented")
        bidir_count = sum(1 for w in self.w_ij.values() if w >= 0.9)
        print(f"  - Bidirectional (strong) constraints: {bidir_count}")
        print(f"  - Unidirectional (weak) constraints: {len(self.constraints) - bidir_count}")
        
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
                violation = max(0, abs(diff) - gamma - alpha_ij.get((i, j), 0))
                
                # Always store the raw violation for each constraint
                weight = self.w_ij.get((i, j), 0)
                weighted_violation = weight * violation
                total_violation += weighted_violation
                
                # Store even small violations for debugging
                if abs(diff) > 0:
                    individual_violations[(i, j)] = (diff, violation, weighted_violation)
            except Exception as e:
                print(f"Error processing constraint ({i}, {j}): {e}")
                continue
        
        # Debug information about violations
        # print(individual_violations)
        if individual_violations:
            # Get the max raw violation 
            max_violation_pair = max(individual_violations.items(), 
                                    key=lambda x: x[1][1])
            max_violation = max_violation_pair[1][1]
            # print(max_violation_pair)
            
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
        # print(max_violation)
        
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
        probs = self.compute_prediction_probs(D_t)
        # More aggressive, adaptive learning rate
        mu_lambda = 0.1 / (1 + np.sqrt(t))
        lambda_new = {}
        violation_count = 0
        max_gradient = 0.0
        
        for (i, j) in self.constraints:
            gradient = abs(probs[i] - probs[j]) - alpha_t.get((i, j), 0) - gamma
            
            if gradient > 1e-3:
                violation_count += 1
                max_gradient = max(max_gradient, gradient)
            
            # Scale update by constraint weight
            weight = self.w_ij.get((i, j), 0)
            
            # Add exploration noise
            noise = np.random.normal(0, 0.01) if t % 10 == 0 else 0
            
            # Apply weight to the gradient update
            update = mu_lambda * gradient * weight + noise
            
            lambda_new[(i, j)] = max(0, min(self.C_lambda, 
                                        lambda_t.get((i, j), 0) + update))
            
            # Periodic aggressive reset for stuck constraints
            if t % 50 == 0 and abs(gradient) > 0.5:
                lambda_new[(i, j)] = np.random.uniform(0, self.C_lambda)
        
        # Tau update with more exploration
        tau_gradient = sum(
            self.w_ij.get((i, j), 0) * alpha_t.get((i, j), 0) / self.A
            for (i, j) in self.constraints
        )
        
        tau_new = max(0.0, min(self.C_tau, tau_t + mu_lambda * tau_gradient))
        
        if violation_count > 0:
            print(f"Iteration {t}: Violations = {violation_count}, Max Gradient = {max_gradient:.6f}")
        
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
                            violation = max(0, abs(probs[i] - probs[j]) - gamma)
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

    def plot_individual_pareto_curve(self, results, subject_id):
        """Plot the Pareto curve for an individual subject"""
        plt.figure(figsize=(12, 8))
        plt.title(f"Pareto Curve for Subject {subject_id}: Error vs. Fairness Violation")
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
        plt.savefig(f"pareto_curve_subject_{subject_id}.png")
        plt.show()

    def plot_collective_tradeoff(self, results):
        """Plot the trade-off curve for collective constraints with varying eta"""
        plt.figure(figsize=(12, 8))
        plt.title(f"Collective Constraint Analysis: Error vs. Fairness Violation (γ={results[list(results.keys())[0]]['gamma']})")
        plt.xlabel("Error")
        plt.ylabel("Maximum Fairness Violation")
        
        # Extract final errors and violations for each eta
        etas = sorted(list(results.keys()))
        errors = [results[eta]['final_error'] for eta in etas]
        violations = [results[eta]['final_max_violation'] for eta in etas]
        
        # Plot the curve with points
        plt.plot(errors, violations, 'o-', markersize=8)
        
        # Add eta labels to points
        for i, eta in enumerate(etas):
            plt.annotate(f"η={eta}", 
                        (errors[i], violations[i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        plt.grid(True)
        plt.savefig("collective_tradeoff_curve.png")
        plt.show()

    def run_individual_subject_analysis(self, subject_id, gamma_values=None):
        """
        Run analysis for a single subject varying gamma values with eta=0
        
        Args:
            subject_id: ID of the specific subject to analyze
            gamma_values: List of gamma values to test. If None, uses default values.
        """
        if gamma_values is None:
            #gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            gamma_values = [0.3]
        
        # Load constraints for this subject
        self.load_constraint_sets_for_subject(subject_id)
        
        # Run the algorithm with fixed eta=0 (no constraint violations allowed)
        results = {}
        
        for gamma in gamma_values:
            print(f"Running individual subject analysis for subject {subject_id} with gamma = {gamma}")
            
            # For individual subject with small constraint set, increase C_lambda for γ=0
            if gamma == 0.0:
                original_C_lambda = self.C_lambda
                self.C_lambda = 20.0  # Use larger C_lambda for strict constraints
                print(f"Using increased C_lambda={self.C_lambda} for gamma=0")
                
                # Run with more iterations for γ=0 to ensure convergence
                original_time_horizon = self.time_horizon
                self.time_horizon = self.time_horizon * 2
            
            # Reset algorithm state
            self.theta = {}
            self.tau = 0.0
            
            # Initialize storages for this gamma
            errors = []
            fairness_violations = []
            max_violations = []
            models = []
            
            # Initialize lambda and tau with small values
            lambda_t = {pair: self.w_ij.get(pair, 0) * 0.1 for pair in self.constraints}            
            tau_t = 0.01  # For individual subject analysis with eta=0, tau has little effect
            
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
                    # Inside your loop or after optimization
                    for pair in self.constraints:
                        i, j = pair
                        diff = abs(probs[i] - probs[j])
                        raw_violation = diff
                        effective_violation = max(0, diff - gamma)
                        # if diff > 0:
                            # print(f"Pair {pair}: Prob diff = {diff:.6f}, Violation beyond γ = {effective_violation:.6f}")
                    # Print lambda values for top violated constraints
                    top_violations = sorted([(pair, probs[pair[0]] - probs[pair[1]]) 
                                            for pair in self.constraints], 
                                            key=lambda x: x[1], reverse=True)[:5]
                    print("Lambda values for top violated constraints:")
                    for pair, diff in top_violations:
                        print(f"Pair {pair}: λ = {lambda_t.get(pair, 0):.4f}, Diff = {diff:.6f}")
                                        
                    prob_diffs = [abs(probs[i] - probs[j]) for i, j in self.constraints]
                    print(f"Mean prob diff: {np.mean(prob_diffs):.6f}")
                    print(f"Max prob diff: {np.max(prob_diffs):.6f}")

                    count_actual_violations = sum(1 for i, j in self.constraints if abs(probs[i] - probs[j]) > 0)
                    count_effective_violations = sum(1 for i, j in self.constraints if abs(probs[i] - probs[j]) > gamma)
                    print(f"Actual violations: {count_actual_violations}/{len(self.constraints)}")
                    print(f"Violations exceeding γ={gamma}: {count_effective_violations}/{len(self.constraints)}")
                                        
                    if t % 50 == 0:
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
            
            # Reset parameters if they were modified
            if gamma == 0.0:
                self.C_lambda = original_C_lambda
                self.time_horizon = original_time_horizon
        
        return results
# Example usage:
def run_experiments():
    algorithm = FairnessElicitationAlgorithm(
        data_path="data/processed/compas_train.parquet",
        constraint_sets_path="constraint_sets/lenient/binary_personas/constraint_sets.json",
        time_horizon=500,
        C_lambda=10.0,
        C_tau=10.0
    )
    
    # 1. Run individual subject analysis
    subject_id = "121"  # Choose a specific subject ID
    individual_results = algorithm.run_individual_subject_analysis(
        subject_id=subject_id,
        # gamma_values = [0.3]
        gamma_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    algorithm.plot_individual_pareto_curve(individual_results, subject_id)
    
    # 2. Run collective analysis
    # collective_results = algorithm.run_collective_analysis(
        # eta_values=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        # fixed_gamma=0.3  # Based on your earlier chart, 0.3 looked promising
    # )
    # algorithm.plot_collective_tradeoff(collective_results)

if __name__ == "__main__":
    run_experiments()

# Usage example:
def main1():
    algorithm = FairnessElicitationAlgorithm(
        data_path="data/processed/compas_train.parquet",
        constraint_sets_path="constraint_sets/binary_personas/constraint_sets.json",
        time_horizon=500,  # Reduced iterations 
        C_lambda=1.0,      # Reduced from 5.0
        C_tau=1.0          # Reduced from 5.0
    )
    
    # Run the algorithm for different gamma values
    gamma_values = [0.3]
    #gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = algorithm.run(gamma_values)
    
    # Plot the results
    algorithm.plot_trajectory(results)
    algorithm.plot_pareto_curves(results)

if __name__ == "__main1__":
    main()