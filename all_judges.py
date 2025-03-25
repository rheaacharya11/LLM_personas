import numpy as np
import pandas as pd
import random as random
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
                 C_tau: float = 100.0):
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
        
        # Add diagnostic printing at the beginning
        print("Diagnostic: Checking for violations...")
        
        # Sample a few constraints to examine in detail
        import random
        sample_constraints = random.sample(list(self.constraints), min(5, len(self.constraints)))
        for i, j in sample_constraints:
            diff = abs(probs[i] - probs[j])
            alpha_val = alpha_ij.get((i, j), 0)
            threshold = gamma
            is_violation = diff > threshold
            print(f"Sample constraint ({i},{j}): probs={probs[i]:.4f},{probs[j]:.4f}, diff={diff:.4f}, threshold={threshold:.4f}, violation={is_violation}")
        
        # Continue with the existing code
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
            violation = max(0, abs_diff - gamma)
            
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
        print(f"Tau value: {tau_value}, Gamma: {gamma}")

        sample_constraints = random.sample(list(self.constraints), min(5, len(self.constraints)))

        for (i, j) in sample_constraints:

            weight = self.w_ij.get((i, j), 0)

            lambda_ij = lambda_values.get((i, j), 0)

            condition = 1 * weight / self.A

            print(f"Constraint ({i},{j}): lambda={lambda_ij:.4f}, tau*w/|A|={condition:.4f}, alpha={1.0 if condition >= lambda_ij else 0.0}")
        # Compute alpha values
        # In best_response_primal:

        # 1. First, create alpha_t with all zeros (opposite of your current approach)
        alpha_t = {pair: 0.0 for pair in self.constraints}

        # 2. Only after that, print the alpha distribution
        zeros = sum(v == 0 for v in alpha_t.values())
        ones = sum(v == 1 for v in alpha_t.values())
        print(f"INITIAL alpha distribution: {zeros} zeros, {ones} ones out of {len(alpha_t)}")

        # 3. NO OTHER CODE that modifies alpha_t should happen after this

        # 4. Add some final verification right before returning:
        final_zeros = sum(v == 0 for v in alpha_t.values())
        final_ones = sum(v == 1 for v in alpha_t.values())
        print(f"FINAL alpha distribution: {final_zeros} zeros, {final_ones} ones out of {len(alpha_t)}")

        # 5. Return the values
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
        
        # Inside update_dual_variables
        # Inside update_dual_variables
        # In update_dual_variables
        for (i, j) in self.constraints:
            diff = abs(probs[i] - probs[j])
            gradient = diff - gamma
            
            if gradient > 0:  # There's a violation
                # MUCH stronger update - exponentially increasing with violation size
                curr_lambda = lambda_values.get((i, j), 0)
                # Increase lambda proportional to violation size
                new_lambda = curr_lambda * np.exp(gradient * 5.0)  # Much stronger update
                lambda_new[(i, j)] = min(self.C_lambda, new_lambda)
                
                # Print some updates
                if i % 1000 == 0 and j % 1000 == 0:
                    print(f"Lambda update ({i},{j}): {curr_lambda:.4f} → {lambda_new[(i, j)]:.4f}, violation={gradient:.4f}")
            else:
                # Keep non-violated constraints the same
                lambda_new[(i, j)] = lambda_values.get((i, j), 0)
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
            tau_t = 10
            
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
        """Create an initial model with varied predictions"""
        # Start with a simple model
        model = LogisticRegression(C=1.0, class_weight='balanced')
        model.fit(self.X, self.y)
        
        # Check initial predictions
        initial_probs = model.predict_proba(self.X)[:, 1]
        print(f"Initial model: min={initial_probs.min():.4f}, max={initial_probs.max():.4f}, mean={initial_probs.mean():.4f}")
        
        # If predictions are too uniform, completely randomize the model
        if initial_probs.min() > 0.8 or initial_probs.max() < 0.2 or np.std(initial_probs) < 0.1:
            print("Initial model predictions too uniform, adding randomness")
            
            # Create a random model
            model = LogisticRegression()
            model.classes_ = np.array([0, 1])
            
            # Set random coefficients to ensure varied predictions
            model.coef_ = np.random.normal(0, 1.0, (1, self.X.shape[1]))
            model.intercept_ = np.array([0.0])  # Start with neutral bias
            
            # Verify randomized predictions
            random_probs = model.predict_proba(self.X)[:, 1]
            print(f"Randomized model: min={random_probs.min():.4f}, max={random_probs.max():.4f}, mean={random_probs.mean():.4f}")
            
            # If still too uniform, try again with more extreme values
            if random_probs.min() > 0.8 or random_probs.max() < 0.2 or np.std(random_probs) < 0.1:
                print("Still too uniform, trying more extreme randomization")
                model.coef_ = np.random.normal(0, 2.0, (1, self.X.shape[1]))
        
        # Explicitly check for variations in predictions across constraint pairs
        constraint_diffs = []
        for i, j in list(self.constraints)[:100]:  # Check a sample of constraints
            pred_i = model.predict_proba(self.X[i:i+1])[:, 1][0]
            pred_j = model.predict_proba(self.X[j:j+1])[:, 1][0]
            diff = abs(pred_i - pred_j)
            constraint_diffs.append(diff)
        
        print(f"Constraint pair differences: min={min(constraint_diffs):.4f}, max={max(constraint_diffs):.4f}, mean={np.mean(constraint_diffs):.4f}")
        
        # If we still don't have differences, force them by directly modifying predictions
        if max(constraint_diffs) < 0.1:
            print("Forcing prediction differences for constraint pairs")
            # Select highly weighted features
            important_features = np.argsort(np.abs(model.coef_[0]))[-5:]
            
            # Amplify these features to create larger differences
            for feature in important_features:
                model.coef_[0, feature] *= 3.0
        
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

    def run_with_all_judges(self, gamma_values=None, eta_values=None):
        """
        Run the algorithm with all judges and explore different eta values.
        
        Args:
            gamma_values: List of gamma values to test
            eta_values: List of eta values to test
        """
        if gamma_values is None:
            gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        if eta_values is None:
            eta_values = [0.2, 0.3]
        
        # Store results in a nested dictionary
        all_results = {}
        
        # Count how many individuals are covered by constraints
        constrained_individuals = set()
        for (i, j) in self.constraints:
            constrained_individuals.add(i)
            constrained_individuals.add(j)
        
        print(f"Total individuals in dataset: {len(self.X)}")
        print(f"Individuals covered by constraints: {len(constrained_individuals)}")
        print(f"Coverage percentage: {len(constrained_individuals)/len(self.X)*100:.2f}%")
        
        for gamma in gamma_values:
            gamma_results = {}
            
            for eta in eta_values:
                print(f"\nRunning algorithm with gamma = {gamma}, eta = {eta}")
                print("=" * 50)
                
                # Set the eta value for this run
                self.eta = eta
                
                # Initialize storages
                errors = []
                fairness_violations = []
                max_violations = []
                models = []
                
                # Initialize lambda and tau
                lambda_t = {pair: 0.01 for pair in self.constraints}
                tau_t = 100
                
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
                    
                    # Update dual variables
                    lambda_t, tau_t = self.update_dual_variables(lambda_t, tau_t, D_t, alpha_t, gamma, t)
                    
                    print(f"Iteration {t}: Error: {error:.4f}, Max violation: {max_violation:.4f}")
                    
                    # Optional early stopping condition
                    if t > 5 and abs(errors[-1] - errors[-2]) < 1e-5 and abs(max_violations[-1] - max_violations[-2]) < 1e-5:
                        print(f"Early stopping at iteration {t}")
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
                
                # Store results for this eta
                gamma_results[eta] = {
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
                
                print(f"Completed gamma = {gamma}, eta = {eta}: Final Error = {final_error:.4f}, Final Max Violation = {final_max_violation:.4f}")
            
            all_results[gamma] = gamma_results
        
        return all_results
    def plot_eta_comparison(self, results):
        """
        Plot how different eta values affect the fairness-accuracy tradeoff.
        
        Args:
            results: Nested dictionary with gamma as first key and eta as second
        """
        plt.figure(figsize=(15, 10))
        
        # Create one subplot for each gamma value
        gamma_values = sorted(list(results.keys()))
        n_gamma = len(gamma_values)
        
        # Determine grid layout
        n_cols = min(3, n_gamma)
        n_rows = (n_gamma + n_cols - 1) // n_cols
        
        for i, gamma in enumerate(gamma_values):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Extract etas for this gamma
            eta_values = sorted(list(results[gamma].keys()))
            errors = [results[gamma][eta]['final_error'] for eta in eta_values]
            violations = [results[gamma][eta]['final_max_violation'] for eta in eta_values]
            
            # Plot points
            plt.scatter(errors, violations, s=100)
            
            # Connect points to show the trend
            plt.plot(errors, violations, 'o-')
            
            # Add eta labels to each point
            for j, eta in enumerate(eta_values):
                plt.annotate(f"η={eta}", 
                            (errors[j], violations[j]),
                            textcoords="offset points", 
                            xytext=(5, 5), 
                            ha='left')
            
            plt.title(f"γ = {gamma}")
            plt.xlabel("Error")
            plt.ylabel("Max Fairness Violation")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("eta_comparison.png")
        plt.show()
        
        # Also create a 3D visualization
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare data for 3D plot
        all_gammas = []
        all_etas = []
        all_errors = []
        all_violations = []
        
        for gamma in gamma_values:
            for eta in sorted(list(results[gamma].keys())):
                all_gammas.append(gamma)
                all_etas.append(eta)
                all_errors.append(results[gamma][eta]['final_error'])
                all_violations.append(results[gamma][eta]['final_max_violation'])
        
        # Create scatter plot
        scatter = ax.scatter(all_gammas, all_etas, all_errors, c=all_violations, 
                            cmap='viridis', s=100, alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Max Fairness Violation')
        
        ax.set_xlabel('Gamma (γ)')
        ax.set_ylabel('Eta (η)')
        ax.set_zlabel('Error')
        
        plt.title('3D Visualization of Error, Fairness Violation, Gamma, and Eta')
        plt.savefig("3d_visualization.png")
        plt.show()
    # Usage example:
def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run fairness elicitation algorithm")
    parser.add_argument("--data_path", type=str, default="data/processed/compas_train.parquet", 
                        help="Path to training data")
    parser.add_argument("--constraints_path", type=str, 
                        default="constraint_sets/lenient/binary_personas/constraint_sets.json",
                        help="Path to constraint sets JSON")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations to run")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the algorithm (using all judges)
    algorithm = FairnessElicitationAlgorithm(
        data_path=args.data_path,
        constraint_sets_path=args.constraints_path,
        time_horizon=args.iterations,
        C_lambda=20.0, 
        C_tau=100
    )
    
    # Define gamma and eta values to test
    gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    eta_values = [0.01, 0.05, 0.1, 0.2]
    
    # Run with all judges
    results = algorithm.run_with_all_judges(gamma_values, eta_values)
    
    # Create visualizations
    algorithm.plot_eta_comparison(results)
    
    # Save results
    import pickle
    with open(os.path.join(args.output_dir, "all_judges_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {os.path.join(args.output_dir, 'all_judges_results.pkl')}")
if __name__ == "__main__":
    main()