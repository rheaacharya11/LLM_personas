import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Set, Any, Union
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
import os
import pickle
import time

class FairnessElicitationAlgorithm:
    """
    Implementation of the algorithmic framework for fairness elicitation from the paper
    "An Algorithmic Framework for Fairness Elicitation"
    """
    
    def __init__(
        self, 
        data_path: str, 
        constraint_sets_path: str,
        categorical_features: List[str] = None,
        target_column: str = 'two_year_recid',
        time_horizon: int = 1000,
        C_lambda: float = 10.0,
        C_tau: float = 10.0,
        eta: float = 0.1,
        gamma: float = 0.1,
        num_judges: int = 1000,
        pairs_per_judge: int = 50,
        total_pairs: int = 5000,
        verbose: bool = True
    ):
        """
        Initialize the algorithm with data and parameters.
        
        Args:
            data_path: Path to the dataset
            constraint_sets_path: Path to the constraint sets JSON
            categorical_features: List of categorical feature names
            target_column: Name of the target column
            time_horizon: Number of iterations to run
            C_lambda: Bound on lambda values
            C_tau: Bound on tau value
            eta: Fairness violation budget
            gamma: Allowed margin for fairness constraint violations
            num_judges: Number of judges providing constraints
            pairs_per_judge: Number of pairs presented to each judge
            total_pairs: Total number of pairs in the dataset
            verbose: Whether to print verbose output
        """
        self.data_path = data_path
        self.constraint_sets_path = constraint_sets_path
        self.categorical_features = categorical_features or ['sex', 'race', 'c_charge_degree']
        self.target_column = target_column
        self.time_horizon = time_horizon
        self.C_lambda = C_lambda
        self.C_tau = C_tau
        self.eta = eta
        self.gamma = gamma
        self.num_judges = num_judges
        self.pairs_per_judge = pairs_per_judge
        self.total_pairs = total_pairs
        self.verbose = verbose
        
        # Load and preprocess data
        self.load_data()
        
        # Load constraint sets
        self.load_constraint_sets()
        
        # Initialize parameters
        self.n = len(self.X)  # Number of samples
        self.d = self.X.shape[1]  # Number of features
        
        # Print initialization summary
        if self.verbose:
            print(f"Initialized Fairness Elicitation Algorithm:")
            print(f"- Dataset: {self.n} samples, {self.d} features")
            print(f"- Constraints: {len(self.constraints)} unique constraint pairs")
            print(f"- Parameters: C_lambda={self.C_lambda}, C_tau={self.C_tau}, eta={self.eta}, gamma={self.gamma}")
            print(f"- Judges: {self.num_judges} judges with {self.pairs_per_judge} pairs each")
    
    def load_data(self):
        """Load and preprocess the dataset"""
        # Load data 
        df = pd.read_parquet(self.data_path)
        
        # Extract target
        self.y = df[self.target_column].values
        
        # Save original data
        self.original_df = df.copy()
        
        # Create a copy for processing
        df_processed = df.copy()
        
        # Identify categorical columns
        categorical_columns = [col for col in self.categorical_features if col in df_processed.columns]
        
        if self.verbose:
            print(f"Processing categorical features: {categorical_columns}")
        
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
                categories = encoder.categories_[i][1:]  # Skip first category (reference level)
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
        
        # Fill NaN values with 0's
        df_processed = df_processed.fillna(0)
        
        # Convert to numpy array
        self.X = df_processed.values
        
        if self.verbose:
            print(f"Loaded data with {self.X.shape[0]} samples and {self.X.shape[1]} features")

    def load_constraint_sets(self):
        """Load fairness constraint sets from JSON file"""
        with open(self.constraint_sets_path, 'r') as f:
            constraint_data = json.load(f)
        
        # Initialize constraint sets
        self.constraints = set()
        self.w_ij = {}  # Weight for each constraint
        
        # Get the list of judge IDs
        judge_ids = list(constraint_data.keys())
        
        if len(judge_ids) < self.num_judges:
            if self.verbose:
                print(f"Warning: Only {len(judge_ids)} judges found in data, less than requested {self.num_judges}")
            self.num_judges = len(judge_ids)
        
        # Calculate the total number of pairs presented
        self.A = self.num_judges * self.pairs_per_judge
        
        # Process each judge's constraints
        for j_id in judge_ids[:self.num_judges]:  # Limit to requested number of judges
            judge_constraints = constraint_data[j_id]
            
            for constraint in judge_constraints:
                pair = tuple(constraint['pair'])  # Extract the pair as tuple
                self.constraints.add(pair)
                
                # Calculate weight - the proportion of judges who selected this constraint
                if pair not in self.w_ij:
                    self.w_ij[pair] = constraint['weight'] / self.num_judges
                else:
                    self.w_ij[pair] += constraint['weight'] / self.num_judges
        
        # Calculate statistics for reporting
        unique_constraints = len(self.constraints)
        avg_constraints_per_judge = sum(len(constraint_data[j_id]) for j_id in judge_ids[:self.num_judges]) / self.num_judges
        
        if self.verbose:
            print(f"Loaded {unique_constraints} unique constraints from {self.num_judges} judges")
            print(f"Judges selected {avg_constraints_per_judge:.2f} pairs on average out of {self.pairs_per_judge} presented")
            
            # Report on the distribution of weights
            weights = list(self.w_ij.values())
            print(f"Constraint weights: min={min(weights):.4f}, max={max(weights):.4f}, mean={np.mean(weights):.4f}")
    
    def cost_sensitive_oracle(self, costs):
        """
        Cost-sensitive classification oracle as described in the paper.
        
        Args:
            costs: Array of shape (n, 2) with cost[i,0] and cost[i,1] representing
                  the cost of classifying example i as 0 or 1, respectively.
                  
        Returns:
            Trained classifier that minimizes the weighted cost
        """
        # Create sample weights based on the absolute cost difference
        cost_difference = costs[:, 0] - costs[:, 1]
        sample_weights = np.abs(cost_difference)
        
        # Create target values based on the sign of cost difference
        # If c₀ > c₁, then we want to predict 1, otherwise 0
        target_values = (cost_difference > 0).astype(int)
        
        # Use logistic regression as the base classifier
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',
            max_iter=1000,
            random_state=42
        )
        
        # Train the model with sample weights
        model.fit(self.X, target_values, sample_weight=sample_weights)
        
        return model
    
    def compute_prediction_probs(self, model):
        """Compute prediction probabilities for positive class"""
        return model.predict_proba(self.X)[:, 1]
    
    def compute_error(self, model):
        """Compute classification error of the model"""
        y_pred = model.predict(self.X)
        return 1 - accuracy_score(self.y, y_pred)
    
    def compute_fairness_violation(self, probs, gamma):
        """
        Calculate fairness violations based on constraints
        
        Args:
            probs: Predicted probabilities for each instance
            gamma: Fairness violation buffer
            
        Returns:
            total_violation: Weighted average of fairness violations
            max_violation: Maximum fairness violation across all constraints
        """
        violations = {}
        weighted_violations = {}
        
        for (i, j) in self.constraints:
            # Calculate absolute difference in predictions
            diff = abs(probs[i] - probs[j])
            
            # Calculate violation (max of 0 and difference minus gamma)
            violation = max(0, diff - gamma)
            
            if violation > 0:
                violations[(i, j)] = violation
                weighted_violations[(i, j)] = self.w_ij.get((i, j), 0) * violation
        
        # Calculate total weighted violation and maximum violation
        total_violation = sum(weighted_violations.values()) / self.A if weighted_violations else 0
        max_violation = max(violations.values()) if violations else 0
        
        if self.verbose and len(violations) > 0:
            print(f"Found {len(violations)} violations out of {len(self.constraints)} constraints")
            print(f"Maximum violation: {max_violation:.4f}, Total weighted violation: {total_violation:.4f}")
        
        return total_violation, max_violation
    
    def best_response_primal(self, lambda_values, tau_value):
        """
        Compute the best response for the primal player (D_t, alpha_t)
        as described in Section 3.2 of the paper.
        
        Args:
            lambda_values: Current lambda values for all constraint pairs
            tau_value: Current tau value
            
        Returns:
            D_t: A classifier model
            alpha_t: Dictionary of alpha values for each constraint pair
        """
        # Initialize costs for cost-sensitive classification
        costs = np.zeros((self.n, 2))
        
        # Set costs based on true labels (classification error)
        for i in range(self.n):
            if self.y[i] == 0:
                costs[i, 0] = 0
                costs[i, 1] = 1/self.n
            else:
                costs[i, 0] = 1/self.n
                costs[i, 1] = 0
        
        # Add costs from lambda terms
        for (i, j) in self.constraints:
            lambda_ij = lambda_values.get((i, j), 0)
            if lambda_ij > 0:
                costs[i, 1] += lambda_ij  # Add to cost of classifying i as positive
                costs[j, 1] -= lambda_ij  # Subtract from cost of classifying j as positive
        
        # Get classifier from cost-sensitive oracle
        D_t = self.cost_sensitive_oracle(costs)
        
        # Compute alpha values based on tau and lambda
        alpha_t = {}
        for (i, j) in self.constraints:
            weight = self.w_ij.get((i, j), 0)
            lambda_ij = lambda_values.get((i, j), 0)
            
            # Set alpha to 1 if tau*w_ij/|A| ≥ lambda_ij, else 0
            alpha_t[(i, j)] = 1.0 if tau_value * weight / self.A >= lambda_ij else 0.0
        
        if self.verbose:
            zeros = sum(v == 0 for v in alpha_t.values())
            ones = sum(v == 1 for v in alpha_t.values())
            print(f"Alpha distribution: {zeros} zeros, {ones} ones out of {len(alpha_t)}")
        
        return D_t, alpha_t
    
    def update_dual_variables(self, lambda_values, tau_value, D_t, alpha_t, gamma, t):
        """
        Update dual variables (lambda, tau) using no-regret learning
        as described in Section 3.3 of the paper.
        
        Args:
            lambda_values: Current lambda values
            tau_value: Current tau value
            D_t: Current classifier model
            alpha_t: Current alpha values
            gamma: Fairness violation buffer
            t: Current iteration
            
        Returns:
            lambda_new: Updated lambda values
            tau_new: Updated tau value
        """
        # Get classification probabilities
        probs = self.compute_prediction_probs(D_t)
        
        # Calculate learning rates based on paper
        mu_lambda = 1.0 / (self.C_lambda * np.sqrt(np.log(self.n)))
        mu_tau = 1.0 / (self.C_tau * np.sqrt(t))
        
        # Update lambda values using exponentiated gradient
        lambda_new = {}
        for (i, j) in self.constraints:
            # Calculate absolute difference in predictions
            diff = abs(probs[i] - probs[j])
            
            # Calculate gradient for lambda update
            gradient = diff - gamma - alpha_t.get((i, j), 0)
            
            # Current lambda value
            curr_lambda = lambda_values.get((i, j), 0)
            
            # Update lambda using exponentiated gradient
            new_lambda = curr_lambda * np.exp(mu_lambda * gradient)
            
            # Bound lambda within [0, C_lambda]
            lambda_new[(i, j)] = min(self.C_lambda, new_lambda)
        
        # Calculate tau gradient
        tau_gradient = (
            sum(self.w_ij.get((i, j), 0) * alpha_t.get((i, j), 0) 
                for (i, j) in self.constraints) / self.A - self.eta
        )
        
        # Update tau using gradient descent
        tau_new = max(0.0, min(self.C_tau, tau_value + mu_tau * tau_gradient))
        
        if self.verbose:
            total_lambda = sum(lambda_new.values())
            print(f"Iteration {t}: lambda sum: {total_lambda:.4f}, tau: {tau_new:.4f}")
        
        return lambda_new, tau_new
    
    def average_models(self, models, weights=None):
        """
        Average multiple models into a single model.
        For logistic regression, we average the coefficients and intercepts.
        
        Args:
            models: List of models to average
            weights: Optional weights for averaging (default: equal weights)
            
        Returns:
            An averaged model
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
            avg_intercept += weights[i] * model.intercept_[0]
        
        # Set the averaged coefficients
        avg_model.coef_ = avg_coef
        avg_model.intercept_ = np.array([avg_intercept])
        
        # Set necessary attributes
        avg_model.classes_ = np.array([0, 1])
        
        return avg_model
    
    def initialize_model(self):
        """Initialize a model for learning"""
        # Use regularized logistic regression with balanced class weights
        model = LogisticRegression(
            C=1.0, 
            class_weight='balanced',
            solver='liblinear',
            max_iter=1000,
            random_state=42
        )
        
        # Fit to the data
        model.fit(self.X, self.y)
        
        return model
    
    def run(self):
        """
        Run the main algorithm as described in the paper.
        
        Returns:
            Dictionary of results including the final model and metrics
        """
        start_time = time.time()
        
        # Initialize storage for results
        errors = []
        fairness_violations = []
        max_violations = []
        models = []
        
        # Initialize lambda and tau
        lambda_t = {pair: 0.0 for pair in self.constraints}  # Start with zero
        tau_t = 0.0  # Start with zero
        
        # Get initial model
        initial_model = self.initialize_model()
        initial_probs = self.compute_prediction_probs(initial_model)
        
        # Calculate initial metrics
        initial_error = self.compute_error(initial_model)
        initial_violation, initial_max = self.compute_fairness_violation(
            initial_probs, self.gamma
        )
        
        # Store initial values
        models.append(initial_model)
        errors.append(initial_error)
        fairness_violations.append(initial_violation)
        max_violations.append(initial_max)
        
        if self.verbose:
            print(f"Initial model - Error: {initial_error:.4f}, Max violation: {initial_max:.4f}")
        
        # Run the algorithm for T iterations
        for t in range(1, self.time_horizon + 1):
            iteration_start = time.time()
            
            # Best response of primal player
            D_t, alpha_t = self.best_response_primal(lambda_t, tau_t)
            
            # Compute metrics
            probs = self.compute_prediction_probs(D_t)
            error = self.compute_error(D_t)
            total_violation, max_violation = self.compute_fairness_violation(probs, self.gamma)
            
            # Track this model and its performance
            models.append(D_t)
            errors.append(error)
            fairness_violations.append(total_violation)
            max_violations.append(max_violation)
            
            # Update dual variables
            lambda_t, tau_t = self.update_dual_variables(
                lambda_t, tau_t, D_t, alpha_t, self.gamma, t
            )
            
            iteration_time = time.time() - iteration_start
            
            if self.verbose:
                print(f"Iteration {t}: Error: {error:.4f}, Max violation: {max_violation:.4f}, Time: {iteration_time:.2f}s")
            
            # Early stopping criteria - convergence check
            if t > 5:
                error_change = abs(errors[-1] - errors[-2])
                violation_change = abs(max_violations[-1] - max_violations[-2])
                
                if error_change < 1e-5 and violation_change < 1e-5:
                    if self.verbose:
                        print(f"Early stopping at iteration {t} due to convergence")
                    break
        
        # Calculate averaged model (Equation 17 in the paper)
        final_model = self.average_models(models)
        final_probs = self.compute_prediction_probs(final_model)
        
        # Calculate final metrics
        final_error = self.compute_error(final_model)
        final_violation, final_max = self.compute_fairness_violation(final_probs, self.gamma)
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"Algorithm completed in {total_time:.2f}s")
            print(f"Final model - Error: {final_error:.4f}, Max violation: {final_max:.4f}")
        
        # Store and return results
        results = {
            'models': models,
            'errors': errors,
            'fairness_violations': fairness_violations,
            'max_violations': max_violations,
            'final_model': final_model,
            'final_error': final_error,
            'final_fairness_violation': final_violation,
            'final_max_violation': final_max,
            'lambda_final': lambda_t,
            'tau_final': tau_t,
            'runtime': total_time
        }
        
        return results
    
    def run_parameter_sweep(self, gamma_values=None, eta_values=None):
        """
        Run algorithm with different parameter values to explore tradeoffs.
        
        Args:
            gamma_values: List of gamma values to test
            eta_values: List of eta values to test
            
        Returns:
            Nested dictionary of results for each parameter combination
        """
        if gamma_values is None:
            gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        if eta_values is None:
            eta_values = [0.0, 0.05, 0.1, 0.2]
        
        # Store results in a nested dictionary
        all_results = {}
        
        # Count how many individuals are covered by constraints
        constrained_individuals = set()
        for (i, j) in self.constraints:
            constrained_individuals.add(i)
            constrained_individuals.add(j)
        
        if self.verbose:
            print(f"Total individuals in dataset: {self.n}")
            print(f"Individuals covered by constraints: {len(constrained_individuals)}")
            print(f"Coverage percentage: {len(constrained_individuals)/self.n*100:.2f}%")
            print("\nStarting parameter sweep:")
            print(f"- Gamma values: {gamma_values}")
            print(f"- Eta values: {eta_values}")
        
        for gamma in gamma_values:
            gamma_results = {}
            
            for eta in eta_values:
                if self.verbose:
                    print(f"\nRunning with gamma = {gamma}, eta = {eta}")
                    print("=" * 50)
                
                # Update parameters for this run
                self.gamma = gamma
                self.eta = eta
                
                # Run the algorithm
                results = self.run()
                
                # Store results for this parameter combination
                gamma_results[eta] = results
            
            all_results[gamma] = gamma_results
        
        return all_results
    
    def plot_pareto_curve(self, results):
        """
        Plot the Pareto curve for error vs. fairness violation.
        
        Args:
            results: Results from running the algorithm with different parameters
        """
        plt.figure(figsize=(10, 7))
        plt.title("Pareto Curve: Error vs. Fairness Violation")
        plt.xlabel("Error")
        plt.ylabel("Maximum Fairness Violation")
        
        # Extract results for each gamma (fixing eta at middle value)
        gamma_values = sorted(list(results.keys()))
        eta_values = sorted(list(results[gamma_values[0]].keys()))
        middle_eta = eta_values[len(eta_values)//2]
        
        errors = [results[gamma][middle_eta]['final_error'] for gamma in gamma_values]
        violations = [results[gamma][middle_eta]['final_max_violation'] for gamma in gamma_values]
        
        # Plot the Pareto curve
        plt.plot(errors, violations, 'o-', linewidth=2, markersize=8)
        
        # Add gamma labels
        for i, gamma in enumerate(gamma_values):
            plt.annotate(f"γ={gamma}", 
                        (errors[i], violations[i]),
                        textcoords="offset points", 
                        xytext=(5, 5), 
                        ha='left')
        
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()  # Return the figure for saving

    def plot_eta_effect(self, results, gamma=0.1):
        """
        Plot the effect of eta on error and fairness violation for a fixed gamma.
        
        Args:
            results: Results from parameter sweep
            gamma: The gamma value to plot
        """
        if gamma not in results:
            raise ValueError(f"No results found for gamma={gamma}")
        
        plt.figure(figsize=(10, 7))
        plt.title(f"Effect of η on Error and Fairness Violation (γ={gamma})")
        plt.xlabel("Error")
        plt.ylabel("Maximum Fairness Violation")
        
        # Extract results for different eta values
        eta_values = sorted(list(results[gamma].keys()))
        errors = [results[gamma][eta]['final_error'] for eta in eta_values]
        violations = [results[gamma][eta]['final_max_violation'] for eta in eta_values]
        
        # Plot the curve
        plt.plot(errors, violations, 'o-', linewidth=2, markersize=8)
        
        # Add eta labels
        for i, eta in enumerate(eta_values):
            plt.annotate(f"η={eta}", 
                        (errors[i], violations[i]),
                        textcoords="offset points", 
                        xytext=(5, 5), 
                        ha='left')
        
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_trajectories(self, results, gamma=0.1, eta=0.1):
        """
        Plot the trajectory of error and fairness violation over iterations.
        
        Args:
            results: Results from running the algorithm
            gamma: Gamma value to plot
            eta: Eta value to plot
        """
        if gamma not in results or eta not in results[gamma]:
            raise ValueError(f"No results found for gamma={gamma}, eta={eta}")
        
        result = results[gamma][eta]
        errors = result['errors']
        violations = result['max_violations']
        
        plt.figure(figsize=(12, 5))
        
        # Error trajectory
        plt.subplot(1, 2, 1)
        plt.title(f"Error Trajectory (γ={gamma}, η={eta})")
        plt.plot(range(len(errors)), errors, 'b-')
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.grid(True)
        
        # Violation trajectory
        plt.subplot(1, 2, 2)
        plt.title(f"Fairness Violation Trajectory (γ={gamma}, η={eta})")
        plt.plot(range(len(violations)), violations, 'r-')
        plt.xlabel("Iteration")
        plt.ylabel("Maximum Violation")
        plt.grid(True)
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def save_results(self, results, output_dir="results"):
        """
        Save results and plots to the specified directory.
        
        Args:
            results: Results from running the algorithm
            output_dir: Directory to save results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results dictionary
        with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
            pickle.dump(results, f)
        
        # Generate and save plots
        try:
            # Pareto curve
            fig = self.plot_pareto_curve(results)
            fig.savefig(os.path.join(output_dir, "pareto_curve.png"))
            plt.close(fig)
            
            # Eta effect for a middle gamma value
            gamma_values = sorted(list(results.keys()))
            middle_gamma = gamma_values[len(gamma_values)//2]
            fig = self.plot_eta_effect(results, gamma=middle_gamma)
            fig.savefig(os.path.join(output_dir, f"eta_effect_gamma_{middle_gamma}.png"))
            plt.close(fig)
            
            # Trajectories for middle gamma and eta
            eta_values = sorted(list(results[middle_gamma].keys()))
            middle_eta = eta_values[len(eta_values)//2]
            fig = self.plot_trajectories(results, gamma=middle_gamma, eta=middle_eta)
            fig.savefig(os.path.join(output_dir, f"trajectories_gamma_{middle_gamma}_eta_{middle_eta}.png"))
            plt.close(fig)
            
            if self.verbose:
                print(f"Results and plots saved to {output_dir}")
                
        except Exception as e:
            print(f"Error generating plots: {e}")

def main():
    """Main function to run the algorithm"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Fairness Elicitation Algorithm")
    parser.add_argument("--data_path", type=str, default="data/processed/compas_train.parquet",
                        help="Path to training data")
    parser.add_argument("--constraints_path", type=str, 
                        default="constraint_sets/lenient/binary_personas/constraint_sets.json",
                        help="Path to constraint sets JSON")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations to run")
    parser.add_argument("--num_judges", type=int, default=1000,
                        help="Number of judges to include")
    parser.add_argument("--pairs_per_judge", type=int, default=50,
                        help="Number of pairs presented to each judge")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Fairness violation buffer")
    parser.add_argument("--eta", type=float, default=0.1,
                        help="Fairness violation budget")
    parser.add_argument("--param_sweep", action="store_true",
                        help="Run parameter sweep over gamma and eta values")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the algorithm
    algorithm = FairnessElicitationAlgorithm(
        data_path=args.data_path,
        constraint_sets_path=args.constraints_path,
        time_horizon=args.iterations,
        gamma=args.gamma,
        eta=args.eta,
        num_judges=args.num_judges,
        pairs_per_judge=args.pairs_per_judge,
        total_pairs=5000,  # Assuming 5000 total pairs as mentioned
        verbose=args.verbose
    )
    
    if args.param_sweep:
        # Define parameter values to test
        gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        eta_values = [0.0, 0.05, 0.1, 0.2, 0.3]
        
        # Run parameter sweep
        print(f"Running parameter sweep with {len(gamma_values)} gamma values and {len(eta_values)} eta values")
        results = algorithm.run_parameter_sweep(gamma_values, eta_values)
    else:
        # Run with specified parameters
        print(f"Running algorithm with gamma={args.gamma}, eta={args.eta}")
        results = {args.gamma: {args.eta: algorithm.run()}}
    
    # Save results and generate plots
    algorithm.save_results(results, args.output_dir)
    
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main()