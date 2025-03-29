import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import os
import argparse
import pickle
import matplotlib.pyplot as plt
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

        # Initialize parameters
        self.n = len(self.X)  # Number of samples
        self.d = self.X.shape[1]  # Number of features

    def load_data(self):
        """Load and preprocess the data"""
        # Load data 
        df = pd.read_parquet(self.data_path)

        # Extract target
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
                categories = encoder.categories_[i][1:]  # Skip first category (dropped)
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

        print(f"Loaded data with {self.X.shape[0]} samples and {self.X.shape[1]} features")
        print(f"Label array shape: {self.y.shape}")

    def load_constraint_sets(self, judge_id):
        """
        Load fairness constraint sets for a specific judge
        
        Args:
            judge_id: ID of the judge to use constraints from
        """
        with open(self.constraint_sets_path, 'r') as f:
            constraint_data = json.load(f)
        
        # Check if judge exists
        if str(judge_id) not in constraint_data:
            raise ValueError(f"Judge ID {judge_id} not found in constraint data")
            
        # Initialize constraint sets
        self.constraints = set()
        self.w_ij = {}  # Weight for each constraint
            
        focused_constraints = constraint_data[str(judge_id)]
        for constraint in focused_constraints:
            print(constraint)
            pair = tuple(constraint['pair'])  # Extract the pair and convert to tuple
            self.constraints.add(pair)
            self.w_ij[pair] = constraint['weight']
        
        pairs_per_judge = 50  # Assuming 50 pairs were presented
        self.A = pairs_per_judge  # A is the total number of pairs presented
        
        print(f"Loaded {len(self.constraints)} constraints from judge {judge_id}")
        print(f"Judge selected {len(focused_constraints)} pairs out of {pairs_per_judge} presented")

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
        target_values = (cost_difference < 0).astype(int)

        # Use a standard classifier with sample weights
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
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

    def compute_fairness_violation(self, probs, alpha_ij, gamma):
        """Compute fairness violations across all constraints"""
        total_violation = 0.0
        individual_violations = {}
        max_violation = 0.0
        violation_count = 0

        for (i, j) in self.constraints:
            try:
                # Calculate violation for this pair
                diff = probs[i] - probs[j]
                violation = max(0, diff - gamma - alpha_ij.get((i, j), 0))
                
                if violation > 0:
                    violation_count += 1
                    individual_violations[(i, j)] = (diff, violation)
                    max_violation = max(max_violation, violation)

                # Store the raw violation for each constraint
                weight = self.w_ij.get((i, j), 0)
                weighted_violation = weight * violation
                total_violation += weighted_violation
                
            except Exception as e:
                print(f"Error processing constraint ({i}, {j}): {e}")
                continue

        # Debug information
        print(f"Violations exceeding γ={gamma}: {violation_count}/{len(self.constraints)}")

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
            weight = self.w_ij.get((i, j), 0)
            lambda_ij = lambda_values.get((i, j), 0)

            if tau_value * weight / self.A <= lambda_ij:
                alpha_t[(i, j)] = 1.0
            else:
                alpha_t[(i, j)] = 0.0

        return D_t, alpha_t

    def update_dual_variables(self, lambda_values, tau_value, D_t, alpha_t, gamma, t):
        """
        Update dual variables using exponentiated gradient descent for lambda 
        and online gradient descent for tau, as specified in the paper.
        """
        # Get prediction probabilities
        probs = self.compute_prediction_probs(D_t)
        
        # Track metrics for reporting
        violation_count = 0
        
        # Calculate learning rates as specified in the paper
        mu_lambda = 1.0 / (self.C_lambda * np.sqrt(np.log(self.n)))
        mu_tau = 1.0 / self.C_tau
        
        # Step 1: Update theta values (log space representation for lambda)
        theta = {pair: np.log(lambda_values.get(pair, 1e-10) + 1e-10) for pair in self.constraints}
        
        # Calculate gradients for all constraints
        for (i, j) in self.constraints:
            # Calculate gradient: E[h(x_i) - h(x_j)] - alpha_ij - gamma
            diff = probs[i] - probs[j]
            grad = diff - alpha_t.get((i, j), 0) - gamma
            
            # Update theta values
            theta[(i, j)] = theta.get((i, j), 0) + mu_lambda * grad
            
            if grad > 0:
                violation_count += 1
        
        # Step 2: Calculate the normalization factor
        Z = 1.0 + sum(np.exp(theta_val) for theta_val in theta.values())
        
        # Step 3: Convert back to lambda values using the formula from the paper
        lambda_new = {pair: self.C_lambda * np.exp(theta_val) / Z for pair, theta_val in theta.items()}
        
        # Step 4: Update tau using projected online gradient descent
        tau_gradient = (1.0 / self.A) * sum(self.w_ij.get((i, j), 0) * alpha_t.get((i, j), 0) 
                        for (i, j) in self.constraints) - self.eta
        
        tau_new = max(0.0, min(self.C_tau, tau_value + mu_tau * tau_gradient))
        
        if violation_count > 0:
            print(f"Iteration {t}: Violations = {violation_count}/{len(self.constraints)}")
        
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

    def run(self, gamma):
        """
        Run the algorithm for a specific gamma value.
        
        Args:
            gamma: The gamma value to use for fairness constraints.
            
        Returns:
            Results dictionary containing models, metrics, and final values.
        """
        print(f"\nRunning algorithm with gamma = {gamma}")

        # Initialize storages
        errors = []
        fairness_violations = []
        max_violations = []
        models = []

        # Initialize lambda and tau
        lambda_t = {pair: 0.0 for pair in self.constraints}
        tau_t = 0.0

        # Run the algorithm for T iterations
        stalled_iterations = 0
        prev_violation = float('inf')
        prev_error = float('inf')
        
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

            # Update dual variables using exponentiated gradient descent
            lambda_t, tau_t = self.update_dual_variables(lambda_t, tau_t, D_t, alpha_t, gamma, t)

            # Print progress
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

        # Store results
        results = {
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
        
    def plot_trajectory(self, results, judge_id, gamma):
        """
        Plot the trajectory of the algorithm.
        
        Args:
            results: Results from running the algorithm.
            judge_id: ID of the judge used for constraints.
            gamma: Gamma value used in the run.
        """
        plt.figure(figsize=(12, 8))
        plt.title(f"Algorithm Trajectory for Judge {judge_id} (γ = {gamma})")
        plt.xlabel("Error")
        plt.ylabel("Maximum Fairness Violation")

        # Add horizontal lines at 0.1 intervals
        for y in np.arange(0, 1.1, 0.1):
            plt.axhline(y=y, color='r', linestyle='-', alpha=0.3)

        # Plot trajectory
        errors = results['errors']
        violations = results['max_violations']
        plt.plot(errors, violations, label=f"γ = {gamma}")

        # Mark start and end points
        plt.scatter(errors[0], violations[0], color='green', s=50, marker='o', label="Start")
        plt.scatter(errors[-1], violations[-1], color='red', s=50, marker='x', label="End")

        plt.legend()
        plt.grid(True)
        plt.savefig(f"trajectory_judge_{judge_id}_gamma_{gamma}.png")
        plt.show()
        
        # Plot error and violation over iterations
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("Error over Iterations")
        plt.plot(range(1, len(errors) + 1), errors)
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.title("Max Fairness Violation over Iterations")
        plt.plot(range(1, len(violations) + 1), violations)
        plt.xlabel("Iteration")
        plt.ylabel("Max Violation")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"convergence_judge_{judge_id}_gamma_{gamma}.png")
        plt.show()


# Main function to run the algorithm
def main():
    parser = argparse.ArgumentParser(description="Run fairness elicitation algorithm for a single judge")
    parser.add_argument("--judge_id", type=str, required=True, help="ID of the judge to use constraints from")
    parser.add_argument("--data_path", type=str, default="data/processed/compas_train.parquet", 
                        help="Path to training data")
    parser.add_argument("--constraints_path", type=str, 
                        default="constraint_sets/lenient/binary_personas/constraint_sets.json",
                        help="Path to constraint sets JSON")
    parser.add_argument("--gamma", type=float, default=0.3, help="Gamma value for fairness violation")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations to run")
    
    args = parser.parse_args()
    
    # Get judge ID and gamma
    judge_id = args.judge_id
    gamma = args.gamma
    
    print(f"Running algorithm for judge {judge_id} with gamma = {gamma}")
    
    # Initialize the algorithm
    algorithm = FairnessElicitationAlgorithm(
        data_path=args.data_path,
        constraint_sets_path=args.constraints_path,
        time_horizon=args.iterations,
        C_lambda=10.0,
        C_tau=1.0
    )
    
    # Load constraints for the specific judge
    algorithm.load_constraint_sets(judge_id=judge_id)
    
    # Run the algorithm
    results = algorithm.run(gamma)
    
    # Plot the results
    algorithm.plot_trajectory(results, judge_id, gamma)
    
    # Save results to file
    output_file = f"results_judge_{judge_id}_gamma_{gamma}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
