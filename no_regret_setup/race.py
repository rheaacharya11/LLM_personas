import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import pickle
import os
import random
import warnings
from typing import Dict, List, Tuple, Set, Any, Union
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Define different gamma values for training and testing
TRAIN_GAMMA = 0.3  # Higher gamma for training (stricter fairness constraints)
TEST_GAMMA = 0.2   # Lower gamma for testing (more relaxed evaluation)

# Plotting colors 
COLORS = {
    "African-American": "#332288",  # navy
    "Caucasian": "#CC6677",  # red-pink
    "Asian": "#44AA99",  # teal
}

# Target races to include in the study
TARGET_RACES = ["African-American", "Caucasian", "Asian"]
JUDGES_PER_RACE = 50  # Standardize to 50 judges per race

class Hedge:
    """
    Implementation of Hedge algorithm (exponentiated gradient descent)
    for no-regret learning on the dual variables.
    """
    def __init__(self, lr=0.1, weight_bound=(0, float('inf'))):
        self.learning_rate = lr
        self.min_bound, self.max_bound = weight_bound
    
    def step(self, weight, gradient):
        # Multiplicative update
        new_weight = weight * np.exp(self.learning_rate * gradient)
        # Apply bounds
        return min(max(new_weight, self.min_bound), self.max_bound)


class FairnessElicitationAlgorithm:
    """
    Implementation of the No-Regret Algorithm for Fairness Elicitation focused on
    specific race groups with standardized judge count per race
    """
    
    def __init__(self, 
                 data_path: str, 
                 constraint_sets_path: str,
                 persona_metadata_path: str,
                 categorical_features: List[str] = None,
                 target_column: str = 'two_year_recid',
                 time_horizon: int = 500,
                 C_lambda: float = 10.0,
                 C_tau: float = 10.0,
                 train_race: str = None,
                 output_dir: str = "./output"):
        """
        Initialize the algorithm with data and parameters.
        
        Args:
            data_path: Path to the training data parquet file
            constraint_sets_path: Path to the constraints JSON file
            persona_metadata_path: Path to judge metadata
            categorical_features: List of categorical feature names
            target_column: Name of the target column
            time_horizon: Number of iterations to run the algorithm
            C_lambda: Upper bound on lambda values
            C_tau: Upper bound on tau value
            train_race: Race to use for training constraints
            output_dir: Directory to save output files
        """
        self.data_path = data_path
        self.constraint_sets_path = constraint_sets_path
        self.persona_metadata_path = persona_metadata_path
        self.categorical_features = categorical_features or ['sex', 'race', 'c_charge_degree']
        self.target_column = target_column
        self.time_horizon = time_horizon
        self.C_lambda = C_lambda
        self.C_tau = C_tau
        self.train_race = train_race
        self.output_dir = output_dir
        self.eta = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load judge metadata
        with open(persona_metadata_path, 'r') as f:
            self.persona_metadata = json.load(f)
        
        # Load and preprocess data
        self.load_data()
        
        # Load constraint sets from specified race (if provided)
        self.load_constraint_sets()
        
        # Initialize parameters
        self.n = len(self.X)  # Number of samples
        self.d = self.X.shape[1]  # Number of features
        
    def load_data(self):
        """Load and preprocess the training data"""
        print(f"Loading data from {self.data_path}")
        
        # Load data 
        df = pd.read_parquet(self.data_path)
        
        # Extract target
        self.y = df[self.target_column].values
        
        # Save original indices
        self.original_df = df.copy()
        
        # Create a copy for processing
        df_processed = df.copy()
        self.id_col = 'id'  # ID column to match with constraints
        self.ids = df[self.id_col].tolist()
        self.id_to_index = {id_: idx for idx, id_ in enumerate(self.ids)}
        self.index_to_id = {idx: id_ for idx, id_ in enumerate(self.ids)}
        
        # Identify categorical columns
        categorical_columns = [col for col in self.categorical_features if col in df_processed.columns]
        print(f"Categorical columns to encode: {categorical_columns}")
        
        # Drop target column from features
        if self.target_column in df_processed.columns:
            df_processed = df_processed.drop(self.target_column, axis=1)
        
        # One-hot encode categorical features
        if categorical_columns:
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            categorical_data = df_processed[categorical_columns].fillna('Unknown')
            encoded_data = encoder.fit_transform(categorical_data)
            
            # Save encoder for test data
            encoder_path = os.path.join(self.output_dir, "onehot_encoder.pkl")
            with open(encoder_path, "wb") as f:
                pickle.dump(encoder, f)
            
            # Get feature names
            encoded_feature_names = []
            for i, feature in enumerate(categorical_columns):
                categories = encoder.categories_[i][1:]  # Drop the first category (reference)
                encoded_feature_names.extend([f"{feature}_{category}" for category in categories])
            
            # Save encoded feature names
            encoded_features_path = os.path.join(self.output_dir, "encoded_feature_names.pkl")
            with open(encoded_features_path, "wb") as f:
                pickle.dump(encoded_feature_names, f)
            
            print(f"Encoded {len(categorical_columns)} categorical features into {len(encoded_feature_names)} binary features")
            
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
        
        # Save final training column order
        self.train_columns = df_processed.columns.tolist()
        train_columns_path = os.path.join(self.output_dir, "train_feature_columns.pkl")
        with open(train_columns_path, "wb") as f:
            pickle.dump(self.train_columns, f)
        
        print(f"Processed data: {self.X.shape[0]} samples, {self.X.shape[1]} features")

    def load_constraint_sets(self):
        """
        Load fairness constraint sets from JSON file, standardizing to exactly
        JUDGES_PER_RACE judges per racial group
        """
        self.constraints = set()
        self.w_ij = {}
        self.judge_constraints = {}  # Track which constraints came from which judges
        
        print(f"Loading constraints from {self.constraint_sets_path}")
        with open(self.constraint_sets_path, 'r') as f:
            all_constraint_data = json.load(f)
        
        # Group judges by race
        judges_by_race = {}
        for judge_id, metadata in self.persona_metadata.items():
            race = metadata.get('race')
            if race in TARGET_RACES and judge_id in all_constraint_data:
                if race not in judges_by_race:
                    judges_by_race[race] = []
                judges_by_race[race].append(judge_id)
        
        # Print distribution of judges by race
        print("Judge distribution by race:")
        for race, judges in judges_by_race.items():
            print(f"  {race}: {len(judges)} judges")
        
        # Sample exactly JUDGES_PER_RACE judges from each race if possible
        standardized_judges = {}
        for race in TARGET_RACES:
            if race in judges_by_race:
                if len(judges_by_race[race]) >= JUDGES_PER_RACE:
                    standardized_judges[race] = random.sample(judges_by_race[race], JUDGES_PER_RACE)
                else:
                    print(f"Warning: Only {len(judges_by_race[race])} {race} judges available, using all of them")
                    standardized_judges[race] = judges_by_race[race]
        
        # If training on a specific race, use only those judges
        if self.train_race:
            if self.train_race not in standardized_judges:
                raise ValueError(f"No judges found for race '{self.train_race}'")
            
            judges_to_use = standardized_judges[self.train_race]
            print(f"Using {len(judges_to_use)} {self.train_race} judges for training")
        else:
            # Use all judges from all races
            judges_to_use = []
            for race, judges in standardized_judges.items():
                judges_to_use.extend(judges)
            print(f"Using {len(judges_to_use)} judges across all races")
        
        # Extract constraints from selected judges
        constraint_data = {j_id: all_constraint_data[j_id] for j_id in judges_to_use if j_id in all_constraint_data}
        
        # Assume each judge was presented with around 100 pairs
        pairs_per_judge = 100  
        self.A = pairs_per_judge * len(constraint_data)  # Total pairs presented
        
        # Process constraints from selected judges
        for j_id, constraints in constraint_data.items():
            # Track judge's constraints
            self.judge_constraints[j_id] = set()
            
            for constraint in constraints:
                pair = tuple(constraint['pair'])
                pair_swapped = tuple(reversed(pair))
                
                # Add to overall constraints
                self.constraints.update([pair, pair_swapped])
                
                # Add to judge's specific constraints
                self.judge_constraints[j_id].add(pair)
                
                # Add weights (normalized by judge count)
                weight = constraint['weight'] / len(constraint_data)
                for p in [pair, pair_swapped]:
                    if p not in self.w_ij:
                        self.w_ij[p] = weight
                    else:
                        self.w_ij[p] += weight
        
        avg_constraints_per_judge = np.mean([len(constraints) for constraints in constraint_data.values()])
        print(f"Loaded {len(self.constraints)} unique constraints from {len(constraint_data)} judges")
        print(f"Average constraints per judge: {avg_constraints_per_judge:.2f}")
    
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
        # Calculate the cost difference
        cost_difference = costs[:, 0] - costs[:, 1]
        
        # Create sample weights based on the absolute cost difference
        sample_weights = np.abs(cost_difference) 
        
        # Create target values based on the sign of cost difference
        target_values = (cost_difference > 0).astype(int)
        
        # Use a standard classifier with sample weights
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',
            max_iter=1000,
            class_weight=None,  # Handle via sample_weights
            random_state=42
        )
        
        # Train the model with sample weights
        model.fit(self.X, target_values, sample_weight=sample_weights)
        
        return model
    
    def compute_prediction_probs(self, model, X=None):
        """Compute prediction probabilities for each sample"""
        if X is None:
            X = self.X
        return model.predict_proba(X)[:, 1]
    
    def compute_error(self, model, X=None, y=None):
        """Compute classification error of the model"""
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        y_pred = model.predict(X)
        return 1 - accuracy_score(y, y_pred)
    
    def compute_fairness_violation(self, probs, alpha_ij, gamma):
        """
        Compute the fairness violation for the current model
        
        Args:
            probs: Array of prediction probabilities
            alpha_ij: Dictionary of alpha values for each constraint pair
            gamma: Fairness violation threshold
            
        Returns:
            total_violation: Average weighted fairness violation
            max_violation: Maximum fairness violation
            violation_by_judge: Dictionary mapping judge IDs to their fairness violations
        """
        total_violation = 0.0
        individual_violations = {}
        max_violation = 0.0
        processed_pairs = set()
        violation_count = 0
        
        # Track violations by judge
        violation_by_judge = {j_id: [] for j_id in self.judge_constraints}
        
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
                
                # Track which judges' constraints were violated
                for j_id, constraints in self.judge_constraints.items():
                    if (i, j) in constraints or (j, i) in constraints:
                        violation_by_judge[j_id].append(violation)
                    
            except KeyError:
                # Skip pairs not in the training set
                pass
        
        # Calculate average violation per judge
        judge_violations = {}
        for j_id, violations in violation_by_judge.items():
            if violations:
                judge_violations[j_id] = np.mean(violations)
            else:
                judge_violations[j_id] = 0.0
        
        if processed_pairs:
            print(f"Violations exceeding γ={gamma}: {violation_count}/{len(processed_pairs)} ({violation_count/len(processed_pairs)*100:.1f}%)")
            return total_violation / len(processed_pairs), max_violation, judge_violations
        return 0.0, 0.0, judge_violations
        
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
                costs[idx, 0] = 0
                costs[idx, 1] = 1/len(self.y)
            else:
                costs[idx, 0] = 1/len(self.y)
                costs[idx, 1] = 0
                
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
            weight = self.w_ij.get((i, j), 0)
            lambda_ij = lambda_values.get((i, j), 0)
            
            if tau_value * weight / self.A <= lambda_ij:
                alpha_t[(i, j)] = 1.0
            else:
                alpha_t[(i, j)] = 0.0
                
        return D_t, alpha_t
        
    def update_dual_variables(self, lambda_values, tau_value, D_t, alpha_t, gamma, t):
        """
        Update the dual variables using no-regret dynamics
        
        Args:
            lambda_values: Current lambda values
            tau_value: Current tau value
            D_t: Current model
            alpha_t: Current alpha values
            gamma: Fairness threshold
            t: Current iteration
            
        Returns:
            lambda_new: Updated lambda values
            tau_new: Updated tau value
        """
        # Get prediction probabilities
        probs = self.compute_prediction_probs(D_t)
        
        # Calculate learning rates based on iteration
        mu_lambda = 1.0 / (self.C_lambda * np.sqrt(np.log(self.n) * t))
        mu_tau = self.C_tau / np.sqrt(self.time_horizon * t)
        
        # Update lambda values
        lambda_new = {}
        violation_count = 0
        
        for (i, j) in self.constraints:
            try:
                diff = probs[self.id_to_index[i]] - probs[self.id_to_index[j]]
                gradient = diff - gamma
                curr_lambda = lambda_values.get((i, j), 0)
                
                # Momentum update
                beta = 0.9
                new_val = beta * curr_lambda + (1-beta) * max(0.0, min(self.C_lambda, curr_lambda + mu_lambda * gradient))
                lambda_new[(i, j)] = new_val
                
                if gradient > 0:  # Constraint violation
                    violation_count += 1
            except KeyError:
                # Skip pairs not in the training set
                lambda_new[(i, j)] = lambda_values.get((i, j), 0)
        
        # Update tau
        tau_gradient = 0
        alpha_sum = 0
        for (i, j) in self.constraints:
            weight = self.w_ij.get((i, j), 0)
            alpha_sum += weight * alpha_t.get((i, j), 0)
        
        tau_gradient = alpha_sum / self.A - self.eta
        tau_new = max(0.0, min(self.C_tau, tau_value + mu_tau * tau_gradient))
    
        if t % 50 == 0:
            print(f"Iteration {t}: λ sum: {sum(lambda_new.values()):.2f}, τ: {tau_new:.4f}, Violations: {violation_count}")
       
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
            judge_violations = []
            
            # Initialize lambda and tau
            lambda_t = {pair: 0 for pair in self.constraints}
            tau_t = 0.0
            
            # Run the algorithm for T iterations
            for t in range(1, self.time_horizon + 1):
                # Best response of primal player
                D_t, alpha_t = self.best_response_primal(lambda_t, tau_t, gamma)
                
                # Compute metrics
                error = self.compute_error(D_t)
                probs = self.compute_prediction_probs(D_t)
                total_violation, max_violation, judge_violation = self.compute_fairness_violation(probs, alpha_t, gamma)
                
                # Track this model and its performance
                models.append(D_t)
                errors.append(error)
                fairness_violations.append(total_violation)
                max_violations.append(max_violation)
                judge_violations.append(judge_violation)
                
                # Update dual variables
                lambda_t, tau_t = self.update_dual_variables(lambda_t, tau_t, D_t, alpha_t, gamma, t)
                
                # Early stopping check
                if t > 100 and t % 50 == 0:
                    recent_errors = errors[-20:]
                    recent_violations = max_violations[-20:]
                    
                    if (max(recent_errors) - min(recent_errors) < 0.001 and 
                        max(recent_violations) - min(recent_violations) < 0.001):
                        print(f"Early stopping at iteration {t} due to convergence")
                        break
            
            # Calculate averaged model
            final_model = self.average_models(models[-20:])  # Average last 20 models
            final_error = self.compute_error(final_model)
            final_probs = self.compute_prediction_probs(final_model)
            
            # Create dummy alpha_t for the final model
            dummy_alpha = {pair: 0.0 for pair in self.constraints}
            
            final_violation, final_max_violation, final_judge_violations = self.compute_fairness_violation(
                final_probs, dummy_alpha, gamma
            )
            
            # Store results for this gamma
            results[gamma] = {
                'models': models,
                'errors': errors,
                'fairness_violations': fairness_violations,
                'max_violations': max_violations,
                'judge_violations': judge_violations,
                'final_model': final_model,
                'final_error': final_error,
                'final_fairness_violation': final_violation,
                'final_max_violation': final_max_violation,
                'final_judge_violations': final_judge_violations,
                'lambda_final': lambda_t,
                'tau_final': tau_t
            }
            
            print(f"Completed gamma = {gamma}: Final Error = {final_error:.4f}, Final Max Violation = {final_max_violation:.4f}")
        
        return results
    
    def evaluate_by_race(self, model, test_constraints_path, gamma=0.2):
        """
        Evaluate model on test constraints grouped by judge race
        
        Args:
            model: Trained model to evaluate
            test_constraints_path: Path to test constraints JSON
            gamma: Fairness violation threshold
            
        Returns:
            race_stats: Dict mapping race to fairness violation stats
        """
        # Load test constraints
        with open(test_constraints_path, 'r') as f:
            test_constraints = json.load(f)
        
        # Get prediction probabilities
        probs = self.compute_prediction_probs(model)
        probs_by_id = {self.index_to_id[idx]: prob for idx, prob in enumerate(probs)}
        
        # Group judges by race
        judges_by_race = {}
        for judge_id, metadata in self.persona_metadata.items():
            race = metadata.get('race')
            if race in TARGET_RACES and judge_id in test_constraints:
                if race not in judges_by_race:
                    judges_by_race[race] = []
                judges_by_race[race].append(judge_id)
        
        # Calculate fairness violations by race
        race_violations = {}
        for race, judges in judges_by_race.items():
            all_violations = []
            
            for judge_id in judges:
                if judge_id not in test_constraints:
                    continue
                    
                judge_constraints = test_constraints[judge_id]
                violations = []
                
                for constraint in judge_constraints:
                    i, j = constraint["pair"]
                    if i not in probs_by_id or j not in probs_by_id:
                        continue
                        
                    violation = max(0, abs(probs_by_id[i] - probs_by_id[j]) - gamma)
                    violations.append(violation)
                
                if violations:
                    avg_violation = sum(violations) / len(violations)
                    all_violations.append(avg_violation)
            
            if all_violations:
                race_violations[race] = {
                    'mean': np.mean(all_violations),
                    'std': np.std(all_violations),
                    'median': np.median(all_violations),
                    'min': np.min(all_violations),
                    'max': np.max(all_violations),
                    'count': len(all_violations)
                }
        
        return race_violations


def plot_cross_race_results(all_results, test_gamma, train_gamma):
    """
    Create plots comparing model performance across race groups
    
    Args:
        all_results: Dict mapping training race to results
        test_gamma: Gamma value used for testing
        train_gamma: Gamma value used for training
    """
    # Plot 1: Bar chart of fairness violations by race
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    races = TARGET_RACES
    train_races = list(all_results.keys())
    
    # Set up bar positions
    bar_width = 0.25
    r1 = np.arange(len(races))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Extract mean violations for each race combination
    data = {train_race: [] for train_race in train_races}
    
    for train_race in train_races:
        for test_race in races:
            if test_race in all_results[train_race]['race_violations']:
                data[train_race].append(all_results[train_race]['race_violations'][test_race]['mean'])
            else:
                data[train_race].append(0)
    
    # Create bars
    positions = [r1, r2, r3]
    for i, train_race in enumerate(train_races):
        plt.bar(positions[i], data[train_race], width=bar_width, color=COLORS[train_race], 
                label=f'Trained on {train_race}')
    
    # Add labels and legend
    plt.xlabel('Test Judge Race', fontsize=14)
    plt.ylabel(f'Mean Fairness Violation (test γ={test_gamma})', fontsize=14)
    plt.title(f'Cross-Race Fairness Violations\n(trained with γ={train_gamma}, tested with γ={test_gamma})', fontsize=16)
    plt.xticks([r + bar_width for r in range(len(races))], races)
    plt.legend()
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f'cross_race_fairness_violations_train_{train_gamma}_test_{test_gamma}.png', dpi=300)
    
    # Plot 2: Model accuracy vs. fairness trade-off
    plt.figure(figsize=(10, 6))
    
    # Extract error and overall violation for each model
    errors = [all_results[race]['train_error'] for race in train_races]
    violations = [all_results[race]['train_violation'] for race in train_races]
    
    # Create scatter plot
    for i, race in enumerate(train_races):
        plt.scatter(errors[i], violations[i], s=100, color=COLORS[race], label=race)
    
    plt.xlabel('Classification Error', fontsize=14)
    plt.ylabel(f'Maximum Fairness Violation (train γ={train_gamma})', fontsize=14)
    plt.title('Error vs. Fairness Trade-off by Race', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'error_fairness_tradeoff_train_{train_gamma}.png', dpi=300)
    
    # Plot 3: Heatmap of fairness violations
    plt.figure(figsize=(10, 8))
    
    # Create matrix of mean violations
    matrix_data = np.zeros((len(train_races), len(races)))
    for i, train_race in enumerate(train_races):
        for j, test_race in enumerate(races):
            if test_race in all_results[train_race]['race_violations']:
                matrix_data[i, j] = all_results[train_race]['race_violations'][test_race]['mean']
    
    # Create heatmap
    sns.heatmap(matrix_data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=races, yticklabels=[f'Trained on {r}' for r in train_races])
    plt.title(f'Fairness Violation Heatmap\n(trained with γ={train_gamma}, tested with γ={test_gamma})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'violation_heatmap_train_{train_gamma}_test_{test_gamma}.png', dpi=300)


def train_and_evaluate_models():
    """
    Train models on each race group and evaluate fairness violations across races.
    Uses a higher gamma for training than for testing.
    """
    # Paths to data files
    data_path = "train200_subset.parquet"
    constraint_path = "multi_persona_data/final_train.json"
    test_constraint_path = "multi_persona_data/final_holdout.json"
    metadata_path = "multi_persona_data/persona_metadata.json"
    
    # Results container
    all_results = {}
    
    # Train a model for each race
    for train_race in TARGET_RACES:
        print(f"\n{'='*20} Training on {train_race} judges {'='*20}")
        
        output_dir = f"output_{train_race}"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Initialize algorithm with race-specific constraints
            algorithm = FairnessElicitationAlgorithm(
                data_path=data_path,
                constraint_sets_path=constraint_path,
                persona_metadata_path=metadata_path,
                train_race=train_race,
                time_horizon=500,
                output_dir=output_dir
            )
            
            # Train model with higher gamma
            print(f"Training with gamma = {TRAIN_GAMMA}")
            results = algorithm.run([TRAIN_GAMMA])
            final_model = results[TRAIN_GAMMA]['final_model']
            
            # Evaluate on test constraints grouped by race using lower gamma
            print(f"Evaluating with gamma = {TEST_GAMMA}")
            race_violations = algorithm.evaluate_by_race(
                model=final_model,
                test_constraints_path=test_constraint_path,
                gamma=TEST_GAMMA
            )
            
            # Store results
            all_results[train_race] = {
                'model': final_model,
                'train_gamma': TRAIN_GAMMA,
                'test_gamma': TEST_GAMMA,
                'train_error': results[TRAIN_GAMMA]['final_error'],
                'train_violation': results[TRAIN_GAMMA]['final_max_violation'],
                'race_violations': race_violations
            }
            
            # Save model
            model_path = os.path.join(output_dir, f"model_gamma_{TRAIN_GAMMA}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(final_model, f)
            
            # Display results
            print(f"\nModel trained on {train_race} judges with gamma = {TRAIN_GAMMA}:")
            print(f"Training Error: {results[TRAIN_GAMMA]['final_error']:.4f}")
            print(f"Training Max Violation: {results[TRAIN_GAMMA]['final_max_violation']:.4f}")
            print(f"\nTest Fairness Violations (gamma = {TEST_GAMMA}) by Judge Race:")
            for race, stats in race_violations.items():
                print(f"  {race}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, n={stats['count']}")
        
        except Exception as e:
            print(f"Error training model for {train_race} judges: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create comparison plots
    plot_cross_race_results(all_results, TEST_GAMMA, TRAIN_GAMMA)
    
    # Create judge-level distribution plots
    plot_judge_level_distribution(all_results, TEST_GAMMA, TRAIN_GAMMA)
    
    return all_results
def main():
    """Main execution function to run the complete fairness elicitation pipeline"""
    try:
        print("Starting fairness elicitation across race groups...")
        
        # Check that required files exist
        required_files = [
            "train200_subset.parquet",
            "test1000_subset.parquet",
            "multi_persona_data/final_train.json",
            "multi_persona_data/final_holdout.json",
            "multi_persona_data/persona_metadata.json"
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"Error: Required file '{file_path}' not found!")
                return None
        
        # Train and evaluate models
        results = train_and_evaluate_models()
        
        # Save results
        with open("cross_race_fairness_results.pkl", "wb") as f:
            pickle.dump(results, f)
        
        print("\nFairness elicitation analysis complete!")
        print("Generated files:")
        print(f"  - cross_race_fairness_violations_train_{TRAIN_GAMMA}_test_{TEST_GAMMA}.png")
        print(f"  - error_fairness_tradeoff_train_{TRAIN_GAMMA}.png")
        print(f"  - violation_heatmap_train_{TRAIN_GAMMA}_test_{TEST_GAMMA}.png")
        for race in TARGET_RACES:
            print(f"  - judge_violations_trained_on_{race}_train_{TRAIN_GAMMA}_test_{TEST_GAMMA}.png")
        
        # Summary of findings
        print("\nSummary of findings:")
        for train_race in TARGET_RACES:
            if train_race in results:
                print(f"\nModel trained on {train_race} judges:")
                print(f"  - Classification error: {results[train_race]['train_error']:.4f}")
                print(f"  - Fairness violation on training data (gamma={TRAIN_GAMMA}): {results[train_race]['train_violation']:.4f}")
                
                # Find the race with minimum and maximum violation
                race_violations = results[train_race]['race_violations']
                if race_violations:
                    min_race = min(race_violations.items(), key=lambda x: x[1]['mean'])
                    max_race = max(race_violations.items(), key=lambda x: x[1]['mean'])
                    
                    print(f"  - Lowest violation on {min_race[0]} judges (gamma={TEST_GAMMA}): {min_race[1]['mean']:.4f}")
                    print(f"  - Highest violation on {max_race[0]} judges (gamma={TEST_GAMMA}): {max_race[1]['mean']:.4f}")
        
        return results
    
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
def plot_judge_level_distribution(all_results, test_gamma, train_gamma):
    """
    Create distributions of judge-level fairness violations
    
    Args:
        all_results: Dict mapping training race to results
        test_gamma: Gamma value used for testing
        train_gamma: Gamma value used for training
    """
    # Load test constraints and metadata
    with open("multi_persona_data/final_holdout.json", 'r') as f:
        test_constraints = json.load(f)
    
    with open("multi_persona_data/persona_metadata.json", 'r') as f:
        persona_metadata = json.load(f)
    
    # For each trained model, calculate detailed judge-level violations
    for train_race, results in all_results.items():
        # Skip if no model available
        if 'model' not in results:
            continue
        
        model = results['model']
        
        # Load data for predictions
        df = pd.read_parquet("test1000_subset.parquet")
        
        # Process test data similar to training
        categorical_columns = ['sex', 'race', 'c_charge_degree']
        
        # Load encoder from training
        with open(f"output_{train_race}/onehot_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        
        with open(f"output_{train_race}/train_feature_columns.pkl", "rb") as f:
            train_columns = pickle.load(f)
        
        # Prepare test data
        categorical_data = df[categorical_columns].fillna('Unknown')
        encoded_test = encoder.transform(categorical_data)
        
        # Drop original categorical + label columns
        test_df_proc = df.drop(columns=categorical_columns + ['two_year_recid'])
        
        # Load encoded feature names
        with open(f"output_{train_race}/encoded_feature_names.pkl", "rb") as f:
            encoded_feature_names = pickle.load(f)
        
        # Add encoded features
        encoded_df = pd.DataFrame(encoded_test, columns=encoded_feature_names, index=test_df_proc.index)
        test_df_proc = pd.concat([test_df_proc, encoded_df], axis=1)
        
        # Ensure columns match training data
        for col in train_columns:
            if col not in test_df_proc.columns:
                test_df_proc[col] = 0
        
        # Drop extra columns
        test_df_proc = test_df_proc[[col for col in train_columns if col in test_df_proc.columns]]
        
        # Convert to numeric and fill NaNs
        test_df_proc = test_df_proc.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Get predictions
        probs = model.predict_proba(test_df_proc.values)[:, 1]
        ids = df['id'].tolist()
        probs_by_id = {id_: prob for id_, prob in zip(ids, probs)}
        
        # Calculate violations for each judge
        judges_by_race = {}
        all_violations = {}
        
        for judge_id, judge_constraints in test_constraints.items():
            if judge_id not in persona_metadata:
                continue
                
            race = persona_metadata[judge_id].get('race')
            if race not in TARGET_RACES:
                continue
                
            # Track judge by race
            if race not in judges_by_race:
                judges_by_race[race] = []
            judges_by_race[race].append(judge_id)
            
            # Calculate violations for this judge's constraints
            violations = []
            for constraint in judge_constraints:
                i, j = constraint["pair"]
                if i not in probs_by_id or j not in probs_by_id:
                    continue
                    
                violation = max(0, abs(probs_by_id[i] - probs_by_id[j]) - test_gamma)
                violations.append(violation)
            
            if violations:
                all_violations[judge_id] = np.mean(violations)
        
        # Create violin plots for each race
        plt.figure(figsize=(12, 8))
        
        # Prepare data for violin plot
        race_data = []
        race_labels = []
        
        for race in TARGET_RACES:
            if race in judges_by_race:
                race_judges = judges_by_race[race]
                race_violations = [all_violations[j] for j in race_judges if j in all_violations]
                
                if race_violations:
                    race_data.append(race_violations)
                    race_labels.append(f"{race} (n={len(race_violations)})")
        
        # Create violin plot
        if len(race_data) > 0:  # Only create plot if we have data
            violin_parts = plt.violinplot(race_data, showmeans=True, showmedians=True)
            
            # Color the violins
            for i, race in enumerate(TARGET_RACES):
                if i < len(violin_parts['bodies']) and i < len(race_data):
                    violin_parts['bodies'][i].set_facecolor(COLORS[race])
                    violin_parts['bodies'][i].set_alpha(0.7)
            
            # Add boxplot inside violins
            plt.boxplot(race_data, widths=0.15, showfliers=False)
            
            # Add labels and title
            plt.xticks(np.arange(1, len(race_labels) + 1), race_labels)
            plt.ylabel(f'Mean Fairness Violation (test γ={test_gamma})', fontsize=14)
            plt.title(f'Distribution of Judge-Level Fairness Violations\nModel Trained on {train_race} Judges (train γ={train_gamma})', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(f'judge_violations_trained_on_{train_race}_train_{train_gamma}_test_{test_gamma}.png', dpi=300)
        else:
            print(f"No judge-level violation data for model trained on {train_race} judges")

if __name__ == "__main__":
    print("Testing script execution...")
    # Simple test to confirm Python is working
    test_array = np.array([1, 2, 3])
    print(f"NumPy test: {test_array.mean()}")
    
    # Run the full analysis
    print("Starting fairness elicitation analysis...")
    results = main()
    print("Analysis complete!")