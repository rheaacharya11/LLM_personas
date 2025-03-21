import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from .classifier import CostSensitiveClassifier
import time

class NoRegretFairness:
    """
    Implementation of the No-Regret Dynamics algorithm for fairness elicitation
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        constraint_weights: Dict[Tuple[int, int], float],
        id_to_index: Dict[int, int] = None,
        X_test: np.ndarray = None,  # Add this parameter
        y_test: np.ndarray = None,  # Add this parameter
        gamma: float = 0.0,
        eta: float = 0.0,
        C_lambda: float = 10.0,
        C_tau: float = 10.0,
        time_horizon: int = 1000,
        base_classifier=None
    ):
        """
        Initialize the no-regret fairness algorithm.
        """
        self.X = X
        self.y = y
        self.n = X.shape[0]
        # Store test data if provided
        self.X_test = X_test
        self.y_test = y_test
        
        # Initialize lists for test metrics if test data is provided
        if X_test is not None and y_test is not None:
            self.test_errors = []
            self.test_fairness_violations = []
        
        # Convert constraints from ID-based to index-based if needed
        self.constraint_weights = {}
        
        if id_to_index:
            skipped = 0
            for (id_i, id_j), weight in constraint_weights.items():
                if id_i in id_to_index and id_j in id_to_index:
                    idx_i, idx_j = id_to_index[id_i], id_to_index[id_j]
                    if idx_i < self.n and idx_j < self.n:
                        self.constraint_weights[(idx_i, idx_j)] = weight
                    else:
                        skipped += 1
                else:
                    skipped += 1
            print(f"Mapped constraints: {len(self.constraint_weights)}, Skipped: {skipped}")
        else:
            # Filter any constraints with invalid indices
            self.constraint_weights = {
                pair: weight for pair, weight in constraint_weights.items()
                if pair[0] < self.n and pair[1] < self.n
            }
            if len(self.constraint_weights) != len(constraint_weights):
                print(f"Filtered {len(constraint_weights) - len(self.constraint_weights)} constraints with invalid indices")
                
        self.constraint_pairs = list(self.constraint_weights.keys())
        self.gamma = gamma
        self.eta = eta
        self.C_lambda = C_lambda
        self.C_tau = C_tau
        self.time_horizon = time_horizon
        
        # Initialize classifier
        self.classifier = CostSensitiveClassifier(base_classifier)
        
        # Initialize algorithm state
        self.lambda_vals = {}
        self.theta = {}
        for i, j in self.constraint_pairs:
            self.lambda_vals[(i, j)] = 0.0
            self.theta[(i, j)] = 0.0
        
        self.tau = 0.0
        
        # Storage for history and results
        self.classifiers = []
        self.alphas = []
        self.errors = []
        self.fairness_violations = []
        
    def compute_costs(self, lambda_vals: Dict[Tuple[int, int], float]) -> List[Tuple[float, float]]:
        """
        Compute the costs for each sample based on current lambda values.
        """
        costs = []
        
        for i in range(self.n):
            # Base classification cost from error term
            cost_0 = 1/self.n if self.y[i] == 1 else 0
            cost_1 = 1/self.n if self.y[i] == 0 else 0
            
            # Add costs from lambda terms
            for (x_i, x_j), lambda_val in lambda_vals.items():
                if i == x_i:
                    # Cost for predicting x_i as 1
                    cost_1 += lambda_val
                
                if i == x_j:
                    # Cost for predicting x_j as 0
                    cost_0 += lambda_val
            
            costs.append((cost_0, cost_1))
        
        return costs
    
    def compute_alpha(self, tau: float, lambda_vals: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """
        Compute alpha values (excess fairness violations) based on current tau and lambda.
        """
        alpha = {}
        
        for pair, weight in self.constraint_weights.items():
            # If τ·w_{ij}/|A| - λ_{ij} ≤ 0, set α_{ij} = 1, else 0
            if tau * weight - lambda_vals.get(pair, 0) <= 0:
                alpha[pair] = 1.0
            else:
                alpha[pair] = 0.0
        
        return alpha
    
    def compute_fairness_violation(self, classifier, alpha: Dict[Tuple[int, int], float], X=None) -> float:
        if X is None:
            X = self.X
            
        total_violation = 0.0
        
        # Get predictions for all samples
        preds = classifier.predict_proba(X)[:, 1]
        
        # Debugging counts
        violation_count = 0
        constraint_count = 0
        
        for (i, j), weight in self.constraint_weights.items():
            # Ensure indices are valid for the dataset
            if i < len(preds) and j < len(preds):
                constraint_count += 1
                # Calculate E[h(x_i) - h(x_j)] - alpha_{ij} - gamma
                diff = preds[i] - preds[j]
                violation = max(0, diff - alpha.get((i, j), 0) - self.gamma)
                
                if violation > 0:
                    violation_count += 1
                    
                total_violation += weight * violation
        
        # Debug info
        if constraint_count > 0 and violation_count == 0:
            # Sample a few constraints to see differences
            sample_constraints = list(self.constraint_weights.items())[:5]
            # print("FAIRNESS DEBUG - No violations found. Sample constraints:")
            for (i, j), weight in sample_constraints:
                if i < len(preds) and j < len(preds):
                    diff = preds[i] - preds[j]
                    threshold = alpha.get((i, j), 0) + self.gamma
                    # print(f"  Pair ({i},{j}): diff={diff:.4f}, threshold={threshold:.4f}, violation={max(0, diff-threshold):.4f}")
            
        result = total_violation / len(self.constraint_pairs) if self.constraint_pairs else 0
        
        # More debug info
        # if X is self.X:  # Only for training data
            # print(f"FAIRNESS CALC - Constraints checked: {constraint_count}, Violations found: {violation_count}, Total violation: {result:.6f}")
        
        return result
    
    def compute_error(self, classifier, X=None, y=None) -> float:
        """
        Compute the classification error for a given classifier.
        
        Args:
            classifier: The classifier to evaluate
            X: Feature matrix (defaults to training data)
            y: True labels (defaults to training labels)
            
        Returns:
            The classification error
        """
        if X is None:
            X = self.X
        if y is None:
            y = self.y
            
        try:
            preds = classifier.predict(X)
            incorrect = np.sum(preds != y)
            error = incorrect / len(y)
            
            return error
        except Exception as e:
            print(f"Error in compute_error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def fit(self, verbose: bool = True, callback: Callable = None) -> List[CostSensitiveClassifier]:
        """
        Run the no-regret algorithm.
        
        Args:
            verbose: Whether to print progress information
            callback: Optional callback function to call after each iteration
            
        Returns:
            List of classifiers trained during the algorithm
        """
        mu_lambda = 1 / (self.C_lambda * np.sqrt(np.log(self.n) / self.time_horizon))
        
        start_time = time.time()
        
        # DEBUG: Print initial lambda and theta values
        if verbose:
            print("\nDEBUG - Initial state:")
            print(f"  Lambda values (first 5): {list(self.lambda_vals.items())[:5]}")
            print(f"  Theta values (first 5): {list(self.theta.items())[:5]}")
        
        for t in range(self.time_horizon):
            # Step 1: Set lambda based on theta
            lambda_sum = sum(np.exp(self.theta[pair]) for pair in self.constraint_pairs)
            lambda_vals = {}
            
            for pair in self.constraint_pairs:
                lambda_vals[pair] = self.C_lambda * np.exp(self.theta[pair]) / (1 + lambda_sum)
            
            # Step 2: Set tau
            mu_tau = self.C_tau / np.sqrt(self.time_horizon)
            if t > 0:
                # Compute weighted sum of alphas
                weighted_alpha_sum = sum(
                    self.constraint_weights[pair] * self.alphas[-1].get(pair, 0) 
                    for pair in self.constraint_pairs
                )
                
                # Project tau onto [0, C_tau]
                self.tau = max(0, min(self.C_tau, self.tau + mu_tau * (weighted_alpha_sum - self.eta)))
            
            # Step 3: Compute costs and train classifier
            costs = self.compute_costs(lambda_vals)
            
            # DEBUG: Print costs for a few examples
            if verbose and t % 100 == 0:
                print(f"\nDEBUG - Iteration {t} - Sample costs:")
                for i in range(min(5, len(costs))):
                    print(f"  Sample {i}: cost_0={costs[i][0]:.4f}, cost_1={costs[i][1]:.4f}, true_label={self.y[i]}")
            
            classifier = CostSensitiveClassifier(self.classifier.base_classifier)
            classifier.fit(self.X, self.y, costs)
            
            # Step 4: Compute alpha
            alpha = self.compute_alpha(self.tau, lambda_vals)
            
            # Step 5: Update theta
            preds = classifier.predict_proba(self.X)[:, 1]
            
            # DEBUG: Check predictions
            if verbose and t % 100 == 0:
                true_preds = (preds > 0.5).astype(int)
                accuracy = np.mean(true_preds == self.y)
                print(f"  Train accuracy: {accuracy:.4f}")
                print(f"  First 10 predictions: {true_preds[:10]}")
                print(f"  First 10 true labels: {self.y[:10]}")
                
                # Check fairness violations
                violations = []
                for pair in list(self.constraint_pairs)[:5]:  # Check a few constraints
                    i, j = pair
                    if i < len(preds) and j < len(preds):
                        diff = preds[i] - preds[j]
                        violations.append((pair, diff, diff > self.gamma))
                
                print(f"  Fairness violations for 5 constraints:")
                for (i, j), diff, is_violation in violations:
                    print(f"    ({i},{j}): diff={diff:.4f}, violation={is_violation}")
            
            for pair in self.constraint_pairs:
                i, j = pair
                # Ensure indices are valid for the dataset
                if i < len(preds) and j < len(preds):
                    violation = preds[i] - preds[j] - alpha.get(pair, 0) - self.gamma
                    self.theta[pair] += mu_lambda * violation
                    
                    # DEBUG: Track significant theta updates
                    #if verbose and t % 100 == 0 and abs(mu_lambda * violation) > 0.01:
                        # print(f"    Significant theta update for pair {pair}: +{mu_lambda * violation:.4f}")
            
            # Store results
            self.classifiers.append(classifier)
            self.alphas.append(alpha)
            
            # Compute metrics on training data
            train_error = self.compute_error(classifier)
            train_fairness_violation = self.compute_fairness_violation(classifier, alpha)
            
            self.errors.append(train_error)
            self.fairness_violations.append(train_fairness_violation)
            
            if hasattr(self, 'X_test') and self.X_test is not None and hasattr(self, 'y_test') and self.y_test is not None:
                test_error = self.compute_error(classifier, self.X_test, self.y_test)
                test_fairness_violation = self.compute_fairness_violation(classifier, alpha, self.X_test)
                
                self.test_errors.append(test_error)
                self.test_fairness_violations.append(test_fairness_violation)

            # Print progress
            if verbose and (t % 100 == 0 or t == self.time_horizon - 1):
                elapsed = time.time() - start_time
                progress_msg = (f"Iteration {t+1}/{self.time_horizon} "
                            f"[{elapsed:.2f}s]: "
                            f"Train Error = {train_error:.4f}, "
                            f"Train Fairness Violation = {train_fairness_violation:.4f}")
                
                # Add test metrics to progress message if available
                if hasattr(self, 'X_test') and self.X_test is not None:
                    progress_msg += (f", Test Error = {test_error:.4f}, "
                                f"Test Fairness Violation = {test_fairness_violation:.4f}")
                
                print(progress_msg)
                
                # DEBUG: Check overall state
                if t % 100 == 0:
                    print(f"  Tau: {self.tau:.4f}")
                    print(f"  Lambda sum: {sum(lambda_vals.values()):.4f}")
                    print(f"  Max lambda: {max(lambda_vals.values()) if lambda_vals else 0:.4f}")
                    print(f"  Max theta: {max(self.theta.values()) if self.theta else 0:.4f}")
            
            # Call callback if provided
            if callback is not None:
                callback(t, classifier, alpha, train_error, train_fairness_violation)
        
        return self.classifiers
    
    def get_final_classifier(self) -> Any:
        """
        Get the final classifier by averaging all classifiers.
        """
        if not self.classifiers:
            raise RuntimeError("Algorithm has not been run yet.")
        
        # Create a simple averaged classifier
        class AveragedClassifier:
            def __init__(self, classifiers):
                self.classifiers = classifiers
            
            def predict(self, X):
                # Average predictions
                preds = np.array([clf.predict(X) for clf in self.classifiers])
                return np.round(np.mean(preds, axis=0)).astype(int)
            
            def predict_proba(self, X):
                # Average probabilities
                probs = np.array([clf.predict_proba(X) for clf in self.classifiers])
                return np.mean(probs, axis=0)
        
        return AveragedClassifier(self.classifiers)
    
    def get_pareto_curve(self, gammas: List[float]) -> Dict[str, List[float]]:
        """
        Generate the Pareto curve for different gamma values.
        """
        results = {
            'gamma': [],
            'error': [],
            'fairness_violation': []
        }
        
        original_gamma = self.gamma
        
        for gamma in gammas:
            # Reset algorithm state
            self.gamma = gamma
            self.lambda_vals = {}
            self.theta = {}
            for i, j in self.constraint_pairs:
                self.lambda_vals[(i, j)] = 0.0
                self.theta[(i, j)] = 0.0
            
            self.tau = 0.0
            self.classifiers = []
            self.alphas = []
            self.errors = []
            self.fairness_violations = []
            
            # Run algorithm
            self.fit(verbose=False)
            
            # Get final classifier
            final_clf = self.get_final_classifier()
            
            # Compute metrics
            error = self.compute_error(final_clf)
            fairness_violation = self.compute_fairness_violation(final_clf, {})
            
            # Store results
            results['gamma'].append(gamma)
            results['error'].append(error)
            results['fairness_violation'].append(fairness_violation)
            
            print(f"Gamma = {gamma:.2f}: Error = {error:.4f}, Fairness Violation = {fairness_violation:.4f}")
        
        # Restore original gamma
        self.gamma = original_gamma
        
        return results