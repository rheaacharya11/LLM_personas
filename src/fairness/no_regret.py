import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from .classifier import CostSensitiveClassifier
import time

class NoRegretFairness:
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        constraint_weights: Dict[Tuple[int, int], float],
        id_to_index: Dict[int, int] = None,
        gamma: float = 0.0,
        eta: float = 0.0,
        C_lambda: float = 10.0,
        C_tau: float = 10.0,
        time_horizon: int = 1000,
        base_classifier=None
    ):

        """
        quick explanation of variables:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
            constraint_weights: Dictionary mapping (i, j) constraint pairs to weights
            gamma: The gamma parameter controlling the ℓ∞ fairness relaxation
            eta: The eta parameter controlling the ℓ1 fairness relaxation
            C_lambda: The bound for lambda dual variables
            C_tau: The bound for tau dual variable
            time_horizon: Number of iterations for the algorithm
            base_classifier: The base classifier to use. If None, uses LogisticRegression.
        """

        

        self.X = X
        self.y = y
        self.n = X.shape[0]

        if id_to_index:
            self.constraint_weights = {}
            skipped = 0
            for (id_i, id_j), weight in constraint_weights.items():
                if id_i in id_to_index and id_j in id_to_index:
                    idx_i, idx_j = id_to_index[id_i], id_to_index[id_j]
                    if idx_i < self.n and idx_j < self.n:
                        self.constraint_weights[(idx_i, idx_j)] = weight
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
                

        self.constraint_weights = constraint_weights
        self.constraint_pairs = list(constraint_weights.keys())
        self.gamma = gamma
        self.eta = eta
        self.C_lambda = C_lambda
        self.C_tau = C_tau
        self.time_horizon = time_horizon

        # initialize classifier
        self.classifier = CostSensitiveClassifier(base_classifier)

        # initialize algorithm state
        self.lambda_vals = {}
        self.theta = {}
        for i, j in self.constraint_pairs:
            self.lambda_vals[(i, j)] = 0.0
            self.theta[(i, j)] = 0.0
        
        self.tau = 0.0
        
        # storage for history and results
        self.classifiers = []
        self.alphas = []
        self.errors = []
        self.fairness_violations = []

    def compute_costs(self, lambda_vals: Dict[Tuple[int, int], float]) -> List[Tuple[float, float]]:
        """
        Compute the costs (c_0, c_1) for each sample based on current lambda values.
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
        compute alpha values (excess fairness violations) based on current tau and lambda.
        """
        alpha = {}
        
        for pair, weight in self.constraint_weights.items():
            # If τ·w_{ij}/|A| - λ_{ij} ≤ 0, set α_{ij} = 1, else 0
            if tau * weight - lambda_vals.get(pair, 0) <= 0:
                alpha[pair] = 1.0
            else:
                alpha[pair] = 0.0
        
        return alpha

    def compute_fairness_violation(self, classifier, alpha: Dict[Tuple[int, int], float]) -> float:
        """
        compute the fairness violation for a classifier and alpha values.
        (it's only a total value)
        """
        total_violation = 0.0
        
        # Get predictions for all samples
        preds = classifier.predict_proba(self.X)[:, 1]
        
        for (i, j), weight in self.constraint_weights.items():
            # Calculate E[h(x_i) - h(x_j)] - alpha_{ij} - gamma
            diff = preds[i] - preds[j]
            violation = max(0, diff - alpha.get((i, j), 0) - self.gamma)
            total_violation += weight * violation
        
        return total_violation / len(self.constraint_pairs)

    def compute_error(self, classifier) -> float:
        """
        Compute the classification error for a given classifier.
        """
        preds = classifier.predict(self.X)
        return np.mean(preds != self.y)

    def my_callback(t, classifier, alpha, error, fairness_violation):
        with open('logs.txt', 'a') as f:
            f.write(f"Iteration {t}: Error={error:.4f}, Violation={fairness_violation:.4f}\n")
        
        # Early stopping example
        if error < 0.2 and fairness_violation < 0.05:
            return False  # Could be used to stop training
    
    def fit(self, verbose: bool = True, callback: Callable = None) -> List[CostSensitiveClassifier]:
        """
        run the no regret algorithm
        """
        mu_lambda = 1 / (self.C_lambda * np.sqrt(np.log(self.n) / self.time_horizon))
    
        start_time = time.time()
        
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
            classifier = CostSensitiveClassifier(self.classifier.base_classifier)
            classifier.fit(self.X, self.y, costs)
            
            # Step 4: Compute alpha
            alpha = self.compute_alpha(self.tau, lambda_vals)
            
            # Step 5: Update theta
            preds = classifier.predict_proba(self.X)[:, 1]
            
            for pair in self.constraint_pairs:
                i, j = pair  # These are now indices in our dataset, not original IDs
                violation = preds[i] - preds[j] - alpha.get(pair, 0) - self.gamma
                self.theta[pair] += mu_lambda * violation
            
            # Store results
            self.classifiers.append(classifier)
            self.alphas.append(alpha)
            
            # Compute metrics
            error = self.compute_error(classifier)
            fairness_violation = self.compute_fairness_violation(classifier, alpha)
            
            self.errors.append(error)
            self.fairness_violations.append(fairness_violation)
            
            # Print progress
            if verbose and (t % 100 == 0 or t == self.time_horizon - 1):
                elapsed = time.time() - start_time
                print(f"Iteration {t+1}/{self.time_horizon} "
                    f"[{elapsed:.2f}s]: "
                    f"Error = {error:.4f}, "
                    f"Fairness Violation = {fairness_violation:.4f}")
            
            # Call callback if provided
            if callback is not None:
                callback(t, classifier, alpha, error, fairness_violation)
        
        return self.classifiers

    def get_final_classifier(self) -> CostSensitiveClassifier:
        """
        return final classifier, average of prev. ones
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
        return dictionary with errors and fairness violations for each gamma
        """

        results = {
            'gamma': [],
            'error': [],
            'fairness_violation': []
        }
        
        original_gamma = self.gamma
        
        for gamma in gammas:
            # Update gamma
            self.gamma = gamma
            
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
        
        # Restore original gamma
        self.gamma = original_gamma
        
        return results