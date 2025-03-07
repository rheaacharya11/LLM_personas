import numpy as np
from typing import List, Dict, Tuple, Set, Callable, Any, Optional
import random

class FairnessElicitation:
    def __init__(self, 
                 X: np.ndarray,
                 y: np.ndarray,
                 constraint_pairs: Set[Tuple[int, int]],
                 weights: Dict[Tuple[int, int], float],
                 gamma: float = 0.0,
                 eta: float = 0.1,
                 C_lambda: float = 1.0,
                 C_tau: float = 1.0,
                 oracle: Optional[Callable] = None):
        """
        Initialize the Fairness Elicitation algorithm.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Labels vector, shape (n_samples,)
            constraint_pairs: Set of tuples (i,j) indicating constraint pairs (x_i, x_j) ∈ C
            weights: Dictionary mapping constraint pairs (i,j) to weights w_{ij}
            gamma: Fairness violation threshold
            eta: The η parameter for allowed fairness violation budget
            C_lambda: Bound for the dual variable λ
            C_tau: Bound for the dual variable τ
            oracle: Cost-sensitive classification oracle (if None, a simple threshold classifier is used)
        """
        self.X = X
        self.y = y
        self.n = len(X)
        self.constraint_pairs = constraint_pairs
        self.weights = weights
        self.gamma = gamma
        self.eta = eta
        self.C_lambda = C_lambda
        self.C_tau = C_tau
        
        # Set A is the set of ordered pairs we've elicited constraints over
        self.A = len(constraint_pairs)
        
        # Initialize the oracle
        self.oracle = oracle if oracle else self._simple_oracle
    
    def _simple_oracle(self, costs: List[Tuple[float, float]]) -> np.ndarray:
        """
        A simple cost-sensitive classification oracle that minimizes the weighted cost.
        For each sample i, it chooses the class (0 or 1) with the minimum cost.
        
        Args:
            costs: List of cost pairs (cost_0, cost_1) for each sample
            
        Returns:
            Binary predictions for all samples
        """
        predictions = np.zeros(self.n)
        for i in range(self.n):
            predictions[i] = 0 if costs[i][0] < costs[i][1] else 1
        return predictions
    
    def best_response(self, lambda_t: np.ndarray, tau_t: float) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
        """
        Compute the best response for the primal player given dual variables lambda and tau.
        
        Args:
            lambda_t: The lambda dual variables at iteration t
            tau_t: The tau dual variable at iteration t
            
        Returns:
            D_t: The classifier (hypothesis) for iteration t
            alpha_t: The alpha values (fairness violation terms) for iteration t
        """
        # Initialize costs for cost-sensitive classification
        costs = [(0, 0) for _ in range(self.n)]
        
        # Set costs based on lambda values and labels
        for i in range(self.n):
            # Sum of (lambda_{ij} - lambda_{ji}) for all j
            lambda_sum = sum(lambda_t[i, j] - lambda_t[j, i] for j in range(self.n) if j != i)
            
            if self.y[i] == 0:
                # Cost for predicting 0 and 1 when true label is 0
                costs[i] = (0, 1/self.n + lambda_sum)
            else:
                # Cost for predicting 0 and 1 when true label is 1
                costs[i] = (1/self.n, lambda_sum)
        
        # Get predictions from oracle
        D_t = self.oracle(costs)
        
        # Compute alpha values for each constraint pair
        alpha_t = {}
        for i, j in self.constraint_pairs:
            weight_term = tau_t * self.weights.get((i, j), 0) / self.A
            # If weight term is less than or equal to lambda, set alpha to 1, else 0
            alpha_t[(i, j)] = 1 if weight_term - lambda_t[i, j] <= 0 else 0
            
        return D_t, alpha_t
    
    def no_regret_dynamics(self, 
                          T: int = 1000, 
                          mu_lambda: float = 0.1,
                          mu_tau: Optional[List[float]] = None) -> np.ndarray:
        """
        Run the no-regret dynamics algorithm to find a fair classifier.
        
        Args:
            T: Number of iterations
            mu_lambda: Step size for lambda updates
            mu_tau: List of step sizes for tau updates (if None, use 1/sqrt(t))
            
        Returns:
            The final classifier (average of all D_t)
        """
        # Initialize parameters
        theta = np.zeros((self.n, self.n))  # Used for exponentiated gradient descent
        tau = 0.0
        
        # Store classifiers and alphas
        all_D = []
        all_alphas = []
        
        # If mu_tau not provided, use default schedule
        if mu_tau is None:
            mu_tau = [self.C_tau / np.sqrt(t+1) for t in range(T)]
        
        for t in range(T):
            # Calculate lambda_t using exponentiated gradient descent
            lambda_t = np.zeros((self.n, self.n))
            denominator = 1.0
            
            for i in range(self.n):
                for j in range(self.n):
                    if (i, j) in self.constraint_pairs:
                        denominator += np.exp(theta[i, j])
            
            # Set lambda_t values
            for i in range(self.n):
                for j in range(self.n):
                    if (i, j) in self.constraint_pairs:
                        lambda_t[i, j] = self.C_lambda * np.exp(theta[i, j]) / denominator
            
            # Get the best response for the primal player
            D_t, alpha_t = self.best_response(lambda_t, tau)
            all_D.append(D_t)
            all_alphas.append(alpha_t)
            
            # Update theta (for lambda)
            for i, j in self.constraint_pairs:
                prediction_diff = D_t[i] - D_t[j]
                # Update theta based on the gradient
                theta[i, j] = theta[i, j] + mu_lambda * (prediction_diff - alpha_t.get((i, j), 0) - self.gamma)
            
            # Calculate the weighted sum of alphas for tau update
            sum_weighted_alphas = 0
            for (i, j), alpha_val in alpha_t.items():
                if (i, j) in self.weights:
                    sum_weighted_alphas += self.weights[(i, j)] * alpha_val
            
            # Normalize by |A|
            weighted_avg = sum_weighted_alphas / self.A
            
            # Update tau with projected gradient descent
            gradient = weighted_avg - self.eta
            tau = max(0, min(self.C_tau, tau + mu_tau[t] * gradient))
            
        # Return average classifier
        return np.mean(all_D, axis=0)
    
    def evaluate_fairness_violation(self, D: np.ndarray) -> Tuple[float, Dict[Tuple[int, int], float]]:
        """
        Evaluate the fairness violation of a classifier.
        
        Args:
            D: Classifier predictions (probabilities)
            
        Returns:
            overall_violation: Overall fairness violation
            violations: Dictionary of violations for each constraint pair
        """
        violations = {}
        total_weighted_violation = 0.0
        
        for i, j in self.constraint_pairs:
            # Calculate violation: max(0, D[i] - D[j] - gamma)
            violation = max(0, D[i] - D[j] - self.gamma)
            violations[(i, j)] = violation
            
            # Add to weighted sum
            weight = self.weights.get((i, j), 0)
            total_weighted_violation += weight * violation
        
        # Normalize by |A|
        overall_violation = total_weighted_violation / self.A
        
        return overall_violation, violations
    
    def evaluate_accuracy(self, D: np.ndarray) -> float:
        """
        Evaluate the accuracy of a classifier.
        
        Args:
            D: Classifier predictions (probabilities between 0 and 1)
            
        Returns:
            Accuracy of the classifier
        """
        # Convert probabilities to binary predictions
        preds = (D >= 0.5).astype(int)
        return np.mean(preds == self.y)
    
    def pareto_curve(self, gamma_values: List[float]) -> List[Tuple[float, float]]:
        """
        Generate the Pareto curve of accuracy vs. fairness violation.
        
        Args:
            gamma_values: List of gamma values to try
            
        Returns:
            List of (accuracy, violation) tuples
        """
        results = []
        
        original_gamma = self.gamma
        for gamma in gamma_values:
            self.gamma = gamma
            D = self.no_regret_dynamics()
            accuracy = self.evaluate_accuracy(D)
            violation, _ = self.evaluate_fairness_violation(D)
            results.append((accuracy, violation))
            
        # Restore original gamma
        self.gamma = original_gamma
        
        return results


def elicit_fairness_constraints(data_pairs: List[Tuple[Dict, Dict]], 
                               stakeholders: List[Any]) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """
    Elicit fairness constraints from stakeholders.
    
    Args:
        data_pairs: List of pairs of individuals to be compared
        stakeholders: List of stakeholders to query
        
    Returns:
        constraint_pairs: Set of pairs (i,j) where stakeholders express fairness constraints
        weights: Dictionary mapping constraint pairs to weights
    """
    constraint_pairs = set()
    weights = {}
    
    for idx, (x_i, x_j) in enumerate(data_pairs):
        # Count stakeholders who express preference for each option
        equal_count = 0
        i_better_than_j = 0
        j_better_than_i = 0
        
        for stakeholder in stakeholders:
            # This is a placeholder for the actual elicitation function
            # In practice, this would ask the stakeholder for their preference
            # 1: No constraint
            # 2: j should be treated as well as i or better
            # 3: i should be treated as well as j or better
            # 4: i and j should be treated similarly
            preference = get_stakeholder_preference(stakeholder, x_i, x_j)
            
            if preference == 2:
                j_better_than_i += 1
                constraint_pairs.add((i, j))
            elif preference == 3:
                i_better_than_j += 1
                constraint_pairs.add((j, i))
            elif preference == 4:
                equal_count += 1
                constraint_pairs.add((i, j))
                constraint_pairs.add((j, i))
        
        # Calculate weights based on stakeholder preferences
        if (i, j) in constraint_pairs:
            weights[(i, j)] = (j_better_than_i + equal_count) / len(stakeholders)
        if (j, i) in constraint_pairs:
            weights[(j, i)] = (i_better_than_j + equal_count) / len(stakeholders)
    
    return constraint_pairs, weights


def get_stakeholder_preference(stakeholder: Any, x_i: Dict, x_j: Dict) -> int:
    """
    Placeholder function for getting stakeholder preference.
    In a real implementation, this would query the stakeholder.
    
    Args:
        stakeholder: Stakeholder to query
        x_i: First individual
        x_j: Second individual
        
    Returns:
        Preference:
            1: No constraint
            2: j should be treated as well as i or better
            3: i should be treated as well as j or better
            4: i and j should be treated similarly
    """
    # This is just a placeholder
    # In practice, this would be a function that queries the stakeholder
    return random.choice([1, 2, 3, 4])


# Example usage
if __name__ == "__main__":
    # Sample data
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Generate random constraint pairs
    constraint_pairs = set()
    weights = {}
    
    # Randomly select 20 pairs for constraints
    for _ in range(20):
        i, j = np.random.choice(len(X), 2, replace=False)
        constraint_pairs.add((i, j))
        weights[(i, j)] = np.random.rand()  # Random weight
    
    # Initialize fairness elicitation
    fairness = FairnessElicitation(
        X=X,
        y=y,
        constraint_pairs=constraint_pairs,
        weights=weights,
        gamma=0.1,
        eta=0.05,
        C_lambda=1.0,
        C_tau=1.0
    )
    
    # Run the algorithm
    D = fairness.no_regret_dynamics(T=100)
    
    # Evaluate results
    accuracy = fairness.evaluate_accuracy(D)
    violation, violations_by_pair = fairness.evaluate_fairness_violation(D)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Overall fairness violation: {violation:.4f}")