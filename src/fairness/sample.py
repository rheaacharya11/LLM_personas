import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse import lil_matrix
from sklearn.datasets import make_classification
import pandas as pd

class CostSensitiveClassificationOracle:
    """
    Cost-sensitive classification oracle as described in Jung et al.
    This implementation uses logistic regression as the base classifier.
    """
    def __init__(self):
        self.model = None
    
    def __call__(self, X, costs):
        """
        Trains a classifier that minimizes the weighted cost.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            costs: List of tuples (c0, c1) where:
                - c0 is the cost of classifying the instance as 0
                - c1 is the cost of classifying the instance as 1
        
        Returns:
            Binary predictions minimizing the cost
        """
        # Convert costs to numpy array
        costs = np.array(costs)
        
        # Training labels are determined by which class has lower cost
        y_train = (costs[:, 0] > costs[:, 1]).astype(int)
        
        # Sample weights are the absolute difference in costs
        sample_weights = np.abs(costs[:, 0] - costs[:, 1])
        
        # Train a weighted logistic regression model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X, y_train, sample_weight=sample_weights)
        
        # Return predictions
        return self.model.predict(X)


class FairnessElicitationAlgorithm:
    """
    Implementation of Jung et al.'s No-Regret Learning for Fairness Elicitation
    with sparse data structures for efficient constraint handling
    """
    def __init__(self, X, y, fairness_constraints, weights=None, gamma=0.0, eta=0.1, C_lambda=2.0, C_tau=2.0):
        """
        Initialize the algorithm with sparse representations.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            fairness_constraints: List of tuples (i, j) indicating that individual i
                                 should be treated at least as well as individual j
            weights: Optional dictionary mapping (i,j) to constraint weight
            gamma: Fairness violation threshold
            eta: Maximum allowed average violation
            C_lambda: Bound on the L1 norm of lambda
            C_tau: Bound on tau
        """
        self.X = X
        self.y = y
        self.n = len(X)
        self.constraints = fairness_constraints
        self.gamma = gamma
        self.eta = eta
        self.C_lambda = C_lambda
        self.C_tau = C_tau
        
        # Create efficient constraint handling structures
        self.constraints_by_i = defaultdict(list)  # Maps i to list of j's where (i,j) is a constraint
        self.constraints_by_j = defaultdict(list)  # Maps j to list of i's where (i,j) is a constraint
        
        # Create weight dictionary for constraints (default to 1.0 if not provided)
        self.w = {} if weights is None else weights.copy()
        
        # Organize constraints by i and j for efficient access
        for i, j in self.constraints:
            self.constraints_by_i[i].append(j)
            self.constraints_by_j[j].append(i)
            if (i, j) not in self.w:
                self.w[(i, j)] = 1.0
        
        # Count total constraints for normalization
        self.num_constraints = len(fairness_constraints)
        
        # Initialize the CSC oracle
        self.csc_oracle = CostSensitiveClassificationOracle()
    
    def run(self, max_iterations=100):
        """
        Run the no-regret algorithm with sparse parameter updates.
        
        Args:
            max_iterations: Maximum number of iterations
        
        Returns:
            Best model, training error, fairness violations history
        """
        # Initialize parameters using sparse structures
        theta = {}  # Sparse dict for lambda updates
        lambda_values = {}  # Sparse dict for lambda values
        tau = 0
        
        # Initialize history tracking
        error_history = []
        fairness_violation_history = []
        
        # Save models and predictions
        models = []
        all_predictions = []
        
        # Step sizes
        mu_lambda = 1.0 / (self.C_lambda * np.sqrt(max_iterations * np.log(self.n)))
        mu_tau = self.C_tau / np.sqrt(max_iterations)
        
        # Run the algorithm
        for t in range(max_iterations):
            # 1. Calculate costs for the CSC oracle
            costs = self._calculate_costs(lambda_values)
            
            # 2. Get predictions from the CSC oracle
            preds = self.csc_oracle(self.X, costs)
            all_predictions.append(preds)
            models.append(self.csc_oracle.model)
            
            # 3. Calculate current error
            error = np.mean(preds != self.y)
            error_history.append(error)
            
            # 4. Calculate optimal alpha values
            alpha = self._calculate_alpha(preds, lambda_values, tau)
            
            # 5. Calculate fairness violation
            fairness_violation = self._calculate_fairness_violation(preds, alpha)
            fairness_violation_history.append(fairness_violation)
            
            # 6. Update dual variables using sparse operations
            # Update theta (for lambda)
            for i, j in self.constraints:
                violation = (preds[i] - preds[j]) - alpha.get((i, j), 0) - self.gamma
                if (i, j) not in theta:
                    theta[(i, j)] = 0
                theta[(i, j)] += mu_lambda * violation
            
            # Calculate normalization term for lambda update
            exp_sum = 1.0  # Start with 1 for the default value
            for val in theta.values():
                exp_sum += np.exp(val)
            
            # Update lambda using exponentiated gradient descent
            for i, j in self.constraints:
                lambda_values[(i, j)] = self.C_lambda * np.exp(theta.get((i, j), 0)) / exp_sum
            
            # Update tau using online gradient descent
            avg_weighted_alpha = sum(self.w.get((i, j), 1.0) * alpha.get((i, j), 0) 
                                   for i, j in self.constraints) / self.num_constraints
            tau = np.clip(tau + mu_tau * (avg_weighted_alpha - self.eta), 0, self.C_tau)
            
            # Print progress
            if (t+1) % 10 == 0 or t == 0:
                print(f"Iteration {t+1}: Error = {error:.4f}, Fairness Violation = {fairness_violation:.4f}")
        
        # Find the best model (minimum error that satisfies fairness constraints)
        valid_indices = [i for i, viol in enumerate(fairness_violation_history) if viol <= self.eta + 1e-6]
        
        if valid_indices:
            best_idx = min(valid_indices, key=lambda i: error_history[i])
            best_model = models[best_idx]
            best_preds = all_predictions[best_idx]
            best_error = error_history[best_idx]
            best_violation = fairness_violation_history[best_idx]
            print(f"Best model: Error = {best_error:.4f}, Fairness Violation = {best_violation:.4f}")
        else:
            print("No model satisfied the fairness constraints. Returning the model with minimum fairness violation.")
            best_idx = np.argmin(fairness_violation_history)
            best_model = models[best_idx]
            best_preds = all_predictions[best_idx]
        
        return best_model, error_history, fairness_violation_history
    
    def _calculate_costs(self, lambda_values):
        """Calculate the costs for the CSC oracle based on current lambda values"""
        costs = []
        
        # Precompute lambda sums for each individual using sparse approach
        lambda_sum_out = defaultdict(float)  # Sum of lambdas where i is first
        lambda_sum_in = defaultdict(float)   # Sum of lambdas where i is second
        
        for (i, j), val in lambda_values.items():
            lambda_sum_out[i] += val
            lambda_sum_in[j] += val
        
        for i in range(self.n):
            # Cost for predicting 0
            c0 = 1 / self.n if self.y[i] == 1 else 0
            
            # Cost for predicting 1
            c1 = 1 / self.n if self.y[i] == 0 else 0
            
            # Adjust costs based on fairness constraints (using precomputed sums)
            c1 += lambda_sum_out.get(i, 0) - lambda_sum_in.get(i, 0)
            
            costs.append((c0, c1))
        
        return costs
    
    def _calculate_alpha(self, preds, lambda_values, tau):
        """
        Calculate the optimal alpha values (excess fairness violations)
        based on the current predictions, lambda values, and tau.
        Returns a sparse dictionary of alpha values.
        """
        alpha = {}
        
        for i, j in self.constraints:
            # Calculate whether we should allocate excess violation
            # If tau * w[i,j] > lambda[i,j], we set alpha=1 (allow max violation)
            # Otherwise, we set alpha=0 (no excess violation allowed)
            if tau * self.w.get((i, j), 1.0) / self.num_constraints >= lambda_values.get((i, j), 0):
                alpha[(i, j)] = 1.0
        
        return alpha
    
    def _calculate_fairness_violation(self, preds, alpha):
        """
        Calculate the current fairness violation using sparse structures.
        Returns the average weighted fairness violation.
        """
        total_violation = 0
        
        for i, j in self.constraints:
            # Calculate the violation: prediction[i] - prediction[j] - alpha[i,j] - gamma
            # Should be <= 0 for fairness
            a_ij = alpha.get((i, j), 0)
            violation = max(0, (preds[i] - preds[j]) - a_ij - self.gamma)
            total_violation += self.w.get((i, j), 1.0) * violation
        
        # Normalize by the number of constraints
        avg_violation = total_violation / self.num_constraints if self.num_constraints > 0 else 0
        
        return avg_violation
    
    def plot_convergence(self, error_history, fairness_violation_history):
        """Plot the convergence of the algorithm."""
        plt.figure(figsize=(12, 5))
        
        # Plot error
        plt.subplot(1, 2, 1)
        plt.plot(error_history, 'b-', label='Training Error')
        plt.axhline(y=min(error_history), linestyle='--', color='gray')
        plt.xlabel('Iteration')
        plt.ylabel('Error Rate')
        plt.title('Training Error vs. Iterations')
        plt.legend()
        
        # Plot fairness violation
        plt.subplot(1, 2, 2)
        plt.plot(fairness_violation_history, 'r-', label='Fairness Violation')
        plt.axhline(y=self.eta, linestyle='--', color='gray', label=f'Target η={self.eta}')
        plt.xlabel('Iteration')
        plt.ylabel('Average Fairness Violation')
        plt.title('Fairness Violation vs. Iterations')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def test_with_synthetic_data():


    # 1. Create synthetic COMPAS-like data
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, n_features=8, n_informative=5, 
                            n_redundant=2, random_state=42)

    # Add a protected attribute (like gender or race)
    protected_attr = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    X = np.column_stack([X, protected_attr])

    # 2. Simulate persona judgments
    # Randomly generate constraints with higher probability between similar individuals
    fairness_constraints = []
    weights = {}

    # Generate similarity-based constraints
    for _ in range(5000):  # Generate about 5000 random constraints
        # Pick two random individuals
        i, j = np.random.choice(n_samples, 2, replace=False)
        
        # More likely to generate constraints between similar individuals
        if np.random.random() < 0.7 and protected_attr[i] == protected_attr[j]:
            fairness_constraints.append((i, j))
            weights[(i, j)] = np.random.uniform(0.5, 1.0)  # Random weight
        elif np.random.random() < 0.3:  # Some cross-group constraints
            fairness_constraints.append((i, j))
            weights[(i, j)] = np.random.uniform(0.1, 0.5)  # Lower weights

    print(f"Generated {len(fairness_constraints)} synthetic fairness constraints")

    # 3. Run the algorithm
    algorithm = FairnessElicitationAlgorithm(X, y, fairness_constraints, weights=weights, 
                                            gamma=0.1, eta=0.05)

    best_model, error_history, fairness_violation_history = algorithm.run(max_iterations=100)

    # 4. Evaluate and visualize the results
    algorithm.plot_convergence(error_history, fairness_violation_history)

    # Test model performance on the protected attribute
    y_pred = best_model.predict(X)
    group0_acc = np.mean(y_pred[protected_attr == 0] == y[protected_attr == 0])
    group1_acc = np.mean(y_pred[protected_attr == 1] == y[protected_attr == 1])

    print(f"Group 0 accuracy: {group0_acc:.4f}")
    print(f"Group 1 accuracy: {group1_acc:.4f}")
    print(f"Accuracy gap: {abs(group0_acc - group1_acc):.4f}")

# Example usage
def run_example():
    """Run a simple example of the fairness elicitation algorithm with sparse structures."""
    # Generate a simple dataset
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 5)
    
    # Create a biased label function that depends on the first feature
    # and has some correlation with the second feature (protected attribute)
    bias = 0.3
    y = ((X[:, 0] + bias * X[:, 1]) > 0).astype(int)
    
    # Reduce number of constraints and make them more consistent
    fairness_constraints = []
    weights = {}

    # Generate fewer, more coherent constraints
    for _ in range(1000):
        i, j = np.random.choice(n_samples, 2, replace=False)
        
        # Generate constraints that follow a clearer pattern
        # Example: similar protected attribute → should be treated similarly 
        if protected_attr[i] == protected_attr[j]:
            # Similar features also suggest similar treatment
            if np.linalg.norm(X[i, :-1] - X[j, :-1]) < 2.0:
                fairness_constraints.append((i, j))
                weights[(i, j)] = 0.8
    
    print(f"Generated {len(fairness_constraints)} fairness constraints")
    
    # Initialize and run the algorithm with sparse structures
    algo = FairnessElicitationAlgorithm(X, y, fairness_constraints, weights=weights, gamma=0.5, eta=0.01)
    best_model, error_history, fairness_violation_history = algo.run(max_iterations=2000)
    
    # Plot results
    algo.plot_convergence(error_history, fairness_violation_history)
    
    return best_model, X, y, fairness_constraints


# Run example only if executed directly
if __name__ == "__main__":
    test_with_synthetic_data()
    # best_model, X, y, fairness_constraints = run_example()