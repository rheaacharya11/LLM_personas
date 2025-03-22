import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from .classifier import CostSensitiveClassifier
import time
from sklearn.ensemble import RandomForestClassifier

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
        C_lambda: float = 50.0,
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
        self.max_violations = []
        
        # Initialize lists for test metrics if test data is provided
        if X_test is not None and y_test is not None:
            self.test_errors = []
            self.test_fairness_violations = []
        
        # Convert constraints from ID-based to index-based if needed
        self.constraint_weights = constraint_weights
        print(f"Total constraints: {len(self.constraint_weights)}")
        self.constraint_pairs = list(self.constraint_weights.keys())
        self.gamma = gamma
        self.eta = eta
        self.C_lambda = C_lambda
        self.C_tau = C_tau
        self.time_horizon = time_horizon
        
        # Initialize classifier
        #self.classifier = CostSensitiveClassifier(base_classifier)
        self.classifier = CostSensitiveClassifier(
            RandomForestClassifier(n_estimators=100, class_weight='balanced')
        )
        # Initialize algorithm state
        self.lambda_vals = {}
        self.theta = {}
        num_constraints = len(self.constraint_pairs)
        base_lambda = 1.0 / num_constraints
        
        self.lambda_vals = {pair: base_lambda for pair in self.constraint_pairs}
        self.theta = {pair: 0.0 for pair in self.constraint_pairs}
        self.tau = 0.0
        
        # Storage for history and results
        self.classifiers = []
        self.alphas = []
        self.errors = []
        self.fairness_violations = []

    def debug_constraint_analysis(self):
        """Comprehensive analysis of constraints and weights"""
        print("\n--- Constraint Analysis ---")
        
        # Weight distribution
        weights = list(self.constraint_weights.values())
        print(f"Total constraints: {len(weights)}")
        print(f"Weight stats: min={min(weights)}, max={max(weights)}, mean={np.mean(weights)}, median={np.median(weights)}")
        
        # Detailed constraint pair statistics
        print("\nSample constraint details:")
        for (i, j), weight in list(self.constraint_weights.items())[:10]:
            print(f"Pair ({i},{j}): weight={weight}, labels={self.y[i]},{self.y[j]}")
        
    def compute_costs(self, lambda_vals: Dict[Tuple[int, int], float]) -> List[Tuple[float, float]]:
        sample_in_constraint = set()
        for i, j in lambda_vals.keys():
            sample_in_constraint.add(i)
            sample_in_constraint.add(j)
        
        costs = []
        for i in range(self.n):
            # Base classification cost from error term
            cost_0 = 1/self.n if self.y[i] == 1 else 0
            cost_1 = 1/self.n if self.y[i] == 0 else 0
            
            # Add fairness constraints
            for (x_i, x_j), lambda_val in lambda_vals.items():
                if i == x_i:
                    cost_1 += lambda_val
                if i == x_j:
                    cost_0 += lambda_val
            
            # Clip costs to reasonable range
            cost_0 = np.clip(cost_0, 1e-6, 10.0)
            cost_1 = np.clip(cost_1, 1e-6, 10.0)
            
            costs.append((cost_0, cost_1))
        
        # Print ratio distribution (avoiding division by zero)
        ratios = [c0/c1 if c1 > 0 else 1.0 for c0, c1 in costs]
        # print(f"Cost ratio (cost0/cost1): min={min(ratios):.4f}, max={max(ratios):.4f}, median={np.median(ratios):.4f}")
        
        # Calculate average costs by class
        class0_costs = [(c0, c1) for (c0, c1), y in zip(costs, self.y) if y == 0]
        class1_costs = [(c0, c1) for (c0, c1), y in zip(costs, self.y) if y == 1]
        
        #if class0_costs:
            #print(f"Class 0 samples - Cost0: {np.mean([c[0] for c in class0_costs]):.4f}, Cost1: {np.mean([c[1] for c in class0_costs]):.4f}")
        #if class1_costs:
            #print(f"Class 1 samples - Cost0: {np.mean([c[0] for c in class1_costs]):.4f}, Cost1: {np.mean([c[1] for c in class1_costs]):.4f}")
        
        # Scale costs if needed to make fairness more impactful
        scaling_factor = 5.0  # Try different values
        scaled_costs = [(c0 * scaling_factor, c1 * scaling_factor) for c0, c1 in costs]
        
        return scaled_costs
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
    
    def compute_fairness_violation(self, classifier, alpha: Dict[Tuple[int, int], float], X=None) -> dict:
        """Enhanced fairness violation tracking"""
        if X is None:
            X = self.X
        
        # Get predictions for all samples
        preds = classifier.predict_proba(X)[:, 1]
        total_violation = 0.0
        max_violation = 0.0
        raw_violations = []
        violated_count = 0
        
        for (i, j), weight in self.constraint_weights.items():
            # Raw difference in predictions
            diff = preds[i] - preds[j]
            
            # Calculate violation after accounting for alpha and gamma
            adjusted_violation = max(0, diff - alpha.get((i, j), 0) - self.gamma)
            
            # Track various metrics
            if diff > self.gamma:
                violated_count += 1
            
            max_violation = max(max_violation, diff)
            total_violation += weight * adjusted_violation
            raw_violations.append((i, j, diff, adjusted_violation))
        
        total_constraints = len(self.constraint_weights) if self.constraint_weights else 1
        
        return {
            "avg_violation": total_violation / total_constraints,
            "max_violation": max_violation,
            "percent_violated": 100 * violated_count / total_constraints,
            "raw_violations": sorted(raw_violations, key=lambda x: -x[2])[:10]  # Top 10 by raw difference
        }
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
        """
        mu_lambda = 1 / (self.C_lambda * np.sqrt(np.log(self.n) / self.time_horizon))
        
        start_time = time.time()
        """
        if verbose:
            print("\nDEBUG - Initial state:")
            print(f"  Lambda values (first 5): {list(self.lambda_vals.items())[:5]}")
            print(f"  Theta values (first 5): {list(self.theta.items())[:5]}")
        """
        
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
            if verbose and (t % 100 == 0 or t == 0):
                self.analyze_costs(costs, lambda_vals, t)
            if verbose and (t % 100 == 0 or t == 0):
                self.debug_costs(costs, t)

            classifier = CostSensitiveClassifier(self.classifier.base_classifier)
            classifier.fit(self.X, self.y, costs)

            if verbose and (t % 100 == 0 or t == 0):
                self.debug_classifier(classifier, t)
            
            # Step 4: Compute alpha
            alpha = self.compute_alpha(self.tau, lambda_vals)
            
            # Step 5: Update theta
            preds = classifier.predict_proba(self.X)[:, 1]
            
            if verbose and t % 100 == 0:
                true_preds = (preds > 0.5).astype(int)
                accuracy = np.mean(true_preds == self.y)
                print(f"  Train accuracy: {accuracy:.4f}")
                print(f"  First 10 predictions: {true_preds[:10]}")
                print(f"  First 10 true labels: {self.y[:10]}")
                                # In verbose print section
                first_10_probs = classifier.predict_proba(self.X)[:, 1]
                print("First 10 prediction probabilities:", first_10_probs[:10])
                print("First 10 true labels:", self.y[:10])
                
                violations = []
                for pair in list(self.constraint_pairs)[:5]:
                    i, j = pair
                    if i < len(preds) and j < len(preds):
                        diff = preds[i] - preds[j]
                        violations.append((pair, diff, diff > self.gamma))
                
                print(f"  Fairness violations for 5 constraints:")
                for (i, j), diff, is_violation in violations:
                    print(f"    ({i},{j}): diff={diff:.4f}, violation={is_violation}")
                """
                if t % 10 == 0 or t in [0, 1, 100, 200, 500, 999]:
                    print(f"\n--- Detailed Debug at Iteration {t} ---")
                    
                    # Prediction probability distribution
                    probs = classifier.predict_proba(self.X)[:, 1]
                    print("Prediction Probability Stats:")
                    print(f"  Min: {probs.min():.4f}")
                    print(f"  Max: {probs.max():.4f}")
                    print(f"  Mean: {probs.mean():.4f}")
                    print(f"  Median: {np.median(probs):.4f}")
                    
                    # Lambda and theta tracking
                    lambda_values = [lambda_vals.get(pair, 0) for pair in self.constraint_pairs]
                    theta_values = [self.theta.get(pair, 0) for pair in self.constraint_pairs]
                    
                    print("\nLambda Values:")
                    print(f"  Min: {min(lambda_values):.4f}")
                    print(f"  Max: {max(lambda_values):.4f}")
                    print(f"  Mean: {np.mean(lambda_values):.4f}")
                    
                    print("\nTheta Values:")
                    print(f"  Min: {min(theta_values):.4f}")
                    print(f"  Max: {max(theta_values):.4f}")
                    print(f"  Mean: {np.mean(theta_values):.4f}")
                    """
            for pair in self.constraint_pairs:
                i, j = pair
                if i < len(preds) and j < len(preds):
                    violation = preds[i] - preds[j] - alpha.get(pair, 0) - self.gamma
                    self.theta[pair] += mu_lambda * violation
            
            # Store results
            self.classifiers.append(classifier)
            self.alphas.append(alpha)
            
            # Compute metrics on training data only
            train_error = self.compute_error(classifier)
            #train_fairness_violation = self.compute_fairness_violation(classifier, alpha)
            
            self.errors.append(train_error)
            #self.fairness_violations.append(train_fairness_violation)
            violation_metrics = self.compute_fairness_violation(classifier, alpha)
            self.fairness_violations.append(violation_metrics["avg_violation"])
            self.max_violations.append(violation_metrics["max_violation"])

            
            # Print progress
            if verbose and (t % 100 == 0 or t == self.time_horizon - 1):
                elapsed = time.time() - start_time
                #print(f"Iteration {t+1}/{self.time_horizon} [{elapsed:.2f}s]: "
                    #f"Train Error = {train_error:.4f}, "
                    #f"Train Fairness Violation = {train_fairness_violation:.4f}")
                print(f"Iteration {t+1}/{self.time_horizon} [{elapsed:.2f}s]: "
                    f"Train Error = {train_error:.4f}, "
                    f"Avg Fairness Violation = {violation_metrics['avg_violation']:.4f}, "
                    f"Max Violation = {violation_metrics['max_violation']:.4f}, "
                    f"Constraints Violated = {violation_metrics['percent_violated']:.2f}%")
                                
                if t % 100 == 0:
                    print(f"  Tau: {self.tau:.4f}")
                    print(f"  Lambda sum: {sum(lambda_vals.values()):.4f}")
                    print(f"  Max lambda: {max(lambda_vals.values()) if lambda_vals else 0:.4f}")
                    print(f"  Max theta: {max(self.theta.values()) if self.theta else 0:.4f}")
            
            # Call callback if provided
            if callback is not None:
                callback(t, classifier, alpha, train_error, train_fairness_violation)
        
        return self.classifiers

    def evaluate_on_test(self, X_test, y_test):
        """Evaluate only prediction accuracy on test data"""
        final_classifier = self.get_final_classifier()
        
        # Calculate test error
        test_error = self.compute_error(final_classifier, X_test, y_test)
        
        return {
            "test_error": test_error,
            "test_accuracy": 1 - test_error
        }
    
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

    def validate_constraints_application(self, costs):
        """Validate that constraints are correctly applied in costs"""
        sample_indices = np.random.choice(self.n, min(10, self.n), replace=False)
        
        for i in sample_indices:
            cost_0, cost_1 = costs[i]
            print(f"Validating sample {i}:")
            
            # Manually recalculate costs
            expected_cost_0 = 1/self.n if self.y[i] == 1 else 0
            expected_cost_1 = 1/self.n if self.y[i] == 0 else 0
            
            for (x_i, x_j), lambda_val in self.lambda_vals.items():
                if i == x_i:
                    expected_cost_1 += lambda_val
                if i == x_j:
                    expected_cost_0 += lambda_val
            
            print(f"  Expected: cost_0={expected_cost_0:.6f}, cost_1={expected_cost_1:.6f}")
            print(f"  Actual: cost_0={cost_0:.6f}, cost_1={cost_1:.6f}")
            
            if abs(expected_cost_0 - cost_0) > 1e-6 or abs(expected_cost_1 - cost_1) > 1e-6:
                print("  ERROR: Costs don't match expected values!")

    def visualize_costs(self, costs, iteration):
        import matplotlib.pyplot as plt
        
        cost_0_values = [c[0] for c in costs]
        cost_1_values = [c[1] for c in costs]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(cost_0_values, bins=20)
        ax1.set_title('Cost_0 Distribution')
        ax1.set_xlabel('Cost Value')
        ax1.set_ylabel('Frequency')
        
        ax2.hist(cost_1_values, bins=20)
        ax2.set_title('Cost_1 Distribution')
        ax2.set_xlabel('Cost Value')
        
        plt.suptitle(f'Cost Distributions at Iteration {iteration}')
        plt.tight_layout()
        plt.savefig(f'cost_distribution_iter_{iteration}.png')
        plt.close()

    def debug_costs(self, costs, iteration_num):
        """Debug cost computation"""
        # Summarize cost distribution
        cost_0_values = [c[0] for c in costs]
        cost_1_values = [c[1] for c in costs]
        
        print(f"\nIteration {iteration_num} - Cost Analysis:")
        print(f"  Cost 0: min={min(cost_0_values):.4f}, max={max(cost_0_values):.4f}, mean={np.mean(cost_0_values):.4f}")
        print(f"  Cost 1: min={min(cost_1_values):.4f}, max={max(cost_1_values):.4f}, mean={np.mean(cost_1_values):.4f}")
        
        # Check for extreme or suspicious values
        zero_costs = sum(1 for c0, c1 in costs if c0 == 0 and c1 == 0)
        very_high_costs = sum(1 for c0, c1 in costs if c0 > 10 or c1 > 10)
        
        print(f"  Samples with zero costs: {zero_costs}/{len(costs)}")
        print(f"  Samples with very high costs: {very_high_costs}/{len(costs)}")
        
        # Show sample costs
        print("\n  Sample costs:")
        for i in range(min(5, len(costs))):
            print(f"    Sample {i}: Label={self.y[i]}, Cost0={costs[i][0]:.4f}, Cost1={costs[i][1]:.4f}")

    def debug_classifier(self, classifier, iteration_num):
        """Debug classifier performance after training"""
        try:
            # Get predictions on training data
            preds = classifier.predict(self.X)
            probs = classifier.predict_proba(self.X)
            
            # Analyze prediction distribution
            accuracy = np.mean(preds == self.y)
            class_distribution = np.bincount(preds, minlength=2)
            
            print(f"\nIteration {iteration_num} - Classifier Analysis:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Prediction distribution: {class_distribution}")
            print(f"  True label distribution: {np.bincount(self.y, minlength=2)}")
            
            # Check prediction probabilities
            prob_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
            prob_counts = np.histogram(probs[:, 1], bins=prob_bins)[0]
            print(f"  Probability distribution: {prob_counts}")
            
            # Check specific samples
            print("\n  Sample predictions:")
            for i in range(min(5, len(preds))):
                print(f"    Sample {i}: True={self.y[i]}, Pred={preds[i]}, Prob={probs[i][1]:.4f}")
                
        except Exception as e:
            print(f"Error in debug_classifier: {e}")

    def analyze_costs(self, costs, lambda_vals=None, iteration=None):
        """Analyze cost distribution to identify biases or issues"""
        cost_0_values = [c[0] for c in costs]
        cost_1_values = [c[1] for c in costs]
        
        print(f"\n{'Iteration '+str(iteration)+' - ' if iteration is not None else ''}Cost Analysis:")
        print(f"  Cost 0: min={min(cost_0_values):.4f}, max={max(cost_0_values):.4f}, mean={np.mean(cost_0_values):.4f}")
        print(f"  Cost 1: min={min(cost_1_values):.4f}, max={max(cost_1_values):.4f}, mean={np.mean(cost_1_values):.4f}")
        
        # Cost ratios analysis
        ratios = []
        for c0, c1 in costs:
            if c1 > 0:
                ratios.append(c0/c1)
            else:
                ratios.append(float('inf'))
        
        finite_ratios = [r for r in ratios if r != float('inf')]
        if finite_ratios:
            print(f"  Cost0/Cost1 ratio: min={min(finite_ratios):.4f}, max={max(finite_ratios):.4f}, mean={np.mean(finite_ratios):.4f}")
        
        # Analyze cost by true label
        cost_0_by_class = [costs[i][0] for i in range(len(costs)) if self.y[i] == 0]
        cost_1_by_class = [costs[i][1] for i in range(len(costs)) if self.y[i] == 1]
        
        if cost_0_by_class:
            print(f"  Class 0 samples - Cost0: {np.mean(cost_0_by_class):.4f}, Cost1: {np.mean([costs[i][1] for i in range(len(costs)) if self.y[i] == 0]):.4f}")
        if cost_1_by_class:
            print(f"  Class 1 samples - Cost0: {np.mean([costs[i][0] for i in range(len(costs)) if self.y[i] == 1]):.4f}, Cost1: {np.mean(cost_1_by_class):.4f}")
        
        # Look at extreme cost samples
        print("\n  Samples with extreme costs:")
        extreme_indices = []
        for i, (c0, c1) in enumerate(costs):
            if c0 > 10 or c1 > 10 or (c0 == 0 and c1 == 0):
                extreme_indices.append(i)
        
        for i in extreme_indices[:5]:
            print(f"    Sample {i}: Label={self.y[i]}, Cost0={costs[i][0]:.4f}, Cost1={costs[i][1]:.4f}")
        
        # Check lambda influence if provided
        if lambda_vals:
            affected_samples = set()
            for (i, j), val in lambda_vals.items():
                if val > 0:
                    affected_samples.add(i)
                    affected_samples.add(j)
            print(f"\n  Lambda affects {len(affected_samples)}/{self.n} samples")