import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier

class CostSensitiveClassifier:
    """
    This is our cst-sensitive classification oracle that will be used in the no-regret algorithm.
    
    This is a binary classifier (by default: logistic regression) and allows for
    cost-sensitive classification by modifying the sample weights based on costs.
    """

    def __init__(self, base_classifier=None):
        """
        base_classifier: If None, default to LogisticRegression.
        """
        self.base_classifier = base_classifier if base_classifier is not None else LogisticRegression(max_iter=1000)
        # self.base_classifier = LogisticRegression(
            #max_iter=1000, 
            #C=0.1,  # Stronger regularization
            #class_weight='balanced'  # Handle class imbalance
        #)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, costs: List[Tuple[float, float]] = None):
        """
        Fit the classifier on data with optional costs.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
            costs: Optional list of (cost_0, cost_1) tuples for each sample
                cost_0 is the cost of predicting 0
                cost_1 is the cost of predicting 1
        """
        # print(f"Fitting classifier with X shape={X.shape}, y shape={y.shape}, y distribution={np.bincount(y)}")
        
        if costs is None:
            # Standard classification without costs
            print("Using standard classification (no costs)")
            self.base_classifier.fit(X, y)
        else:
            self.sample_costs = costs
            # Debug costs
            # print(f"Using cost-sensitive classification with {len(costs)} cost pairs")
            cost_array = np.array(costs)
            # print(f"Cost ranges - cost_0: [{cost_array[:, 0].min():.4f}, {cost_array[:, 0].max():.4f}], "
                # f"cost_1: [{cost_array[:, 1].min():.4f}, {cost_array[:, 1].max():.4f}]")
            
            # Compute sample weights based on costs
            sample_weights = np.zeros(len(y))
            
            for i, (c0, c1) in enumerate(costs):
                if y[i] == 0:
                    sample_weights[i] = c1  # Cost of incorrect prediction (true=0, pred=1)
                else:
                    sample_weights[i] = c0  # Cost of incorrect prediction (true=1, pred=0)
            
            # Debug weights
            # print(f"Sample weights - min: {sample_weights.min():.4f}, max: {sample_weights.max():.4f}, "
                #f"mean: {sample_weights.mean():.4f}, zeros: {np.sum(sample_weights == 0)}")
            
            # Normalize weights
            if np.sum(sample_weights) > 0:
                sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
                # print(f"Normalized weights - min: {sample_weights.min():.4f}, max: {sample_weights.max():.4f}, "
                    # f"mean: {sample_weights.mean():.4f}")
            else:
                # print("WARNING: All sample weights are zero! Using uniform weights.")
                sample_weights = np.ones(len(y))
            
            try:
                self.base_classifier.fit(X, y, sample_weight=sample_weights)
                # print("Classifier fitted successfully")
            except Exception as e:
                print(f"Error fitting classifier: {e}")
                import traceback
                traceback.print_exc()
                # Try without sample weights as fallback
                print("Attempting to fit without sample weights...")
                self.base_classifier.fit(X, y)
        
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predicting")
        
        # Get probabilities
        probs = self.predict_proba(X)
        
        # Compare costs directly rather than relying on classifier's decision boundary
        preds = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            # If cost of predicting 0 > cost of predicting 1, then predict 1
            if getattr(self, 'sample_costs', None) is not None and i < len(self.sample_costs):
                cost0, cost1 = self.sample_costs[i]
                if cost0 > cost1:
                    preds[i] = 1
            else:
                # Fall back to probability if costs not available
                preds[i] = 1 if probs[i, 1] > 0.5 else 0
        
        return preds
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for each class.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Probability estimates of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before making predictions.")
        
        try:
            probs = self.base_classifier.predict_proba(X)
            # Debug: Check probability distribution
            # print(f"Probability range for class 1: [{probs[:, 1].min():.4f}, {probs[:, 1].max():.4f}]")
            return probs
        except Exception as e:
            print(f"Error in predict_proba: {e}")
            import traceback
            traceback.print_exc()
            # Return uniform probabilities as fallback
            result = np.ones((len(X), 2)) * 0.5
            return result
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        goal: get decision function scores for each subject
        (positive means more likely to belong to positive class, negative for negative class)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before making predictions.")
        
        try:
            return self.base_classifier.decision_function(X)
        except AttributeError:
            # If base classifier doesn't have decision_function, use predict_proba
            proba = self.predict_proba(X)
            return proba[:, 1] - proba[:, 0]