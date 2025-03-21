import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Dict, Any

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
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, costs: List[Tuple[float, float]] = None):
        """
        Goal: Fit the classifier on data with optional costs.
        
        Inputs:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
            costs: Optional list of (cost_0, cost_1) tuples for each sample
                  cost_0 is the cost of predicting 0
                  cost_1 is the cost of predicting 1
        """
        if costs is None:
            # Standard classification without costs
            self.base_classifier.fit(X, y)
        else:
            # Compute sample weights based on costs
            sample_weights = np.zeros(len(y))
            
            for i, (c0, c1) in enumerate(costs):
                if y[i] == 0:
                    sample_weights[i] = c1  # Cost of incorrect prediction (true=0, pred=1)
                else:
                    sample_weights[i] = c0  # Cost of incorrect prediction (true=1, pred=0)
            
            # Normalize weights
            if np.sum(sample_weights) > 0:
                sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
            else:
                sample_weights = np.ones(len(y))
            
            self.base_classifier.fit(X, y, sample_weight=sample_weights)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        goal: use classifier to make predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before making predictions.")
        
        return self.base_classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        goal: what are the probability estimates for each class?
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before making predictions.")
        
        return self.base_classifier.predict_proba(X)
    
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