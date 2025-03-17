import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from typing import List, Tuple, Dict, Union, Optional, Any
import pandas as pd

class CostSensitiveClassificationOracle:
    """
    A cost-sensitive classification oracle that implements the primal player's best response
    in the algorithmic fairness elicitation framework from Jung et al.
    
    The oracle takes in a dataset with cost structures for classifying each instance as 0 or 1,
    and returns the hypothesis that minimizes the weighted cost.
    """
    
    def __init__(self, hypothesis_class: str = 'linear', random_state: int = 42):
        """
        Initialize the CSC oracle.
        
        Args:
            hypothesis_class: The type of classifier to use ('linear' for linear classifier)
            random_state: Random seed for reproducibility
        """
        self.hypothesis_class = hypothesis_class
        self.random_state = random_state
        self.model = None
        
    def _initialize_model(self):
        """Initialize the model based on the hypothesis class."""
        if self.hypothesis_class == 'linear':
            # Using sklearn's LogisticRegression as our linear classifier
            return LogisticRegression(random_state=self.random_state, 
                                     solver='liblinear',
                                     max_iter=1000)
        else:
            raise ValueError(f"Unsupported hypothesis class: {self.hypothesis_class}")
    
    def best_response(self, X: np.ndarray, costs: List[Tuple[float, float]]) -> np.ndarray:
        """
        Compute the best response to the given costs, following the approach in the paper.
        
        Args:
            X: Feature vectors of shape (n_samples, n_features)
            costs: List of tuples (c0, c1) where:
                  - c0 is the cost of classifying instance i as 0
                  - c1 is the cost of classifying instance i as 1
        
        Returns:
            Binary predictions minimizing the weighted cost
        """
        # Convert costs to numpy array for easier manipulation
        costs_array = np.array(costs)
        c0 = costs_array[:, 0]  # costs for labeling as 0
        c1 = costs_array[:, 1]  # costs for labeling as 1
        
        # Create labels based on which prediction has lower cost
        # Following the paper, we're looking for h that minimizes sum_i h(x_i)c1_i + (1-h(x_i))c0_i
        labels = np.where(c0 > c1, 1, 0)
        
        # Create sample weights based on the absolute difference in costs
        # This ensures that examples with larger cost differences have more influence
        weights = np.abs(c0 - c1)
        
        # Initialize and fit the model
        self.model = self._initialize_model()
        self.model.fit(X, labels, sample_weight=weights)
        
        # Return predictions
        return self.model.predict(X)
    
    def __call__(self, X: np.ndarray, costs: List[Tuple[float, float]]) -> np.ndarray:
        """
        Directly call the oracle with features and costs.
        
        Args:
            X: Feature vectors
            costs: Cost pairs for each instance
            
        Returns:
            Optimal predictions
        """
        return self.best_response(X, costs)


class ImprovedCostSensitiveOracle:
    """
    An improved cost-sensitive classification oracle that implements the primal player's
    best response in the algorithmic fairness elicitation framework, with enhancements
    for better performance.
    """
    
    def __init__(self, 
                hypothesis_class: str = 'logistic', 
                random_state: int = 42,
                class_weight: Optional[Union[Dict, str]] = 'balanced',
                C: float = 1.0,
                scale_features: bool = True):
        """
        Initialize the CSC oracle.
        
        Args:
            hypothesis_class: The type of classifier to use ('logistic', 'random_forest', or 'gbm')
            random_state: Random seed for reproducibility
            class_weight: Weight adjustment for imbalanced classes
            C: Regularization parameter for logistic regression
            scale_features: Whether to scale features
        """
        self.hypothesis_class = hypothesis_class
        self.random_state = random_state
        self.class_weight = class_weight
        self.C = C
        self.scale_features = scale_features
        self.model = None
        self.scaler = StandardScaler() if scale_features else None
        
    def _initialize_model(self):
        """Initialize the model based on the hypothesis class."""
        if self.hypothesis_class == 'logistic':
            # Using LogisticRegression with balanced class weights and adjusted C
            base_model = LogisticRegression(
                random_state=self.random_state,
                class_weight=self.class_weight,
                C=self.C,  # Controls regularization strength
                solver='liblinear',  # Works well with small datasets
                max_iter=1000
            )
        elif self.hypothesis_class == 'random_forest':
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight=self.class_weight,
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5
            )
        elif self.hypothesis_class == 'gbm':
            base_model = GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                min_samples_split=10
            )
        else:
            raise ValueError(f"Unsupported hypothesis class: {self.hypothesis_class}")
        
        # Create a pipeline with optional scaling
        if self.scale_features:
            return Pipeline([
                ('scaler', self.scaler),
                ('model', base_model)
            ])
        else:
            return base_model
    
    def best_response(self, X: np.ndarray, costs: List[Tuple[float, float]], verbose=False) -> np.ndarray:
        """
        Compute the best response to the given costs, with improved weighting.
        
        Args:
            X: Feature vectors of shape (n_samples, n_features)
            costs: List of tuples (c0, c1) where:
                  - c0 is the cost of classifying instance i as 0
                  - c1 is the cost of classifying instance i as 1
            verbose: Whether to print debugging information
            
        Returns:
            Binary predictions minimizing the weighted cost
        """
        if verbose:
            print(f"Computing best response with {self.hypothesis_class} model")
            
        # Convert costs to numpy array for easier manipulation
        costs_array = np.array(costs)
        c0 = costs_array[:, 0]  # costs for labeling as 0
        c1 = costs_array[:, 1]  # costs for labeling as 1
        
        # Create labels based on which prediction has lower cost
        labels = np.where(c0 > c1, 1, 0)
        
        # Create sample weights using a more sophisticated approach:
        # - Absolute difference in costs (basic)
        # - Add a minimum weight of 0.1 to ensure all samples have some impact
        # - Scale weights to sum to the number of samples for stability
        weights = np.abs(c0 - c1) + 0.1
        weights = weights * (len(weights) / weights.sum())  # Normalize
        
        if verbose:
            positive_ratio = np.mean(labels)
            print(f"Induced label distribution: {positive_ratio:.2f} positive, {1-positive_ratio:.2f} negative")
            print(f"Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
        
        # Initialize and fit the model
        self.model = self._initialize_model()
        
        # Handle sample weights differently depending on whether we're using a pipeline
        if self.scale_features:
            # For Pipeline, we need to route sample_weight to the final estimator
            model_name = 'model'  # The name of the estimator in the pipeline
            fit_params = {f"{model_name}__sample_weight": weights}
            self.model.fit(X, labels, **fit_params)
        else:
            # For standalone estimators, we can pass sample_weight directly
            self.model.fit(X, labels, sample_weight=weights)
        
        # Return predictions
        return self.model.predict(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call best_response() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability predictions if supported by the model."""
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call best_response() first.")
            
        # Check if model has predict_proba method (all our models should)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("This model doesn't support probability predictions")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Feature vectors
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call best_response() first.")
            
        y_pred = self.predict(X)
        
        # Calculate various metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'positive_rate': np.mean(y_pred),
        }
        
        # Add AUC if predict_proba is available
        try:
            y_prob = self.predict_proba(X)[:, 1]  # Probability of class 1
            metrics['auc'] = roc_auc_score(y, y_prob)
        except:
            metrics['auc'] = None
            
        return metrics
    
    def __call__(self, X: np.ndarray, costs: List[Tuple[float, float]], verbose=False) -> np.ndarray:
        """Directly call the oracle with features and costs."""
        return self.best_response(X, costs, verbose)


def create_better_costs(y_true: np.ndarray, skew_factor: float = 1.0) -> List[Tuple[float, float]]:
    """
    Create improved cost structures for classification.
    
    Args:
        y_true: True labels
        skew_factor: Factor to adjust the relative importance of each class
                    (>1 emphasizes positive class, <1 emphasizes negative class)
                    
    Returns:
        List of cost pairs (c0, c1) for each instance
    """
    # Calculate class distribution for better weighting
    pos_ratio = np.mean(y_true)
    neg_ratio = 1 - pos_ratio
    
    # Create balanced cost structure
    # - For positive examples (y=1):
    #   * c0 = 1.0 (cost of false negative)
    #   * c1 = 0.0 (cost of true positive)
    # - For negative examples (y=0):
    #   * c0 = 0.0 (cost of true negative)
    #   * c1 = 1.0 (cost of false positive)
    # Adjust by class distribution and skew factor
    costs = []
    for y in y_true:
        if y == 1:
            # Positive example - balance by negative ratio and skew
            c0 = 1.0 * neg_ratio * skew_factor  # False negative cost
            c1 = 0.0  # True positive cost
        else:
            # Negative example - balance by positive ratio and inverse skew
            c0 = 0.0  # True negative cost
            c1 = 1.0 * pos_ratio / skew_factor  # False positive cost
        costs.append((c0, c1))
    
    return costs


def load_and_preprocess_compas(train_path="data/compas_train.parquet", 
                              test_path="data/compas_test.parquet",
                              verbose=True):
    """
    Load and preprocess COMPAS data with improved feature engineering.
    
    Args:
        train_path: Path to training data parquet
        test_path: Path to test data parquet
        verbose: Whether to print information
        
    Returns:
        Processed data as (X_train, y_train, X_test, y_test, protected_train, protected_test)
    """
    try:
        # Load data
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        if verbose:
            print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
            print(f"Training columns: {train_df.columns.tolist()}")
        
        # Check if data already has preprocessed features
        if 'features' in train_df.columns:
            if verbose:
                print("Using pre-computed feature vectors")
            
            # Extract features and labels
            X_train = np.array(train_df['features'].tolist())
            y_train = train_df['two_year_recid'].values
            
            X_test = np.array(test_df['features'].tolist())
            y_test = test_df['two_year_recid'].values
            
            # Extract protected attributes (sex/race if available)
            protected_train = {}
            protected_test = {}
            
            if 'sex' in train_df.columns:
                protected_train['sex'] = train_df['sex'].map({'Female': 0, 'Male': 1}).values
                protected_test['sex'] = test_df['sex'].map({'Female': 0, 'Male': 1}).values
            
            if 'race' in train_df.columns:
                protected_train['race'] = train_df['race'].values
                protected_test['race'] = test_df['race'].values
                
            return X_train, y_train, X_test, y_test, protected_train, protected_test
        
        # If no preprocessed features, do manual feature engineering
        if verbose:
            print("Performing manual feature engineering")
            
        # Select and process features
        num_features = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']
        cat_features = ['sex', 'race', 'c_charge_degree']
        
        # Function to process a single dataframe
        def process_df(df):
            # Create copy to avoid modifying original
            df_proc = df.copy()
            
            # Ensure all required columns exist
            missing_cols = set(num_features + cat_features) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Basic preprocessing for categorical features
            df_proc['sex_code'] = df_proc['sex'].map({'Female': 0, 'Male': 1})
            df_proc['c_charge_degree_code'] = df_proc['c_charge_degree'].map({'F': 1, 'M': 0})
            
            # Create dummy variables for race
            race_dummies = pd.get_dummies(df_proc['race'], prefix='race')
            df_proc = pd.concat([df_proc, race_dummies], axis=1)
            
            # Extract target
            y = df_proc['two_year_recid'].values
            
            # Create feature matrix
            X_cols = (
                num_features +  # Numeric features as-is
                ['sex_code', 'c_charge_degree_code'] +  # Encoded categorical features
                list(race_dummies.columns)  # Race dummy variables
            )
            X = df_proc[X_cols].values
            
            # Create protected attributes dict
            protected = {
                'sex': df_proc['sex_code'].values,
                'race': df_proc['race'].values
            }
            
            return X, y, protected
        
        # Process both dataframes
        X_train, y_train, protected_train = process_df(train_df)
        X_test, y_test, protected_test = process_df(test_df)
        
        if verbose:
            print(f"Processed features: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
            
        return X_train, y_train, X_test, y_test, protected_train, protected_test
        
    except Exception as e:
        print(f"Error loading/preprocessing data: {e}")
        raise


def compare_models_on_compas(train_path="data/compas_train.parquet", 
                           test_path="data/compas_test.parquet"):
    """
    Compare different CSC oracle configurations on the COMPAS dataset.
    
    Args:
        train_path: Path to training data parquet
        test_path: Path to test data parquet
        
    Returns:
        Dictionary of results for different model configurations
    """
    print("\nLoading and preprocessing COMPAS data...")
    X_train, y_train, X_test, y_test, protected_train, protected_test = load_and_preprocess_compas(
        train_path, test_path
    )
    
    # Model configurations to test
    configs = [
        {"name": "Logistic (basic)", "model_class": "logistic", "scale": True, "costs": "basic"},
        {"name": "Logistic (better costs)", "model_class": "logistic", "scale": True, "costs": "better"},
        {"name": "Random Forest (basic)", "model_class": "random_forest", "scale": False, "costs": "basic"},
        {"name": "Random Forest (better costs)", "model_class": "random_forest", "scale": False, "costs": "better"},
        {"name": "GBM (better costs)", "model_class": "gbm", "scale": False, "costs": "better"}
    ]
    
    # Basic costs: simple misclassification penalty
    basic_costs = [(0 if y == 0 else 1, 0 if y == 1 else 1) for y in y_train]
    
    # Better costs: account for class imbalance
    better_costs = create_better_costs(y_train, skew_factor=1.2)  # Slightly favor positive class
    
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        # Initialize model with configuration
        model = ImprovedCostSensitiveOracle(
            hypothesis_class=config['model_class'],
            scale_features=config['scale']
        )
        
        # Select cost structure
        costs = better_costs if config['costs'] == 'better' else basic_costs
        
        # Train model
        model.best_response(X_train, costs, verbose=True)
        
        # Evaluate on train and test
        train_metrics = model.evaluate(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)
        
        # Calculate fairness metrics - disparate impact by sex
        male_idx_test = protected_test['sex'] == 1
        female_idx_test = protected_test['sex'] == 0
        
        y_pred_test = model.predict(X_test)
        male_positive_rate = np.mean(y_pred_test[male_idx_test])
        female_positive_rate = np.mean(y_pred_test[female_idx_test])
        
        # Calculate disparate impact (min/max formula so always ≤ 1)
        disparate_impact = min(female_positive_rate, male_positive_rate) / max(female_positive_rate, male_positive_rate) if max(female_positive_rate, male_positive_rate) > 0 else 1.0
        
        # Store results
        results[config['name']] = {
            'train_accuracy': train_metrics['accuracy'],
            'test_accuracy': test_metrics['accuracy'],
            'train_f1': train_metrics['f1_score'],
            'test_f1': test_metrics['f1_score'],
            'train_auc': train_metrics['auc'],
            'test_auc': test_metrics['auc'],
            'male_positive_rate': male_positive_rate,
            'female_positive_rate': female_positive_rate,
            'disparate_impact': disparate_impact
        }
        
        # Print summary
        print(f"Training accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Training F1: {train_metrics['f1_score']:.4f}")
        print(f"Test F1: {test_metrics['f1_score']:.4f}")
        print(f"Training AUC: {train_metrics['auc']:.4f}" if train_metrics['auc'] else "Training AUC: N/A")
        print(f"Test AUC: {test_metrics['auc']:.4f}" if test_metrics['auc'] else "Test AUC: N/A")
        print(f"Male positive rate: {male_positive_rate:.4f}")
        print(f"Female positive rate: {female_positive_rate:.4f}")
        print(f"Disparate impact: {disparate_impact:.4f}")
    
    # Print comparison table
    print("\n=== Model Comparison ===")
    print(f"{'Model':<30} {'Test Acc':<10} {'Test F1':<10} {'Test AUC':<10} {'Disp. Impact':<15}")
    print("-" * 75)
    for name, metrics in results.items():
        auc_str = f"{metrics['test_auc']:.4f}" if metrics['test_auc'] else "N/A"
        print(f"{name:<30} {metrics['test_accuracy']:<10.4f} {metrics['test_f1']:<10.4f} {auc_str:<10} {metrics['disparate_impact']:<15.4f}")
    
    return results


if __name__ == "__main__":
    # Run comparison on COMPAS data
    compare_models_on_compas()