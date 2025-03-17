import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from csc_oracle import CostSensitiveClassificationOracle, ImprovedCostSensitiveOracle, compare_models_on_compas

# Quick comparison of all models
results = compare_models_on_compas()

def load_compas_data_from_parquet(train_path="data/compas_train.parquet", 
                             test_path="data/compas_test.parquet"):
    """
    Load COMPAS dataset from parquet files.
    
    Args:
        train_path: Path to training data parquet file
        test_path: Path to test data parquet file
        
    Returns:
        train_df, test_df: DataFrames with COMPAS data
    """
    try:
        train_df = pd.read_parquet(train_path)
        print(f"Loaded training data from {train_path}: {len(train_df)} rows")
        
        test_df = pd.read_parquet(test_path)
        print(f"Loaded test data from {test_path}: {len(test_df)} rows")
        
        return train_df, test_df
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        print("Falling back to loading from URL...")
        
        # Fallback to loading from URL
        import requests
        from io import StringIO
        from sklearn.model_selection import train_test_split
        
        url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text))
        
        # Split into train/test sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        return train_df, test_df

def prepare_compas_features(df):
    """
    Prepare COMPAS dataset for classification.
    
    Args:
        df: COMPAS dataframe
        
    Returns:
        X: Feature matrix
        y: Target labels (recidivism)
    """
    # The parquet files may already have processed features
    # Check the columns to determine if preprocessing is needed
    
    if 'features' in df.columns and 'two_year_recid' in df.columns:
        # Dataset already has a features column
        print("Using pre-existing features column")
        X = np.array(df['features'].tolist())
        y = df['two_year_recid'].values
        
        # Try to get feature names if available
        if 'feature_names' in df.columns:
            feature_names = df['feature_names'].iloc[0]
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        return X, y, feature_names
    
    # If not preprocessed, do standard processing
    print("Processing raw features...")
    
    # Select relevant features
    features = ['age', 'sex', 'race', 'priors_count', 'c_charge_degree', 
                'juv_fel_count', 'juv_misd_count', 'juv_other_count']
    
    # Copy the dataframe to avoid modifying the original
    df_proc = df.copy()
    
    # Filter out rows with missing values
    df_proc = df_proc.dropna(subset=features + ['two_year_recid'])
    
    # Encode categorical variables
    df_proc['sex'] = df_proc['sex'].map({'Female': 0, 'Male': 1})
    df_proc['c_charge_degree'] = df_proc['c_charge_degree'].map({'F': 1, 'M': 0})
    
    # One-hot encode race
    race_dummies = pd.get_dummies(df_proc['race'], prefix='race')
    df_proc = pd.concat([df_proc, race_dummies], axis=1)
    
    # Drop original race column
    df_proc.drop('race', axis=1, inplace=True)
    
    # Get features and target
    features_processed = ['age', 'sex', 'priors_count', 'c_charge_degree', 
                         'juv_fel_count', 'juv_misd_count', 'juv_other_count'] + list(race_dummies.columns)
    X = df_proc[features_processed].values
    y = df_proc['two_year_recid'].values
    
    return X, y, features_processed

def test_csc_oracle_basic():
    """Test the CSC oracle on synthetic data."""
    print("Testing CSC oracle on synthetic data...")
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 5)  # 100 samples, 5 features
    
    # Create a simple linear decision boundary
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create uniform costs
    costs = [(1-y_i, y_i) for y_i in y]  # Cost 1 for misclassification, 0 for correct classification
    
    # Create a non-uniform cost structure
    importance = np.random.rand(100)  # Random importance weights
    weighted_costs = [((1-y_i) * importance[i], y_i * importance[i]) for i, y_i in enumerate(y)]
    
    # Initialize and test the oracle
    oracle = CostSensitiveClassificationOracle()
    
    # Test with uniform costs
    uniform_preds = oracle(X, costs)
    uniform_acc = accuracy_score(y, uniform_preds)
    
    # Test with weighted costs
    weighted_preds = oracle(X, weighted_costs)
    weighted_acc = accuracy_score(y, weighted_preds)
    
    print(f"Accuracy with uniform costs: {uniform_acc:.4f}")
    print(f"Accuracy with weighted costs: {weighted_acc:.4f}")
    
    return uniform_acc, weighted_acc

def test_csc_oracle_compas():
    """Test the CSC oracle on the COMPAS dataset."""
    print("\nTesting CSC oracle on COMPAS dataset...")
    
    # Load and prepare COMPAS data from parquet files
    train_df, test_df = load_compas_data_from_parquet()
    
    # Process features
    X_train, y_train, train_feature_names = prepare_compas_features(train_df)
    X_test, y_test, test_feature_names = prepare_compas_features(test_df)
    
    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing data: {X_test.shape[0]} samples")
    
    # Create uniform costs for training data
    costs_train = [(1-y_i, y_i) for y_i in y_train]
    
    # Initialize and train the oracle
    oracle = CostSensitiveClassificationOracle()
    train_preds = oracle(X_train, costs_train)
    train_acc = accuracy_score(y_train, train_preds)
    
    # Test on held-out data
    test_preds = oracle.model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds))
    
    return train_acc, test_acc, oracle.model

def create_imbalanced_costs(y, protected_attribute, bias_direction=1.0, bias_factor=2.0):
    """
    Create imbalanced costs based on a protected attribute.
    This simulates a situation where classification errors for one group
    are considered more costly than for another group.
    
    Args:
        y: True labels
        protected_attribute: Binary protected attribute (0 or 1)
        bias_direction: 1.0 to favor attribute=1, -1.0 to favor attribute=0
        bias_factor: How much more costly errors are for the favored group
        
    Returns:
        List of (c0, c1) cost pairs
    """
    costs = []
    
    for i in range(len(y)):
        # Base misclassification costs
        misclass_cost = 1.0
        
        # Adjust costs based on protected attribute
        if (protected_attribute[i] == 1 and bias_direction > 0) or \
           (protected_attribute[i] == 0 and bias_direction < 0):
            misclass_cost *= bias_factor
        
        # c0: cost of classifying as 0, c1: cost of classifying as 1
        c0 = misclass_cost if y[i] == 1 else 0.0  # Cost of false negative
        c1 = misclass_cost if y[i] == 0 else 0.0  # Cost of false positive
        
        costs.append((c0, c1))
    
    return costs

def test_csc_oracle_with_demographic_bias():
    """Test the CSC oracle with costs that vary by demographic group."""
    print("\nTesting CSC oracle with demographically imbalanced costs...")
    
    # Load and prepare COMPAS data from parquet files
    train_df, test_df = load_compas_data_from_parquet()
    
    # Process features
    X_train, y_train, train_feature_names = prepare_compas_features(train_df)
    X_test, y_test, test_feature_names = prepare_compas_features(test_df)
    
    # Extract gender as protected attribute
    # Check if 'sex' is available in the dataframe
    if 'sex' in train_df.columns:
        gender_train = train_df['sex'].map({'Female': 0, 'Male': 1}).values
        gender_test = test_df['sex'].map({'Female': 0, 'Male': 1}).values
    else:
        # If not, look for it in features
        # This assumes sex/gender is the second feature (index 1)
        gender_train = X_train[:, 1]
        gender_test = X_test[:, 1]
    
    # Create three different cost structures
    # 1. Uniform costs (baseline)
    uniform_costs = [(1-y_i, y_i) for y_i in y_train]
    
    # 2. Higher costs for errors on women (gender=0)
    female_higher_costs = create_imbalanced_costs(y_train, gender_train, bias_direction=-1.0, bias_factor=3.0)
    
    # 3. Higher costs for errors on men (gender=1)
    male_higher_costs = create_imbalanced_costs(y_train, gender_train, bias_direction=1.0, bias_factor=3.0)
    
    # Train three different oracles
    uniform_oracle = CostSensitiveClassificationOracle()
    uniform_oracle(X_train, uniform_costs)
    
    female_biased_oracle = CostSensitiveClassificationOracle()
    female_biased_oracle(X_train, female_higher_costs)
    
    male_biased_oracle = CostSensitiveClassificationOracle()
    male_biased_oracle(X_train, male_higher_costs)
    
    # Get predictions for each
    uniform_preds = uniform_oracle.model.predict(X_test)
    female_biased_preds = female_biased_oracle.model.predict(X_test)
    male_biased_preds = male_biased_oracle.model.predict(X_test)
    
    # Analyze performance for different groups
    # 1. Overall accuracy
    print(f"Overall accuracy (uniform costs): {accuracy_score(y_test, uniform_preds):.4f}")
    print(f"Overall accuracy (female-favoring): {accuracy_score(y_test, female_biased_preds):.4f}")
    print(f"Overall accuracy (male-favoring): {accuracy_score(y_test, male_biased_preds):.4f}")
    
    # 2. Group-specific metrics
    female_idx = gender_test == 0
    male_idx = gender_test == 1
    
    # Female accuracy
    female_acc_uniform = accuracy_score(y_test[female_idx], uniform_preds[female_idx])
    female_acc_female_biased = accuracy_score(y_test[female_idx], female_biased_preds[female_idx])
    female_acc_male_biased = accuracy_score(y_test[female_idx], male_biased_preds[female_idx])
    
    # Male accuracy
    male_acc_uniform = accuracy_score(y_test[male_idx], uniform_preds[male_idx])
    male_acc_female_biased = accuracy_score(y_test[male_idx], female_biased_preds[male_idx])
    male_acc_male_biased = accuracy_score(y_test[male_idx], male_biased_preds[male_idx])
    
    print("\nAccuracy by gender:")
    print(f"Female accuracy (uniform costs): {female_acc_uniform:.4f}")
    print(f"Female accuracy (female-favoring): {female_acc_female_biased:.4f}")
    print(f"Female accuracy (male-favoring): {female_acc_male_biased:.4f}")
    
    print(f"Male accuracy (uniform costs): {male_acc_uniform:.4f}")
    print(f"Male accuracy (female-favoring): {male_acc_female_biased:.4f}")
    print(f"Male accuracy (male-favoring): {male_acc_male_biased:.4f}")
    
    # Calculate false positive rates by gender
    def calc_fpr(y_true, y_pred):
        """Calculate false positive rate"""
        fp = sum((y_pred == 1) & (y_true == 0))
        tn = sum((y_pred == 0) & (y_true == 0))
        return fp / (fp + tn) if (fp + tn) > 0 else 0
    
    female_fpr_uniform = calc_fpr(y_test[female_idx], uniform_preds[female_idx])
    female_fpr_female_biased = calc_fpr(y_test[female_idx], female_biased_preds[female_idx])
    female_fpr_male_biased = calc_fpr(y_test[female_idx], male_biased_preds[female_idx])
    
    male_fpr_uniform = calc_fpr(y_test[male_idx], uniform_preds[male_idx])
    male_fpr_female_biased = calc_fpr(y_test[male_idx], female_biased_preds[male_idx])
    male_fpr_male_biased = calc_fpr(y_test[male_idx], male_biased_preds[male_idx])
    
    print("\nFalse positive rate by gender:")
    print(f"Female FPR (uniform costs): {female_fpr_uniform:.4f}")
    print(f"Female FPR (female-favoring): {female_fpr_female_biased:.4f}")
    print(f"Female FPR (male-favoring): {female_fpr_male_biased:.4f}")
    
    print(f"Male FPR (uniform costs): {male_fpr_uniform:.4f}")
    print(f"Male FPR (female-favoring): {male_fpr_female_biased:.4f}")
    print(f"Male FPR (male-favoring): {male_fpr_male_biased:.4f}")
    
    # Compute FPR difference (a common fairness metric)
    fpr_diff_uniform = abs(female_fpr_uniform - male_fpr_uniform)
    fpr_diff_female_biased = abs(female_fpr_female_biased - male_fpr_female_biased)
    fpr_diff_male_biased = abs(female_fpr_male_biased - male_fpr_male_biased)
    
    print("\nFPR difference (|female_FPR - male_FPR|):")
    print(f"Uniform costs: {fpr_diff_uniform:.4f}")
    print(f"Female-favoring: {fpr_diff_female_biased:.4f}")
    print(f"Male-favoring: {fpr_diff_male_biased:.4f}")
    
    # Return key metrics
    return {
        'uniform': {
            'overall_acc': accuracy_score(y_test, uniform_preds),
            'female_acc': female_acc_uniform,
            'male_acc': male_acc_uniform,
            'female_fpr': female_fpr_uniform,
            'male_fpr': male_fpr_uniform,
            'fpr_diff': fpr_diff_uniform
        },
        'female_biased': {
            'overall_acc': accuracy_score(y_test, female_biased_preds),
            'female_acc': female_acc_female_biased,
            'male_acc': male_acc_female_biased,
            'female_fpr': female_fpr_female_biased,
            'male_fpr': male_fpr_female_biased,
            'fpr_diff': fpr_diff_female_biased
        },
        'male_biased': {
            'overall_acc': accuracy_score(y_test, male_biased_preds),
            'female_acc': female_acc_male_biased,
            'male_acc': male_acc_male_biased,
            'female_fpr': female_fpr_male_biased,
            'male_fpr': male_fpr_male_biased,
            'fpr_diff': fpr_diff_male_biased
        }
    }

if __name__ == "__main__":
    # Run basic test on synthetic data
    uniform_acc, weighted_acc = test_csc_oracle_basic()
    
    # Run test on COMPAS dataset
    train_acc, test_acc, model = test_csc_oracle_compas()
    
    # Run test with demographic bias in costs
    bias_metrics = test_csc_oracle_with_demographic_bias()
    
    print("\nAll tests completed!")