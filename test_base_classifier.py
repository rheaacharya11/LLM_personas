#!/usr/bin/env python
"""
Test script to verify classifiers work with the training data.
Usage: python test_base_classifier.py [data_path]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

def load_data(filepath):
    """Load data with preprocessing for categorical features."""
    df = pd.read_parquet(filepath)
    
    # Assume 'two_year_recid' is the target variable
    if 'two_year_recid' in df.columns:
        y = df['two_year_recid'].values
        X_df = df.drop(columns=['two_year_recid'])
    else:
        # If column names are different, assume last column is target
        y = df.iloc[:, -1].values
        X_df = df.iloc[:, :-1]
    
    # Remove ID column if it exists
    if 'id' in X_df.columns:
        X_df = X_df.drop(columns=['id'])
    
    # Handle categorical features
    categorical_columns = X_df.select_dtypes(include=['object', 'category']).columns
    numerical_columns = X_df.select_dtypes(include=['number']).columns
    
    # Apply one-hot encoding to categorical columns
    if not categorical_columns.empty:
        print(f"Categorical columns: {categorical_columns.tolist()}")
        X_categorical = pd.get_dummies(X_df[categorical_columns], drop_first=True)
        X_numerical = X_df[numerical_columns]
        X_processed = pd.concat([X_numerical, X_categorical], axis=1)
    else:
        X_processed = X_df
    
    X = X_processed.values
    
    print(f"Data shape: {X.shape} features, {len(y)} samples")
    return X, y

def test_classifiers(X, y):
    """Test multiple classifiers on the dataset."""
    # Print class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution: {dict(zip(unique_classes, counts))}")
    
    # Test different classifiers
    classifiers = {
        "LogisticRegression (default)": LogisticRegression(max_iter=1000),
        "LogisticRegression (balanced)": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "RandomForest (default)": RandomForestClassifier(n_estimators=100),
        "RandomForest (balanced)": RandomForestClassifier(n_estimators=100, class_weight='balanced')
    }
    
    for name, clf in classifiers.items():
        print(f"\n{name}:")
        
        # Basic fitting
        clf.fit(X, y)
        train_acc = clf.score(X, y)
        
        # Predictions
        preds = clf.predict(X)
        pred_proba = clf.predict_proba(X)[:,1] if hasattr(clf, "predict_proba") else None
        
        # Predicted class distribution
        pred_unique, pred_counts = np.unique(preds, return_counts=True)
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Predicted class distribution: {dict(zip(pred_unique, pred_counts))}")
        
        # Probability distribution (for models that support it)
        if pred_proba is not None:
            print(f"Prediction probabilities: min={pred_proba.min():.4f}, max={pred_proba.max():.4f}, "
                  f"unique values={len(np.unique(pred_proba))}")
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y, preds))
        
        # Cross-validation (optional, can be slow on large datasets)
        try:
            cv_scores = cross_val_score(clf, X, y, cv=3)
            print(f"Cross-validation score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        except Exception as e:
            print(f"Cross-validation failed: {e}")


def test_improved_classifiers(X, y):
    """Test enhanced classifiers with hyperparameter tuning and cross-validation."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingClassifier
    
    print("\n=== IMPROVED CLASSIFIERS ===")
    
    # 1. Feature Engineering: Add polynomial features
    print("\nStep 1: Adding polynomial features...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    print(f"Original features: {X.shape[1]}, With interactions: {X_poly.shape[1]}")
    
    # 2. Improved LogisticRegression with grid search
    print("\nStep 2: Tuning LogisticRegression...")
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create pipeline with scaling
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    # Grid search for best parameters
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__class_weight': [None, 'balanced']
    }
    
    # Run grid search (can be time consuming)
    grid_search = GridSearchCV(lr_pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    grid_search.fit(X, y)
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")
    
    # Get predictions with best model
    best_lr = grid_search.best_estimator_
    y_pred = best_lr.predict(X)
    print("\nBest LogisticRegression Classification Report:")
    print(classification_report(y, y_pred))
    
    # 3. Gradient Boosting Classifier
    print("\nStep 3: Training Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    # Use cross-validation to evaluate
    gb_scores = cross_val_score(gb, X, y, cv=cv, scoring='f1')
    print(f"Gradient Boosting CV F1 score: {gb_scores.mean():.4f} ± {gb_scores.std():.4f}")
    
    # Fit the model and evaluate
    gb.fit(X, y)
    y_pred_gb = gb.predict(X)
    print("\nGradient Boosting Classification Report:")
    print(classification_report(y, y_pred_gb))
    
    # 4. Feature importance analysis
    print("\nStep 4: Feature Importance Analysis...")
    feature_importance = gb.feature_importances_
    
    # Get top 10 features
    indices = np.argsort(feature_importance)[::-1]
    top_n = min(10, len(indices))
    print(f"Top {top_n} important features:")
    for i in range(top_n):
        print(f"Feature {indices[i]}: {feature_importance[indices[i]]:.4f}")
    
    return best_lr, gb

def test_advanced_classifiers(X, y):
    """Advanced classification with feature selection and XGBoost."""
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
    from sklearn.feature_selection import SelectFromModel
    from sklearn.metrics import classification_report, roc_auc_score
    import numpy as np
    
    print("\n=== ADVANCED CLASSIFIERS WITH FEATURE SELECTION ===")
    
    # Split data for proper evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 1. Feature selection based on importance
    print("\nStep 1: Feature selection using Gradient Boosting...")
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Train a GB model for feature selection
    selector = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                         max_depth=3, random_state=42)
    selector.fit(X_train, y_train)
    
    # Select important features
    sfm = SelectFromModel(selector, threshold="mean")
    sfm.fit(X_train, y_train)
    
    # Transform the data
    X_train_selected = sfm.transform(X_train)
    X_test_selected = sfm.transform(X_test)
    
    # Show selected features
    selected_indices = np.where(sfm.get_support())[0]
    print(f"Selected {len(selected_indices)} out of {X.shape[1]} features")
    print(f"Selected feature indices: {selected_indices}")
    
    # 2. Train XGBoost on selected features
    print("\nStep 2: Training XGBoost on selected features...")
    try:
        import xgboost as xgb
        
        # Create and train XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            objective='binary:logistic',
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),  # Balance classes
            random_state=42
        )
        
        xgb_model.fit(
            X_train_selected, y_train,
            eval_set=[(X_test_selected, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate on test set
        y_pred = xgb_model.predict(X_test_selected)
        y_prob = xgb_model.predict_proba(X_test_selected)[:, 1]
        
        print("\nXGBoost Classification Report (Test Set):")
        print(classification_report(y_test, y_pred))
        
        # Calculate ROC AUC
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC Score: {auc:.4f}")
        
        # 3. Feature importance from XGBoost
        print("\nStep 3: XGBoost Feature Importance...")
        feature_importance = xgb_model.feature_importances_
        for i, importance in enumerate(feature_importance):
            # Map back to original feature index
            orig_idx = selected_indices[i]
            print(f"Selected Feature {orig_idx}: {importance:.4f}")
            
        return xgb_model, sfm
        
    except ImportError:
        print("XGBoost not installed. Using Gradient Boosting instead.")
        
        # Alternative: Use Gradient Boosting
        from sklearn.ensemble import GradientBoostingClassifier
        
        # Create and train Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        gb_model.fit(X_train_selected, y_train)
        
        # Evaluate on test set
        y_pred = gb_model.predict(X_test_selected)
        y_prob = gb_model.predict_proba(X_test_selected)[:, 1]
        
        print("\nGradient Boosting Classification Report (Test Set):")
        print(classification_report(y_test, y_pred))
        
        # Calculate ROC AUC
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC Score: {auc:.4f}")
        
        # Feature importance
        print("\nFeature Importance Analysis...")
        feature_importance = gb_model.feature_importances_
        for i, importance in enumerate(feature_importance):
            # Map back to original feature index
            orig_idx = selected_indices[i]
            print(f"Selected Feature {orig_idx}: {importance:.4f}")
            
        return gb_model, sfm

def test_with_external_data(train_path, test_path):
    """Test classifiers using separate training and test data."""
    import pandas as pd
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_selection import SelectFromModel
    
    print("\n=== EVALUATION WITH EXTERNAL TEST DATA ===")
    
    # Load training data
    X_train, y_train = load_data(train_path)
    
    # Load test data
    X_test, y_test = load_data(test_path)
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Feature selection
    print("\nPerforming feature selection...")
    selector = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    selector.fit(X_train, y_train)
    
    # Get important features
    sfm = SelectFromModel(selector, threshold="mean")
    sfm.fit(X_train, y_train)
    
    # Apply selection
    X_train_selected = sfm.transform(X_train)
    X_test_selected = sfm.transform(X_test)
    
    selected_indices = np.where(sfm.get_support())[0]
    print(f"Selected {len(selected_indices)} out of {X_train.shape[1]} features")
    print(f"Selected feature indices: {selected_indices}")
    
    # Try XGBoost first
    try:
        import xgboost as xgb
        
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            objective='binary:logistic',
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
            random_state=42
        )
        
        xgb_model.fit(
            X_train_selected, y_train,
            eval_set=[(X_test_selected, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        y_pred = xgb_model.predict(X_test_selected)
        y_prob = xgb_model.predict_proba(X_test_selected)[:, 1]
        
        print("\nXGBoost Performance on Test Data:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
        
        best_model = xgb_model
    
    except ImportError:
        print("\nXGBoost not available, using Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        gb_model.fit(X_train_selected, y_train)
        
        y_pred = gb_model.predict(X_test_selected)
        y_prob = gb_model.predict_proba(X_test_selected)[:, 1]
        
        print("\nGradient Boosting Performance on Test Data:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
        
        best_model = gb_model
    
    return best_model, sfm

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_base_classifier.py [train_data_path] (optional: [test_data_path])")
        sys.exit(1)
    
    train_path = sys.argv[1]
    
    # Basic classifier tests on training data
    X, y = load_data(train_path)
    test_classifiers(X, y)
    
    # If test data provided, use it
    if len(sys.argv) > 2:
        test_path = sys.argv[2]
        best_model, selector = test_with_external_data(train_path, test_path)
    else:
        # Run improved classifiers on training data only
        test_improved_classifiers(X, y)

if __name__ == "__main__":
    main()