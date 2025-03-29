import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import logging

def preprocess_data(df):
        """
        Preprocess dataframe to convert categorical columns to numeric
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Preprocessed dataframe with numeric columns
        """
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Identify column types
        categorical_columns = processed_df.select_dtypes(include=['object']).columns
        numeric_columns = processed_df.select_dtypes(include=['int64', 'float64']).columns
        
        # One-hot encode categorical columns
        processed_df = pd.get_dummies(processed_df, columns=categorical_columns)
        
        # Fill any remaining NaN values
        processed_df = processed_df.fillna(0)
        
        return processed_df


class FairnessDebugger:
    def __init__(self, data_path, constraint_sets_path):
        """
        Initialize the debugger with more robust logging and analysis
        """
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        
        self.logger = logging.getLogger(__name__)
        
        # Load data and constraints
        self.load_data(data_path)
        self.load_constraints(constraint_sets_path)
        
    # Modify load_data method
    def load_data(self, data_path):
        """Enhanced data loading with comprehensive preprocessing"""
        try:
            df = pd.read_parquet(data_path)
            
            # Detailed data summary
            self.logger.info(f"Original dataset shape: {df.shape}")
            self.logger.info("Original column types:\n" + str(df.dtypes))
            
            # Preprocess the data
            target_col = 'two_year_recid'
            processed_df = preprocess_data(df)
            
            # Detailed processed data summary
            self.logger.info(f"Processed dataset shape: {processed_df.shape}")
            self.logger.info("Processed column types:\n" + str(processed_df.dtypes))
            
            # Check target variable distribution
            class_distribution = df[target_col].value_counts(normalize=True)
            self.logger.info(f"Target variable distribution:\n{class_distribution}")
            
            # Separate features and target
            self.X = processed_df.drop(columns=[target_col]).values
            self.y = df[target_col].values
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            raise
        
    def load_constraints(self, constraint_sets_path):
            """
            Enhanced constraint loading with more validation
            """
            try:
                with open(constraint_sets_path, 'r') as f:
                    constraint_data = json.load(f)
                
                # Analyze constraint diversity
                total_constraints = sum(len(constraints) for constraints in constraint_data.values())
                num_judges = len(constraint_data)
                
                self.logger.info(f"Total judges: {num_judges}")
                self.logger.info(f"Total constraint pairs: {total_constraints}")
                
                # Basic constraint validation
                valid_constraints = [
                    pair for judge_constraints in constraint_data.values() 
                    for pair in judge_constraints 
                    if len(pair) == 2 and pair[0] < len(self.X) and pair[1] < len(self.X)
                ]
                
                self.logger.info(f"Valid constraint pairs: {len(valid_constraints)}")
                
                # Store constraints
                self.constraints = valid_constraints
                
            except Exception as e:
                self.logger.error(f"Constraint loading failed: {e}")
                raise   
    
    def validate_constraints(self):
        """
        Deep dive into constraint characteristics with type conversion
        """
        # Ensure X is numeric
        X_numeric = self.X.astype(float)
        
        feature_diffs = []
        target_diffs = []
        
        for pair in self.constraints:
            i, j = pair
            try:
                # Feature difference
                feature_diff = np.abs(X_numeric[i] - X_numeric[j])
                feature_diffs.append(feature_diff.mean())
                
                # Target difference
                target_diffs.append(abs(self.y[i] - self.y[j]))
            except Exception as e:
                print(f"Error processing pair {pair}: {e}")
                print(f"X[{i}] = {self.X[i]}")
                print(f"X[{j}] = {self.X[j]}")
        
        self.logger.info("Constraint Feature Analysis:")
        self.logger.info(f"Mean feature difference: {np.mean(feature_diffs):.4f}")
        self.logger.info(f"Std feature difference: {np.std(feature_diffs):.4f}")
        
        self.logger.info("Constraint Target Analysis:")
        self.logger.info(f"Proportion of constraints with different targets: {np.mean(target_diffs):.4f}")
    def baseline_model_analysis(self):
        """
        Simple baseline model with comprehensive analysis
        """
        # Basic logistic regression
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X, self.y)
        
        # Predictions
        y_pred = model.predict(self.X)
        
        # Comprehensive performance metrics
        self.logger.info("Baseline Model Performance:")
        self.logger.info(f"Accuracy: {accuracy_score(self.y, y_pred):.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y, y_pred)
        self.logger.info("Confusion Matrix:\n" + str(cm))
        
        # Feature importance
        feature_importance = np.abs(model.coef_[0])
        top_features = np.argsort(feature_importance)[-5:]
        
        self.logger.info("Top 5 most important features:")
        for idx in top_features[::-1]:
            self.logger.info(f"Feature {idx}: {feature_importance[idx]:.4f}")
    
    def run_diagnostics(self):
        """
        Comprehensive diagnostics pipeline
        """
        self.validate_constraints()
        self.baseline_model_analysis()

# Diagnostic script
def main():
    debugger = FairnessDebugger(
        data_path='data/processed/compas_train.parquet',
        constraint_sets_path='constraint_sets/binary_personas/constraint_sets.json'
    )
    debugger.run_diagnostics()

if __name__ == "__main__":
    main()