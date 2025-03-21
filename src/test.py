# test.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_parquet("../data/processed/compas_train.parquet")
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame columns: {df.columns.tolist()}")
print(f"Sample data:\n{df.head()}")

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns: {numeric_cols}")

# Setup target
if 'two_year_recid' in df.columns:
    y = df['two_year_recid'].values
    X_df = df.drop('two_year_recid', axis=1)
else:
    print("Column 'two_year_recid' not found")
    # Try last column
    y = df.iloc[:, -1].values  
    X_df = df.iloc[:, :-1]

# Remove categorical columns that cause issues
categorical_cols = [col for col in categorical_cols if col in X_df.columns]
numeric_cols = [col for col in numeric_cols if col in X_df.columns]

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create pipeline with preprocessing and classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Fit and evaluate
pipeline.fit(X_df, y)
preds = pipeline.predict(X_df)

print(f"Class distribution in target: {np.bincount(y)}")
print(f"Accuracy: {accuracy_score(y, preds):.4f}")
print(f"Prediction distribution: {np.bincount(preds)}")
print(f"Classification report:\n{classification_report(y, preds)}")