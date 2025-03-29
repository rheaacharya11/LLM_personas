# prepare_data.py
"""
This script prepares the COMPAS dataset and saves the cleaned data to files.
Run this once before launching any parallel jobs.
"""
import pandas as pd
import os
from persona_size.fixed_comparisons_study import load_compas_data, prepare_compas_data
from persona_size.generate_fixed_comparisons import generate_fixed_comparisons


# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

print("Loading and cleaning COMPAS dataset...")
compas_df = load_compas_data()
train_df, test_df = prepare_compas_data(compas_df, test_size=0.2, random_state=42)

# Save cleaned data to files
train_df.to_parquet("data/compas_train.parquet")
test_df.to_parquet("data/compas_test.parquet")
print(f"Saved cleaned train data ({len(train_df)} rows) to data/compas_train.parquet")
print(f"Saved cleaned test data ({len(test_df)} rows) to data/compas_test.parquet")

# Generate fixed comparisons from the cleaned training data
print("Generating fixed comparisons...")
generate_fixed_comparisons(train_df, num_comparisons=100, output_file="data/fixed_comparisons.json", random_state=42)

print("Data preparation complete!")