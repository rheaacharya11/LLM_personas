import pandas as pd

# Read original parquet
full_df = pd.read_parquet('data/processed/compas.parquet')

# Read train and test splits
train_df = pd.read_parquet('data/processed/compas_train.parquet')
test_df = pd.read_parquet('data/processed/compas_test.parquet')

# Combine train and test
split_combined = pd.concat([train_df, test_df], ignore_index=True)

# Check if original and split-combined IDs are identical
original_ids = full_df['id'].tolist()
split_ids = split_combined['id'].tolist()

print("Original IDs match split combined IDs:", original_ids == split_ids)