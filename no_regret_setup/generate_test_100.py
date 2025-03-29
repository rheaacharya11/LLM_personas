import pandas as pd
import json

def create_compas_subset(json_file_path, parquet_file_path, output_file_path):
    """
    Create a subset of the compas_train.parquet file based on individual IDs from a JSON file.
    
    Parameters:
    -----------
    json_file_path : str
        Path to the JSON file containing comparison data
    parquet_file_path : str
        Path to the compas_train.parquet file
    output_file_path : str
        Path to save the output parquet file
    max_comparison_id : int, default=99
        Maximum comparison_id to include (0 to max_comparison_id inclusive)
    """
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        comparisons = json.load(f)
    
    # Filter comparisons to include only those with comparison_id from 0 to max_comparison_id
    
    # Extract all individual IDs and indices
    individual_ids = set()
    individual_indices = set()
    
    for comp in comparisons:
        individual_ids.add(comp['individual1_id'])
        individual_ids.add(comp['individual2_id'])
        individual_indices.add(comp['individual1_index'])
        individual_indices.add(comp['individual2_index'])
    
    # Read the parquet file
    df = pd.read_parquet(parquet_file_path)
    
    # Create a subset based on individual IDs or indices
    # Assuming the columns are named 'id' and 'index' - adjust if they're named differently
    id_column = 'id'  # Replace with actual column name if different
    index_column = 'index'  # Replace with actual column name if different
    
    # Try to filter by ID first
    if id_column in df.columns:
        subset_df = df[df[id_column].isin(individual_ids)]
        print("hi")
    # If ID column doesn't exist or subset is empty, try filtering by index
    elif index_column in df.columns or len(subset_df) == 0:
        subset_df = df[df[index_column].isin(individual_indices)]
    
    # Save the subset to a new parquet file
    subset_df.to_parquet(output_file_path)
    
    print(f"Created subset with {len(subset_df)} rows out of {len(df)} total rows")
    print(f"Saved to {output_file_path}")

if __name__ == "__main__":
    # Example usage
    json_file_path = "fixed_comparisons/final_fixed_comparisons.json"  # Replace with your JSON file path
    parquet_file_path = "data/processed/compas_train.parquet"  # Replace with your parquet file path
    output_file_path = "train200_subset.parquet"  # Output file path
    
    create_compas_subset(json_file_path, parquet_file_path, output_file_path)