import pandas as pd
import glob
import os
import re

# Path to your CSV files
path = '../results/chunked_outputs/'  

all_files = glob.glob(os.path.join(path, "equal_run_*.csv"))
all_files.sort()

df_list = []

for filename in all_files:
    try:
        # Extract ID range from filename
        match = re.search(r'equal_run_(\d+)_to_(\d+)', filename)
        if not match:
            print(f"Warning: Couldn't extract ID range from {filename}, skipping")
            continue
            
        start_id = int(match.group(1))
        end_id = int(match.group(2))
        
        print(f"Processing {filename} (persona IDs: {start_id}-{end_id})")
        
        # Read the file
        df = pd.read_csv(filename, low_memory=False)
        
        # Filter rows where persona_id is within range
        filtered_df = df[(df['persona_id'] >= start_id) & (df['persona_id'] <= end_id)]
        
        if filtered_df.empty:
            print(f"Warning: No matching rows in {filename} after filtering by persona_id")
            continue
        
        # For each persona, select exactly one row for each comparison_number from 0-49
        result_dfs = []
        for persona_id, persona_group in filtered_df.groupby('persona_id'):
            # Create empty list to hold one row per comparison number
            comparison_rows = []
            
            # For each comparison number 0-49, select one row
            for comp_num in range(1, 51):
                comp_rows = persona_group[persona_group['comparison_number'] == comp_num]
                if not comp_rows.empty:
                    # Take the first row for this comparison number
                    comparison_rows.append(comp_rows.iloc[0:1])
            
            # Combine all selected rows for this persona
            if comparison_rows:
                persona_df = pd.concat(comparison_rows, ignore_index=True)
                result_dfs.append(persona_df)
        
        # Combine results for this file
        if result_dfs:
            file_df = pd.concat(result_dfs, ignore_index=True)
            file_df['source_file'] = os.path.basename(filename)
            df_list.append(file_df)
            
            # Count unique personas and comparison numbers
            unique_personas = file_df['persona_id'].nunique()
            unique_comparisons = file_df['comparison_number'].nunique()
            total_expected = unique_personas * 50  # If we had all 50 comparison numbers
            
            print(f"  Added {len(file_df)} rows from {unique_personas} personas")
            print(f"  Found {unique_comparisons}/50 unique comparison numbers")
            if len(file_df) < total_expected:
                print(f"  Note: Missing {total_expected - len(file_df)} expected combinations")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

if not df_list:
    print("Error: No files were successfully processed")
    exit(1)

print("\nCombining filtered dataframes...")
combined_df = pd.concat(df_list, ignore_index=True)

print(f"Combined dataset has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns")

# Verify we have the right comparison numbers
comp_counts = combined_df['comparison_number'].value_counts().sort_index()
print("\nComparison number distribution:")
print(comp_counts)

# Save the combined dataset
output_file = "combined_filtered_equal_runs.csv"
combined_df.to_csv(output_file, index=False)
print(f"\nFiltered dataset saved to {output_file}")

# Summary
unique_personas = combined_df['persona_id'].nunique()
print(f"\nSummary:")
print(f"- Total rows: {len(combined_df)}")
print(f"- Unique personas: {unique_personas}")
print(f"- Unique comparison numbers: {combined_df['comparison_number'].nunique()}")
print(f"- Files processed: {len(df_list)}")