import pandas as pd
import sys
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file_path>")
        sys.exit(1)

    csv_file_path = sys.argv[1]

    output_directory = "../../constraint_sets/lenient/no_personas_binary/"
    os.makedirs(output_directory, exist_ok=True)

    try:
        judgments_df = pd.read_csv(csv_file_path)
        print(f"Loaded fairness judgments: {len(judgments_df)} rows")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    try:
        train_df = pd.read_parquet('../../data/processed/compas_train.parquet')
        print(f"Loaded training COMPAS dataset: {len(train_df)} rows")
    except Exception as e:
        print(f"Error loading training COMPAS dataset: {e}")
        sys.exit(1)

    train_ids = set(train_df['id'].astype(int))
    train_id_to_index = {int(id_val): i for i, id_val in enumerate(train_df['id'])}
    
    judgment_ids = set()
    for i in range(0, len(judgments_df), 2):
        if i+1 < len(judgments_df):
            row1 = judgments_df.iloc[i]
            row2 = judgments_df.iloc[i+1]
            judgment_ids.add(int(row1['individual1_id']))
            judgment_ids.add(int(row1['individual2_id']))
    
    print(f"Fairness judgments contain {len(judgment_ids)} unique IDs")
    
    missing_from_train = judgment_ids - train_ids
    print(f"{len(missing_from_train)} IDs from judgments are not in the training set")
    if missing_from_train and len(missing_from_train) < 20:
        print(f"Missing IDs: {sorted(list(missing_from_train))}")
    
    valid_judgments = []
    for i in range(0, len(judgments_df), 2):
        if i+1 < len(judgments_df):
            row1 = judgments_df.iloc[i]
            row2 = judgments_df.iloc[i+1]
            id1 = int(row1['individual1_id'])
            id2 = int(row1['individual2_id'])
            
            if id1 in train_ids and id2 in train_ids:
                valid_judgments.append((i, i+1))
    
    print(f"Found {len(valid_judgments)} valid judgment pairs (both IDs in training set)")
    
    if len(valid_judgments) < 10:
        print("\nWARNING: Very few valid judgment pairs found!")

    # Process and store the judgments as constraint sets
    constraint_sets = {}

    for i, j in valid_judgments:
        row1 = judgments_df.iloc[i]
        row2 = judgments_df.iloc[j]
        id1 = int(row1['individual1_id'])
        id2 = int(row1['individual2_id'])
        
        if id1 in train_ids and id2 in train_ids:
            idx1 = train_id_to_index[id1]
            idx2 = train_id_to_index[id2]
            
            persona_id = int(row1['persona_id'])
            if persona_id not in constraint_sets:
                constraint_sets[persona_id] = []

            constraint_sets[persona_id].append({
                "pair": [idx1, idx2],
                "judgment1": row1['judgment'],
                "judgment2": row2['judgment']
            })

    # Save the constraint sets to a JSON file
    json_file_path = os.path.join(output_directory, 'all_judgments.json')
    with open(json_file_path, 'w') as file:
        json.dump(constraint_sets, file, indent=4)

    print(f"Constraint sets saved to {json_file_path}")

    # Optional: Implement visualization functions or additional analysis if needed.

if __name__ == "__main__":
    main()
