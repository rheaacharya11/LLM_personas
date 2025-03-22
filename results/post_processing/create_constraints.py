import pandas as pd
import sys
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Check if the command line argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file_path>")
        sys.exit(1)

    csv_file_path = sys.argv[1]

    # Load the CSV file with the fairness judgments
    try:
        judgments_df = pd.read_csv(csv_file_path)
        print(f"Loaded fairness judgments: {len(judgments_df)} rows")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    # Load the training dataset
    try:
        train_df = pd.read_parquet('../../data/processed/compas_train.parquet')
        print(f"Loaded training COMPAS dataset: {len(train_df)} rows")
    except Exception as e:
        print(f"Error loading training COMPAS dataset: {e}")
        sys.exit(1)
    
    # Create a map from original IDs to training set indices
    train_ids = set(train_df['id'].astype(int))
    train_id_to_index = {int(id_val): i for i, id_val in enumerate(train_df['id'])}
    
    # Extract all unique IDs from fairness judgments
    judgment_ids = set()
    for i in range(0, len(judgments_df), 2):
        if i+1 < len(judgments_df):
            row1 = judgments_df.iloc[i]
            row2 = judgments_df.iloc[i+1]
            judgment_ids.add(int(row1['individual1_id']))
            judgment_ids.add(int(row2['individual2_id']))
    
    print(f"Fairness judgments contain {len(judgment_ids)} unique IDs")
    
    # Check which IDs are not in the training set
    missing_from_train = judgment_ids - train_ids
    print(f"{len(missing_from_train)} IDs from judgments are not in the training set")
    if missing_from_train and len(missing_from_train) < 20:
        print(f"Missing IDs: {sorted(list(missing_from_train))}")
    
    # Filter the fairness judgments to only include pairs where both IDs are in the training set
    valid_judgments = []
    for i in range(0, len(judgments_df), 2):
        if i+1 < len(judgments_df):
            row1 = judgments_df.iloc[i]
            row2 = judgments_df.iloc[i+1]
            id1 = int(row1['individual1_id'])
            id2 = int(row2['individual2_id'])
            
            if id1 in train_ids and id2 in train_ids:
                valid_judgments.append((i, i+1))
    
    print(f"Found {len(valid_judgments)} valid judgment pairs (both IDs in training set)")
    
    # If too few valid pairs, warn the user
    if len(valid_judgments) < 10:
        print("\nWARNING: Very few valid judgment pairs found!")
        print("This might not be enough for meaningful fairness constraints.")
    
    # Dictionaries to store the constraints and the constrained by data
    constraint_sets = {}     # Maps persona IDs to sets of constrained pairs
    constraining_people = {} # Maps constrained pairs to personas who chose them

    # Process fairness judgments
    skipped = 0
    processed = 0
    
    # Process only the valid judgment pairs
    for i, j in valid_judgments:
        row1 = judgments_df.iloc[i]
        row2 = judgments_df.iloc[j]

        # Check if both judgments are 'similar'
        if row1['judgment'] == 'similar' or row2['judgment'] == 'similar':
            # Get original IDs
            id1 = int(row1['individual1_id'])
            id2 = int(row1['individual2_id'])
            
            # Map to indices in the COMPAS training dataset
            try:
                idx1 = train_id_to_index[id1]
                idx2 = train_id_to_index[id2]
                
                # Create the pair using indices (not original IDs)
                pair = (idx1, idx2)
                
                # Add to constraint set of 'persona_id'
                persona_id = int(row1['persona_id'])
                if persona_id not in constraint_sets:
                    constraint_sets[persona_id] = set()
                constraint_sets[persona_id].add(pair)

                # Add 'persona_id' to constraining people of the pair
                if pair not in constraining_people:
                    constraining_people[pair] = set()
                constraining_people[pair].add(persona_id)
                
                processed += 1
            except KeyError:
                skipped += 1
                continue

    print(f"Processed {processed} constraint pairs, skipped {skipped} pairs due to ID mapping issues")

    # Convert sets to list for JSON serialization
    for key in constraint_sets:
        constraint_sets[key] = list(map(tuple, constraint_sets[key]))
    for key in constraining_people:
        constraining_people[key] = list(constraining_people[key])

    # Specify the directory path
    output_directory = "../../constraint_sets/lenient/binary_personas/"
    # Ensure the directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Save the dictionaries to JSON files
    save_to_json(constraint_sets, os.path.join(output_directory, 'constraint_sets.json'))
    save_to_json(constraining_people, os.path.join(output_directory, 'constraining_people.json'))

    # Create a file mapping original IDs to indices for reference
    id_mapping = {
        "id_to_index": {str(id_val): idx for id_val, idx in train_id_to_index.items()}
    }
    with open(os.path.join(output_directory, 'id_mapping.json'), 'w') as f:
        json.dump(id_mapping, f, indent=4)

    # Print summary statistics
    print(f"\nSummary:")
    print(f"  - Number of personas: {len(constraint_sets)}")
    print(f"  - Total pairs constrained: {len(constraining_people)}")
    
    # Calculate average constraints per persona
    constraints_per_persona = [len(pairs) for pairs in constraint_sets.values()]
    avg_constraints = np.mean(constraints_per_persona) if constraints_per_persona else 0
    print(f"  - Average constraints per persona: {avg_constraints:.2f}")
    
    # Calculate average personas per pair
    personas_per_pair = [len(personas) for personas in constraining_people.values()]
    avg_personas = np.mean(personas_per_pair) if personas_per_pair else 0
    print(f"  - Average personas per pair: {avg_personas:.2f}")

    # Create visualizations
    create_visualizations(constraint_sets, constraining_people, output_directory)

def create_visualizations(constraint_sets, constraining_people, output_directory):
    """Create and save visualizations for constraint analysis"""
    # Convert to DataFrame for easier manipulation
    persons_df = pd.DataFrame([(k, len(v)) for k, v in constraint_sets.items()], 
                              columns=['PersonID', 'NumPairs'])
    pairs_df = pd.DataFrame([(k, len(v)) for k, v in constraining_people.items()], 
                            columns=['Pair', 'NumPeople'])

    # Setting up directories for saving histograms
    viz_directory = os.path.join(output_directory, "viz")
    os.makedirs(viz_directory, exist_ok=True)

    # Creating histograms
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # Histogram for Number of Pairs Each Person Picks
    if not persons_df.empty:
        axes[0].hist(persons_df['NumPairs'], bins=range(1, persons_df['NumPairs'].max() + 2), 
                    color='skyblue', alpha=0.75)
    axes[0].set_title('Number of Pairs Each Person Picks')
    axes[0].set_xlabel('Number of Pairs')
    axes[0].set_ylabel('Frequency')

    # Histogram for Number of People Picking Each Pair
    if not pairs_df.empty:
        axes[1].hist(pairs_df['NumPeople'], bins=range(1, pairs_df['NumPeople'].max() + 2), 
                    color='lightgreen', alpha=0.75)
    axes[1].set_title('Number of People Picking Each Pair')
    axes[1].set_xlabel('Number of People')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()

    # Save the entire figure as a PNG file in the 'viz' directory
    fig.savefig(os.path.join(viz_directory, 'histograms.png'))
    print(f"Saved visualization to {os.path.join(viz_directory, 'histograms.png')}")

def save_to_json(data, file_path):
    """Utility function to save data to JSON file with custom key handling."""
    # Preprocess the dictionary to convert tuple keys to string
    if any(isinstance(key, tuple) for key in data):
        # Create a new dictionary with string keys if the key is a tuple
        data = {str(key): value for key, value in data.items()}
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save data to {file_path}. Error: {e}")

if __name__ == "__main__":
    main()