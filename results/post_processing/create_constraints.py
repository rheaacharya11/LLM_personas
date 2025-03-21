import pandas as pd
import sys
import json
import os
import matplotlib.pyplot as plt

def main():
    # Check if the command line argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file_path>")
        sys.exit(1)

    csv_file_path = sys.argv[1]

    # Load the CSV file
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    # Dictionaries to store the constraints and the constrained by data
    constraint_sets = {}
    constraining_people = {}

    # Iterate through the DataFrame by two rows each time
    for i in range(0, len(df), 2):
        if i+1 < len(df):  # Check if there's a pair to process
            row1 = df.iloc[i]
            row2 = df.iloc[i+1]

            # Check if both judgments are 'similar'
            if row1['judgment'] == 'similar' or row2['judgment'] == 'similar':
                pair = (int(row1['individual1_id']), int(row2['individual2_id']))
                
                # Add to constraint set of 'persona_id'
                persona_id = int(row1['persona_id'])
                if persona_id not in constraint_sets:
                    constraint_sets[persona_id] = set()
                constraint_sets[persona_id].add(pair)

                # Add 'persona_id' to constraining people of the pair
                if pair not in constraining_people:
                    constraining_people[pair] = set()
                constraining_people[pair].add(persona_id)

    # Convert sets to list for JSON serialization
    for key in constraint_sets:
        constraint_sets[key] = list(constraint_sets[key])
    for key in constraining_people:
        constraining_people[key] = list(constraining_people[key])

    # Specify the directory path
    output_directory = "../../constraint_sets/lenient/binary_personas/"
    # Ensure the directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Save the dictionaries to JSON files in the specified directory
    save_to_json(constraint_sets, os.path.join(output_directory, 'constraint_sets.json'))
    save_to_json(constraining_people, os.path.join(output_directory, 'constraining_people.json'))


    # Convert to DataFrame for easier manipulation
    persons_df = pd.DataFrame([(k, len(v)) for k, v in constraint_sets.items()], columns=['PersonID', 'NumPairs'])
    pairs_df = pd.DataFrame([(k, len(v)) for k, v in constraining_people.items()], columns=['Pair', 'NumPeople'])

    # Setting up directories for saving histograms
    viz_directory = os.path.join(output_directory, "viz")
    os.makedirs(viz_directory, exist_ok=True)

    # Add this code before the histograms section
    print(f"Total number of people: {len(constraint_sets)}")
    print(f"Total number of unique pairs selected: {len(constraining_people)}")

    # Count total pairs selected across all personas
    total_pairs_selected = sum(len(pairs) for pairs in constraint_sets.values())
    print(f"Total pairs selected: {total_pairs_selected}")
    print(f"Average pairs selected per person: {total_pairs_selected/len(constraint_sets):.2f}")

    # Distribution summary for blue histogram (pairs per person)
    pairs_per_person = [len(pairs) for pairs in constraint_sets.values()]
    print(f"Min pairs selected by a person: {min(pairs_per_person)}")
    print(f"Max pairs selected by a person: {max(pairs_per_person)}")

    # Distribution summary for green histogram (people per pair)
    people_per_pair = [len(people) for people in constraining_people.values()]
    print(f"Min people selecting a pair: {min(people_per_pair)}")
    print(f"Max people selecting a pair: {max(people_per_pair)}")
    print(f"Most common: {max(set(people_per_pair), key=people_per_pair.count)} people selecting a pair")
        # Creating histograms
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # Histogram for Number of Pairs Each Person Picks
    axes[0].hist(persons_df['NumPairs'], bins=range(1, persons_df['NumPairs'].max() + 2), color='skyblue', alpha=0.75)
    axes[0].set_title('Number of Pairs Each Person Picks')
    axes[0].set_xlabel('Number of Pairs')
    axes[0].set_ylabel('Frequency')

    # Histogram for Number of People Picking Each Pair
    axes[1].hist(pairs_df['NumPeople'], bins=range(1, pairs_df['NumPeople'].max() + 2), color='lightgreen', alpha=0.75)
    axes[1].set_title('Number of People Picking Each Pair')
    axes[1].set_xlabel('Number of People')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()

    # Save the entire figure as a PNG file in the 'viz' directory
    fig.savefig(os.path.join(viz_directory, 'histograms.png'))

    # Optionally, save individual plots if needed
    # axes[0].get_figure().savefig(os.path.join(viz_directory, 'persons_picking_pairs_histogram.png'))
    # axes[1].get_figure().savefig(os.path.join(viz_directory, 'people_picking_each_pair_histogram.png'))


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

    

# Main execution logic remains similar, ensure the conversion logic is applied during serialization


if __name__ == "__main__":
    main()
    