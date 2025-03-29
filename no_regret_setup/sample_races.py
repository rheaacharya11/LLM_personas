import json
import random

# Load the JSON data from a file
with open("multi_persona_data/persona_metadata.json", "r") as f:
    data = json.load(f)

# Define the races to sample and how many IDs we need for each
races_to_sample = {
    "White": 10,
    "Asian": 10,
    "Black or African American": 10
}

sampled_ids = {}

# Iterate over each race and sample IDs
for race, count in races_to_sample.items():
    # Get a list of keys where the 'race' field matches the desired race
    matching_ids = [key for key, entry in data.items() if entry.get("race") == race]
    
    if len(matching_ids) < count:
        print(f"Warning: Only found {len(matching_ids)} entries for race '{race}', less than {count} required.")
        sampled_ids[race] = matching_ids
    else:
        sampled_ids[race] = random.sample(matching_ids, count)

# Print the results
print("Sampled IDs by race:")
for race, ids in sampled_ids.items():
    print(f"{race}: {ids}")
