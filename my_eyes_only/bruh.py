'''
import pandas as pd

# Load the Parquet file
df = pd.read_parquet("train100_subset.parquet")

# Count unique individual_ids
unique_ids = df["id"].nunique()

print(f"Number of unique individual_id values: {unique_ids}")

import json

# Load the JSON
with open("multi_persona_data/train_persona_judgments.json", "r") as f:
    persona_data = json.load(f)

# Calculate number of pairs per persona
num_pairs_list = [len(pairs) for pairs in persona_data.values()]

# Compute the average
average_pairs = sum(num_pairs_list) / len(num_pairs_list)

print(f"Average number of pairs per persona: {average_pairs:.2f}")

import pandas as pd

# Load the Parquet file
df = pd.read_parquet('data/adult_personas.parquet')

# Print the first 5 rows
print(df.head())
'''
import pandas as pd
import re

# Load Parquet
df = pd.read_parquet('data/adult_personas.parquet')

# Extract race values using regex
race_values = []
for persona_str in df['persona']:
    match = re.search(r"political views: (.+)", persona_str)
    if match:
        race_values.append(match.group(1).strip())

# Get unique values
unique_races = set(race_values)

# Print
print("Unique race values:")
for race in sorted(unique_races):
    print(race)
