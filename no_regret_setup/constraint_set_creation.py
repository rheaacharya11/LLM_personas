import pandas as pd
import json
from collections import defaultdict

# Load the CSV
df = pd.read_csv("multi_persona_data/800_test_persona_judgments.csv")

# Prepare output structure
persona_data = defaultdict(list)

# Group by individual_id and then by comparison_id
for individual_id, group in df.groupby("persona_id"):
    for comp_id, comp_group in group.groupby("comparison_id"):
        if len(comp_group) != 2:
            continue  # skip incomplete pairs

        # Extract relevant info
        judgments = comp_group["judgment"].values
        orders = comp_group["order"].values
        ids1 = comp_group["individual1_id"].values
        ids2 = comp_group["individual2_id"].values

        # Get consistent ordering
        pair = [int(ids1[0]), int(ids2[0])]

        # Count how many judgments are "similar"
        similar_count = sum(j == "similar" for j in judgments)

        if similar_count == 0:
            continue  # skip this pair

        weight = 0.5 if similar_count == 1 else 1.0

        persona_data[str(individual_id)].append({
            "pair": pair,
            "weight": weight
        })

# Save to JSON
with open("multi_persona_data/final_holdout.json", "w") as f:
    json.dump(persona_data, f, indent=4)

print("JSON saved to multi_persona_data/final_test.json")
