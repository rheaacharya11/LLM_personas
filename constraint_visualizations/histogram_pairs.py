import json
import pyarrow.parquet as pq
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

# Load the Parquet file and extract ages
df = pq.read_table('../data/unique_personas.parquet').to_pandas()
df['age'] = df['persona'].apply(lambda x: int(re.search(r"age: (\d+)", x).group(1)) if re.search(r"age: (\d+)", x) else None)

# Filter indices where age is greater than or equal to 18
valid_personas = df[df['age'] >= 18].index.tolist()  # Assuming the index in DataFrame matches persona IDs

# Load JSON data
with open('../constraint_sets/lenient/binary_personas/constraining_people.json', 'r') as file:
    data = json.load(file)

# Initialize a dictionary to count the number of personas per pair
pair_persona_count = {}

# Iterate through each pair in the JSON data
for pair, content in data.items():
    # Initialize the count for this pair
    pair_count = 0
    # Go through each persona in the pair
    for persona in content['personas']:
        # Check if the persona is valid (age >= 18)
        if persona in valid_personas:
            # Add the weight of this persona to the pair's count
            pair_count += content['weight']  # Assuming weight is uniformly applied to all in the list
    # Store the weighted count for the pair
    if pair_count > 0:
        pair_persona_count[pair] = pair_count

# Values for the histogram
counts = list(pair_persona_count.values())

# Calculate bin edges with a custom step size, ensure at least 10 bins if possible
max_count = 16  # Avoid errors if counts is empty
bins = range(1, int(max_count) + 2, max((int(max_count) // 10), 1))

mean_val = np.mean(counts)
median_val = np.median(counts)
percentile_25 = np.percentile(counts, 25)
percentile_75 = np.percentile(counts, 75)
# Creating the histogram
plt.figure(figsize=(10, 6))
plt.hist(counts, bins=bins, align='left', color='#88CCEE', edgecolor='black')
plt.title('Histogram of Similarity Weights (Binary Similarity, Personified LLM)')
plt.xlim(0, 16)
plt.ylim(0, 1600)
plt.xlabel('Cumulative Similarity Weight of Pair across all Judges')
plt.ylabel('Frequency of Weight')
plt.xticks(bins)  # Set x-ticks to the bin edges for clearer labeling
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid lines for better readability
plt.annotate(f'Mean: {mean_val:.2f}', xy=(0.70, 0.95), xycoords='axes fraction')
plt.annotate(f'Median: {median_val:.2f}', xy=(0.70, 0.90), xycoords='axes fraction')
plt.annotate('25th Percentile: {:.2f}'.format(percentile_25), xy=(0.70, 0.85), xycoords='axes fraction')
plt.annotate('75th Percentile: {:.2f}'.format(percentile_75), xy=(0.70, 0.80), xycoords='axes fraction')

plt.savefig('../constraint_sets/lenient/binary_personas/viz/histogram_pairs.png', dpi=300)

