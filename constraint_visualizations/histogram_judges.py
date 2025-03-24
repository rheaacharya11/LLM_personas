import pyarrow.parquet as pq
import pandas as pd
import re 
import numpy as np
# Function to extract age from persona string using regular expression
def extract_age(persona_str):
    match = re.search(r"age: (\d+)", persona_str)
    if match:
        return int(match.group(1))  # Return age as integer
    return None  # Return None if no age found

# Load the Parquet file
df = pq.read_table('../data/unique_personas.parquet').to_pandas()

# Apply the function to extract ages
df['age'] = df['persona'].apply(extract_age)

# Filter indices where age is greater than or equal to 18
indices = df[df['age'] >= 18].index + 1  # Add 1 to convert index to 1-based

# Print the indices
print(indices.tolist())
print(len(indices))

import json

# Load JSON data
with open('../constraint_sets/lenient/no_personas_binary/constraint_sets.json', 'r') as file:
    data = json.load(file)

# Filter pairs to include only those related to personas with age >= 18
filtered_pairs = {person_id: pairs for person_id, pairs in data.items() if int(person_id) in indices}
pairs_count = {person_id: len(pairs) for person_id, pairs in filtered_pairs.items()}


import matplotlib.pyplot as plt

# Values for the histogram
counts = list(pairs_count.values())

# Calculate bin edges with a custom step size
max_count = 50
bins = range(1, max_count + 2, max((max_count // 15), 1))  # Ensure at least 10 bins if possible

# Creating the histogram
plt.figure(figsize=(10, 5))  # Set figure size
plt.hist(counts, bins=bins, align='left', color='#DDCC77', edgecolor='black')
plt.title('Histogram of Constraint Pairs per Judge (Binary Similarity, Vanilla LLM)')
plt.xlabel('Number of Pairs selected by Judge (out of 50)')
plt.ylabel('Frequency')
# Setting the axis limits
plt.xlim(0, 50)  # Set the limits of the x-axis from 0 to 50
plt.ylim(0, 300)  # Set the limits of the y-axis from 0 to 300
plt.xticks(bins)  # Set x-ticks to the bin edges for clearer labeling
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid lines for better readability

mean_val = np.mean(counts)
median_val = np.median(counts)
percentile_25 = np.percentile(counts, 25)
percentile_75 = np.percentile(counts, 75)

plt.annotate(f'Mean: {mean_val:.2f}', xy=(0.75, 0.95), xycoords='axes fraction')
plt.annotate(f'Median: {median_val:.2f}', xy=(0.75, 0.90), xycoords='axes fraction')
plt.annotate('25th Percentile: {:.2f}'.format(percentile_25), xy=(0.75, 0.85), xycoords='axes fraction')
plt.annotate('75th Percentile: {:.2f}'.format(percentile_75), xy=(0.75, 0.80), xycoords='axes fraction')


plt.savefig('../constraint_sets/lenient/no_personas_binary/viz/histogram_18+.png', dpi=300)
