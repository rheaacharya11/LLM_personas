import pandas as pd
import re
import json

# Load the Parquet file
df = pd.read_parquet('data/unique_personas.parquet')  # change to your actual file path

# Race mapping for cleaner visualization
race_mapping = {
    "Asian": [
        "Asian", "Asian Indian", "Asian Indian alone", "Burmese alone",
        "Cambodian alone", "Chinese", "Chinese, except Taiwanese, alone",
        "Filipino alone", "Japanese", "Korean alone", "Laotian alone",
        "Other Asian alone", "Pakistani alone", "Taiwanese alone",
        "Vietnamese", "Vietnamese alone"
    ],
    "Black or African American": [
        "Black or African American", "Black or African American alone"
    ],
    "White": [
        "White", "White alone"
    ],
    "American Indian or Alaska Native": [
        "All other specified American Indian tribe combinations",
        "American Indian and Alaska Native, not specified",
        "American Indian, tribe not specified", "Cherokee alone",
        "Navajo alone", "Mexican American Indian alone",
        "Other specified American Indian tribes alone", "Yaqui alone"
    ],
    "Native Hawaiian or Pacific Islander": [
        "Native Hawaiian alone"
    ],
    "Other / Mixed": [
        "Two or More Races",
        "Some Other Race alone",
        "Other"
    ]
}

# Helper to map race value to a broader category
def map_race(value):
    for group, keywords in race_mapping.items():
        if value in keywords:
            return group
    return "Unknown"

# Prepare metadata
persona_metadata = {}

for idx, persona_str in enumerate(df['persona']):
    persona_id = idx

    age_match = re.search(r"age: (\d+)", persona_str)
    sex_match = re.search(r"sex: (.+)", persona_str)
    race_match = re.search(r"race: (.+)", persona_str)
    ancestry_match = re.search(r"ancestry: (.+)", persona_str)
    birthplace_match = re.search(r"place of birth: (.+)", persona_str)
    personality_match = re.search(r"big five scores: (.+)", persona_str)
    political_match = re.search(r"political views: (.+)", persona_str)
    religion_match = re.search(r"religion: (.+)", persona_str)

    # Get raw race string and map to group
    race_raw = race_match.group(1).strip() if race_match else None
    race_group = map_race(race_raw) if race_raw else None

    persona_metadata[persona_id] = {
        "age": age_match.group(1) if age_match else None,
        "sex": sex_match.group(1).strip() if sex_match else None,
        "race": race_group,
        "ancestry": ancestry_match.group(1).strip() if ancestry_match else None,
        "birthplace": birthplace_match.group(1).strip() if birthplace_match else None,
        "personality": personality_match.group(1).strip() if personality_match else None,
        "political_views": political_match.group(1).strip() if political_match else None,
        "religion": religion_match.group(1).strip() if religion_match else None,
    }

# Save to JSON
with open("multi_persona_data/persona_metadata.json", "w") as f:
    json.dump(persona_metadata, f, indent=2)


import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Colorblind-safe Tol palette
tol_colorblind = [
    "#332288", "#88CCEE", "#44AA99", "#117733",
    "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"
]
# Convert metadata to DataFrame
meta_df = pd.DataFrame.from_dict(persona_metadata, orient='index')

# Create output directory
output_dir = "815_personas"
os.makedirs(output_dir, exist_ok=True)

# Clean age data
meta_df_cleaned = meta_df.dropna(subset=['age'])
meta_df_cleaned['age'] = pd.to_numeric(meta_df_cleaned['age'], errors='coerce')

# Plot styles
sns.set(style="whitegrid")


# Plot setup
sns.set(style="whitegrid")
output_dir = "815_personas"
os.makedirs(output_dir, exist_ok=True)

# Race plot
# --- RACE DISTRIBUTION with % overlay ---
plt.figure(figsize=(10, 5))
ax = sns.countplot(
    data=meta_df,
    y='race',
    order=meta_df['race'].value_counts().index,
    palette=tol_colorblind
)

# Add percentages on the bars
total = len(meta_df)
for container in ax.containers:
    for bar in container:
        width = bar.get_width()
        percentage = (width / total) * 100
        ax.text(
            width + 1, bar.get_y() + bar.get_height() / 2,
            f"{percentage:.1f}%",
            va='center',
            fontsize=10
        )

plt.title('Distribution of Race')
plt.xlabel('Count')
plt.ylabel('Race Group')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "race_distribution.png"))
plt.close()

# --- SEX DISTRIBUTION with % overlay ---
plt.figure(figsize=(6, 4))
ax = sns.countplot(
    data=meta_df,
    x='sex',
    order=meta_df['sex'].value_counts().index,
    palette=tol_colorblind
)

# Add percentages above each bar
total = len(meta_df)
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        percentage = (height / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{percentage:.1f}%",
            ha='center',
            fontsize=10
        )

plt.title('Distribution of Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sex_distribution.png"))
plt.close()
# --- AGE DISTRIBUTION with summary stats ---
plt.figure(figsize=(8, 4))

# Clean and convert age
meta_df_cleaned = meta_df.dropna(subset=['age']).copy()
meta_df_cleaned['age'] = pd.to_numeric(meta_df_cleaned['age'], errors='coerce')

# Plot histogram
sns.histplot(meta_df_cleaned['age'], bins=20, kde=True, color=tol_colorblind[1])

# Calculate stats
mean_age = meta_df_cleaned['age'].mean()
median_age = meta_df_cleaned['age'].median()
std_age = meta_df_cleaned['age'].std()

# Annotate stats on the plot
stats_text = f"Mean: {mean_age:.1f}\nMedian: {median_age:.1f}\nStd Dev: {std_age:.1f}"
plt.text(
    0.95, 0.95,
    stats_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
)

plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "age_distribution.png"))
plt.close()

print("✅ Plots saved using colorblind-safe palette.")