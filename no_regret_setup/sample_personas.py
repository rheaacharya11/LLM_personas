import pandas as pd
import re

# Step 1: Load and process personas
df_personas = pd.read_parquet("data/unique_personas.parquet")

# Extract age
df_personas["age"] = df_personas["persona"].apply(lambda x: int(re.search(r"age: (\d+)", x).group(1)) if re.search(r"age: (\d+)", x) else None)

# Filter adults and sample 100
df_adults = df_personas[df_personas["age"] >= 18]
sampled_personas = df_adults.sample(n=100, random_state=11)

# Step 2: Get adjusted persona_ids (1 + index)
selected_persona_ids = (sampled_personas.index).tolist()

# Step 3: Load the aggregated judgments
df_judgments = pd.read_csv("multi_persona_data/test_persona_judgments.csv")

# Step 4: Filter to include only the selected persona_ids
filtered_judgments = df_judgments[df_judgments["persona_id"].isin(selected_persona_ids)]

# Step 5 (Optional): Save to a new file
filtered_judgments.to_csv("multi_persona_data/100heldout_persona_judgments.csv", index=False)

# For verification
print(filtered_judgments.head())
