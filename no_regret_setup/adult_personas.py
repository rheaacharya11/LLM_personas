import pandas as pd
import re

# Load the parquet file
df = pd.read_parquet("data/unique_personas.parquet")

# Function to extract age from persona string
def extract_age(persona_str):
    match = re.search(r"age:\s*(\d+)", persona_str)
    return int(match.group(1)) if match else None

# Apply to dataframe
df['age_extracted'] = df['persona'].apply(extract_age)

# Filter for age 18+
df_18_plus = df[df['age_extracted'] >= 18]

# Drop the temporary column if you don't want to keep it
df_18_plus = df_18_plus.drop(columns=['age_extracted'])

# Save to a new parquet file
df_18_plus.to_parquet("data/adult_personas.parquet", index=False)

print(f"Saved {len(df_18_plus)} personas age 18 or older.")
