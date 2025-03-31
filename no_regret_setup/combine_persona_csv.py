import os
import re
import pandas as pd

# Define input and output paths
input_folder = "results/fixed_expert_binary/"
output_folder = "multi_persona_data"
persona_file = "data/unique_personas.parquet"

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the persona DataFrame
persona_df = pd.read_parquet(persona_file)

# Extract age from the 'persona' column using regex
def extract_age(persona_str):
    match = re.search(r"age: (\d+)", persona_str)
    return int(match.group(1)) if match else None

persona_df['age'] = persona_df['persona'].apply(extract_age)

# Collect and sort all relevant CSV files
csv_files = sorted([
    f for f in os.listdir(input_folder)
    if f.startswith("A_fairness_judgments_") and f.endswith(".csv")
])

# Combine all CSVs
df_list = []
for file in csv_files:
    print("hi")
    file_path = os.path.join(input_folder, file)
    df = pd.read_csv(file_path)
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

# Subtract 1 from persona_id
combined_df['persona_id'] = combined_df['persona_id'] - 1

# Filter rows where adjusted persona_id has age > 18
def is_age_valid(pid):
    if 0 <= pid < len(persona_df):
        age = persona_df.iloc[pid]['age']
        return age is not None and age >= 18
    return False

# valid_ids = combined_df['persona_id'].apply(is_age_valid)
# filtered_df = combined_df[valid_ids]
filtered_df = combined_df
# Save the result with splits based on comparison_id
train_df = filtered_df[(filtered_df['comparison_id'] >= 0) & (filtered_df['comparison_id'] <= 199)]
# test_df = filtered_df[(filtered_df['comparison_id'] >= 200) & (filtered_df['comparison_id'] <= 999)]

# Save train and test sets
train_output_file = os.path.join(output_folder, "200_expert_judgments.csv")
# test_output_file = os.path.join(output_folder, "800_test_persona_judgments.csv")

train_df.to_csv(train_output_file, index=False)
# test_df.to_csv(test_output_file, index=False)

print(f"Train set saved to: {train_output_file} ({len(train_df)} rows)")
# print(f"Test set saved to: {test_output_file} ({len(test_df)} rows)")
