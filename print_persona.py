import pandas as pd

# Load the Parquet file
file_path = "data/persona_train1.parquet"  # Replace with your actual file path
df = pd.read_parquet(file_path)

# Print the persona attribute of the first row
if 'persona' in df.columns:
    print(df.loc[0, 'persona'])
else:
    print("The 'persona' attribute is not found in the dataset.")
