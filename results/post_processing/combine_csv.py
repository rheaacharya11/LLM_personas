import pandas as pd
import os
import csv
import tempfile

# Define the path to the directory containing the CSV files
folder_path = '../binary_personas/'

# Ensure the folder exists
if not os.path.exists(folder_path):
    print(f"The specified directory {folder_path} does not exist.")
    exit()

# List all CSV files in the directory that match the naming pattern
file_names = [f for f in os.listdir(folder_path) if f.startswith('fairness_judgments_') and f.endswith('.csv')]

# Check if files were found
if not file_names:
    print("No files found in the directory.")
    exit()

# Sort the files by the numbers in the filenames to maintain the order
file_names.sort(key=lambda f: int(f.split('_')[2].split('.')[0]))

# Initialize an empty list to store the dataframes
dfs = []

# Read each CSV file and append it to the list
for file in file_names:
    file_path = os.path.join(folder_path, file)
    try:
        # Preprocess the file to remove internal newlines within quotes
        with open(file_path, 'r', encoding='utf-8') as original, tempfile.NamedTemporaryFile('w+', delete=False) as temp:
            writer = csv.writer(temp, quoting=csv.QUOTE_MINIMAL)
            for line in csv.reader(original, quotechar='"', delimiter=',', quoting=csv.QUOTE_MINIMAL):
                writer.writerow([field.replace('\n', ' ').replace('\r', '') for field in line])
        
        # Read the cleaned file
        df = pd.read_csv(temp.name)
        dfs.append(df)
        os.unlink(temp.name)  # Remove the temporary file
    except Exception as e:
        print(f"Failed to read {file_path}. Error: {e}")

# Check if dataframes were appended
if not dfs:
    print("No data was loaded into dataframes.")
    exit()

# Concatenate all dataframes into one
complete_df = pd.concat(dfs, ignore_index=True)

# Save the concatenated dataframe to a new CSV file
output_directory = "../combined_results/binary_personas"
os.makedirs(output_directory, exist_ok=True)  # Avoids error if the directory already exists
output_file_path = os.path.join(output_directory, 'COMPLETE_binary_personas.csv')

try:
    complete_df.to_csv(output_file_path, index=False)
    print(f"All files have been combined into {output_file_path}")
except Exception as e:
    print(f"Failed to save the combined CSV. Error: {e}")
