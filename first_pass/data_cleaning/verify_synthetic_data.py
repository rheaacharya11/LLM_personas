import pandas as pd
import glob
import os

# Path to your CSV files - adjust this to your directory
path = '../results/chunked_outputs/'  

# Use glob to get all files matching the pattern
all_files = glob.glob(os.path.join(path, "equal_run_*.csv"))

# Sort files to display in a logical order
all_files.sort()

# Create a dictionary to store file names and row counts
file_counts = {}
total_rows = 0

# Read each file and count rows
print("File Name\t\t\tRow Count")
print("-" * 50)

for filename in all_files:
    base_name = os.path.basename(filename)
    df = pd.read_csv(filename)
    row_count = len(df)
    file_counts[base_name] = row_count
    total_rows += row_count
    print(f"{base_name}\t\t{row_count}")

print("-" * 50)
print(f"Total number of files: {len(file_counts)}")
print(f"Total rows across all files: {total_rows}")
print(f"Average rows per file: {total_rows / len(file_counts) if file_counts else 0:.2f}")

# Check if all files have the same number of rows
all_same = all(count == list(file_counts.values())[0] for count in file_counts.values())
print(f"All files have the same number of rows: {all_same}")

# If not all the same, show min and max
if not all_same and file_counts:
    min_file = min(file_counts, key=file_counts.get)
    max_file = max(file_counts, key=file_counts.get)
    print(f"Min rows: {file_counts[min_file]} in {min_file}")
    print(f"Max rows: {file_counts[max_file]} in {max_file}")

# Verify column structure of the first file
if all_files:
    first_file = all_files[0]
    df = pd.read_csv(first_file)
    print(f"\nColumn structure of {os.path.basename(first_file)}:")
    print(f"Number of columns: {len(df.columns)}")
    print("Column names:")
    for col in df.columns:
        print(f"  - {col}")