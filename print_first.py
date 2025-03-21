# print_ids.py
import pyarrow.parquet as pq

# Path to your Parquet file
file_path = 'data/processed/compas_train.parquet'

# Read the Parquet file
table = pq.read_table(file_path, columns=['id'])

# Print the first 10 'id' values
for i in range(min(10, table.num_rows)):
    print(table.column('id')[i].as_py())
