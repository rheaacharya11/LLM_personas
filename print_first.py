# print_ids.py
import pyarrow.parquet as pq

# Path to your Parquet file
file_path = 'data/processed/compas_train.parquet'

# Read the Parquet file
table = pq.read_table(file_path, columns=['id', 'two_year_recid'])

# Print the first 10 'id' values
for i in range(min(10, table.num_rows)):
    print(table.column('id')[i].as_py())

for x in [2464, 197, 4096, 2057, 108, 2641, 4546, 3608, 464, 4016]:
    print(table.column('two_year_recid')[x])
