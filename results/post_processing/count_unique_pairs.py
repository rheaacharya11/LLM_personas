import pandas as pd
import matplotlib.pyplot as plt
import sys

def count_unique_pairs(csv_file_path):
    """Counts and prints the number of unique (individual1_id, individual2_id) pairs in the given CSV file."""
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file_path)

        # Create a set of unique pairs using zip
        unique_pairs = set(zip(df['individual1_id'], df['individual2_id']))

        # Calculate the number of unique pairs
        num_unique_pairs = len(unique_pairs)
        print(f"There are {num_unique_pairs} unique 'individual1_id'-'individual2_id' combinations in the dataset.")
    
    except Exception as e:
        print(f"Failed to process the file: {e}")


def plot_pair_distribution(csv_file_path):
    """Plots a histogram of the distribution of pair appearances in the dataset."""
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file_path)

        # Create a DataFrame of pairs
        pairs_df = df[['individual1_id', 'individual2_id']]

        # Count occurrences of each pair
        pair_counts = pairs_df.value_counts().reset_index(name='count')

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create a histogram
        ax.hist(pair_counts['count'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title('Distribution of Pair Appearances')
        ax.set_xlabel('Number of Appearances')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        fig.savefig('../combined_results/binary_personas/distribution.png')

    except Exception as e:
        print(f"Error: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python count_unique_pairs.py <csv_file_path>")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    count_unique_pairs(csv_file_path)
    plot_pair_distribution(csv_file_path)

if __name__ == "__main__":
    main()
