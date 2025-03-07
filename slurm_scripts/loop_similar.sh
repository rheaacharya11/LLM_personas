#!/bin/bash
#SBATCH --job-name=llama3_instruct_test
#SBATCH --output=jobs/%j/llama3_instruct_test_%j.out
#SBATCH --error=jobs/%j/llama3_instruct_test_%j.err
#SBATCH --time=10:00:00 # Set the time limit for the job
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --mem=32G # Allocate 32GB of memory
#SBATCH --cpus-per-task=4 # Allocate 4 CPUs
#SBATCH --partition=gpu # Specify the GPU partition

# Load required modules (adjust according to your cluster setup)
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
source ~/my_llama_python/bin/activate

# Define parameters as variables (easier to change for each submission)
START_INDEX=950
END_INDEX=999
OUTPUT_NAME="similar_run"

# Create results directory if it doesn't exist
mkdir -p results/chunked_outputs

# Create the output file path
OUTPUT_FILE="results/chunked_outputs/${OUTPUT_NAME}_${START_INDEX}_to_${END_INDEX}.csv"

echo "Processing personas from $START_INDEX to $END_INDEX, saving to $OUTPUT_FILE"

# Run the Python script with specified parameters
python instruct_queries/similar_query.py --start_index $START_INDEX --end_index $END_INDEX --output $OUTPUT_FILE

echo "Processing complete"