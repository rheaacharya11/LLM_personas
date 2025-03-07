#!/bin/bash
#SBATCH --job-name=llama3_instruct_test
#SBATCH --output=jobs/%j/llama3_instruct_test_%j.out
#SBATCH --error=jobs/%j/llama3_instruct_test_%j.err
#SBATCH --time=01:00:00 # Set the time limit for the job
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --mem=32G # Allocate 32GB of memory
#SBATCH --cpus-per-task=4 # Allocate 4 CPUs
#SBATCH --partition=gpu # Specify the GPU partition

# Load required modules (adjust according to your cluster setup)
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
source ~/my_llama_python/bin/activate

# Define the chunk size (number of personas to process in each iteration)
CHUNK_SIZE=50

# Define the total number of personas
TOTAL_PERSONAS=1000  # Replace with your actual total

# Create results directory if it doesn't exist
mkdir -p results/chunked_outputs

# Loop through chunks
for ((start=0; start<TOTAL_PERSONAS; start+=CHUNK_SIZE)); do
    # Calculate end index for this chunk
    end=$((start + CHUNK_SIZE - 1))
    
    # Make sure we don't exceed the total
    if [ $end -ge $TOTAL_PERSONAS ]; then
        end=$((TOTAL_PERSONAS - 1))
    fi
    
    # Create a unique output file name based on the chunk range
    OUTPUT_FILE="results/chunked_outputs/equal_run_${start}_to_${end}.csv"
    
    echo "Processing personas from $start to $end, saving to $OUTPUT_FILE"
    
    # Run the Python script with the current chunk and specific output file
    python instruct_queries/big_query.py --start_index $start --end_index $end --output $OUTPUT_FILE
done

echo "All chunks processed"