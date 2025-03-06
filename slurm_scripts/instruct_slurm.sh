#!/bin/bash
#SBATCH --job-name=llama3_instruct_test
#SBATCH --output=llama3_instruct_test_%j.out
#SBATCH --error=llama3_instruct_test_%j.err
#SBATCH --time=01:00:00              # Set the time limit for the job
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --mem=32G                     # Allocate 32GB of memory
#SBATCH --cpus-per-task=4             # Allocate 4 CPUs
#SBATCH --partition=gpu               # Specify the GPU partition

# Load required modules (adjust according to your cluster setup)
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01

source ~/my_llama_python/bin/activate

# Activate virtual environment if you're using one
# source /path/to/your/venv/bin/activate

# Run the Python script
python query_instruct.py