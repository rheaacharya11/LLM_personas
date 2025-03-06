#!/bin/bash
#SBATCH --job-name=llama3_download
#SBATCH --output=llama3_download_%j.out
#SBATCH --error=llama3_download_%j.err

#SBATCH --time=01:00:00              # Set the time limit for the job
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --mem=32G                     # Allocate 32GB of memory
#SBATCH --cpus-per-task=4             # Allocate 4 CPUs
#SBATCH --partition=gpu               # Specify the GPU partition

# Activate your environment if necessary (e.g., conda environment)
# source activate myenv

# Create the ../models/ directory if it doesn't exist
# mkdir -p ../models/

# Download the model from Hugging Face and specify the directory to save it
huggingface-cli download meta-3-8B-instruct --cache-dir ../models