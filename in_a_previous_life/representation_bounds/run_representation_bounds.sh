#!/bin/bash
#SBATCH --job-name=rep_bounds
#SBATCH --output=jobs/rep_bounds_%j.out
#SBATCH --error=jobs/rep_bounds_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu_test

# Load any necessary modules
module load python/3.10.9-fasrc01

# Activate your Python environment if needed
source ~/my_llama_python/bin/activate

# Print job information
echo "Running representation bounds analysis"
echo "Started at $(date)"
echo "Running on $(hostname)"

# Run the analysis script
python representation_bounds.py

echo "Job completed at $(date)"