#!/bin/bash
#SBATCH --job-name=fairness_analysis
#SBATCH --output=logs/fairness_analysis_%j.out
#SBATCH --error=logs/fairness_analysis_%j.err
#SBATCH --time=04:00:00                # Set time limit (HH:MM:SS)
#SBATCH --partition=sapphire           # Or whatever your cluster supports
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # Good for multiprocessing
#SBATCH --mem=32G                      # Adjust based on need

# Activate your conda environment
source ~/my_llama_python/bin/activate

# Run your analysis
python analyze_explanations.py --folder results/fixed_personas_binary --output fairness_analysis_results_1
