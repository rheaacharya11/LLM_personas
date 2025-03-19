#!/bin/bash
#SBATCH --job-name=vanilla_llm_cot
#SBATCH --output=vanilla_llm_cot.out
#SBATCH --error=vanilla_llm_cot.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

# Load required modules
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
source ~/my_llama_python/bin/activate

# Print job information
echo "Running vanilla LLM COT COMPAS fairness study"

# Run the vanilla LLM script 
python vanilla_fixed.py \
    --output ../results/vanilla_llm_cot.csv \
    --model llama3-8b-instruct \
    --fixed_comparisons_file ../data/fixed_comparisons.json \
    --model_path_prefix "../models/"

echo "Processing complete for vanilla LLM COT study"