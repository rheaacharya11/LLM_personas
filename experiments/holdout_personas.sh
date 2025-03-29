#!/bin/bash
#SBATCH --job-name=holdout_personas
#SBATCH --output=logs/%A/holdout_personas_%A_%a.out
#SBATCH --error=logs/%A/holdout_personas_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=seas_gpu
#SBATCH --array=0-271  

# Load required modules
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01

# Activate virtual environment
source ~/my_llama_python/bin/activate

# Parse configuration from the JSON file
EXP_NAME="holdout_personas"
CONFIG_FILE="configs/experiment_config.json"

# Extract parameters for the specified experiment
PARAMS=$(python -c "
import json
with open('${CONFIG_FILE}', 'r') as f:
    config = json.load(f)
for experiment in config:
    if '_name' in experiment and experiment['_name'] == '${EXP_NAME}':
        print(' '.join([f'--{k} {v}' for k, v in experiment.items() if not k.startswith('_')]))
        break
")

# Calculate persona range for this job
BATCH_SIZE=3
START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
END=$((START + BATCH_SIZE - 1))
if [ $END -ge 815 ]; then END=814; fi

echo "=== SLURM Job Started ==="
echo "Processing personas $START to $END"

# Create required directories
mkdir -p ../results/${EXP_NAME}/
mkdir -p logs/${EXP_NAME}/

# Run with correct paths
python -u ../src/fixed_comparison.py \
${PARAMS} \
--persona_start ${START} \
--persona_end ${END} \
--output ../results/${EXP_NAME}/fairness_judgments_${START}_${END}.csv

echo "=== Job $SLURM_ARRAY_TASK_ID finished: personas $START-$END ==="
