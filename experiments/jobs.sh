#!/bin/bash
#SBATCH --job-name=three_option_persona
#SBATCH --output=logs/%A/three_option_persona_%A_%a.out
#SBATCH --error=logs/%A/three_option_persona_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --array=0-19  # 20 jobs with no limit on concurrent execution

# Load required modules
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01

# Activate virtual environment
source ~/my_llama_python/bin/activate

# Parse configuration from the JSON file
EXP_NAME="three_option_persona"
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
BATCH_SIZE=50
START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE + 1))
END=$((START + BATCH_SIZE - 1))
if [ $END -gt 1000 ]; then END=1000; fi

echo "=== SLURM Job Started ==="
echo "Processing personas $START to $END"

# Create required directories
mkdir -p ../results/${EXP_NAME}/
mkdir -p logs/${EXP_NAME}/

# Run with correct paths
python -u ../src/comparison_elicitation.py \
${PARAMS} \
--persona_start ${START} \
--persona_end ${END} \
--output ../results/${EXP_NAME}/fairness_judgments_${START}_${END}.csv

echo "=== Job $SLURM_ARRAY_TASK_ID finished: personas $START-$END ==="