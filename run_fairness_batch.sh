#!/bin/bash
#SBATCH --job-name=fairness_algo
#SBATCH --output=logs/no_regret/binary_personas/no_regret_%A_%a.out
#SBATCH --error=logs/no_regret/binary_personas/no_regret_%A_%a.err
#SBATCH --array=0-999  # Adjust this based on your total judges / batch_size
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Create logs directory
mkdir -p logs/no_regret/binary_personas/

# Activate your environment if needed
source ~/my_llama_python/bin/activate

# Define your parameters
CONSTRAINT_PATH="constraint_sets/lenient/binary_personas/constraint_sets.json"
DATA_PATH="data/processed/compas_train.parquet"
OUT_DIR="no_regret_results/binary_personas/"
GAMMA_VALUES="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
ITERATIONS=1000
BATCH_SIZE=10  # Number of judges to process per job
JUDGE_START=1
JUDGE_END=999

# Log the configuration
echo "Job started at $(date)"
echo "Processing batch ${SLURM_ARRAY_TASK_ID} with batch size ${BATCH_SIZE}"

# Run the batch fairness script for this batch
python batch_fairness.py \
    --constraint_path "$CONSTRAINT_PATH" \
    --data_path "$DATA_PATH" \
    --out_dir "$OUT_DIR" \
    --gamma_values "$GAMMA_VALUES" \
    --iterations $ITERATIONS \
    --batch_size $BATCH_SIZE \
    --batch_id ${SLURM_ARRAY_TASK_ID} \
    --judge_start $JUDGE_START \
    --judge_end $JUDGE_END

echo "Job finished at $(date)"