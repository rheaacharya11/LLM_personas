#!/bin/bash
#SBATCH --job-name=test_binary_personas   # SLURM job name
#SBATCH --output=logs/test_binary_personas/test_binary_personas_161_162.out
#SBATCH --error=logs/test_binary_personas/test_binary_personas_161_162.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu_test

# Print job details
echo "=== SLURM Job Started ==="
echo "Current directory: $(pwd)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running as user: $(whoami)"

# Load required modules
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01

# Activate virtual environment
source ~/my_llama_python/bin/activate

# Create required directories if they don't exist
mkdir -p ../results/binary_personas/
mkdir -p logs/test_binary_personas/

# Debug: Check directories before execution
echo "Listing src directory:"
ls -lh ../src/
echo "Listing data directory:"
ls -lh ../data/processed/
echo "Listing results directory before execution:"
ls -lh ../results/binary_personas/

# Run with correct paths
echo "Now calling Python script..."
python -u ../src/comparison_elicitation.py \
  --train_path ../data/processed/compas_train.parquet \
  --personas_path ../data/unique_personas.parquet \
  --output ../results/binary_personas/fairness_judgments_test1.csv \
  --prompt_config ../prompts/binary_config.yaml \
  --pairs_per_persona 5 \
  --use_personas True \
  --prompt_type chain_of_thought \
  --judgment_type binary \
  --persona_start 162 \
  --persona_end 163 \
  > logs/test_binary_personas/test_python_stdout1.log 2> logs/test_binary_personas/test_python_stderr1.log

echo "=== Python script finished ==="

# Check if output file was created
ls -lh ../results/binary_personas/fairness_judgments_test1.csv
cat ../results/binary_personas/fairness_judgments_test1.csv || echo "Output file is empty or missing."

echo "=== SLURM Job Finished ==="