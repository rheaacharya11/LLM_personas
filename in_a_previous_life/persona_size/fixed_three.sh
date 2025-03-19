#!/bin/bash
#SBATCH --job-name=fixed_personas
#SBATCH --output=jobs/fixed_personas_%A_%a.out
#SBATCH --error=jobs/fixed_personas_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --array=0-19  # Adjust this based on how many job chunks you need

# Load required modules
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
source ~/my_llama_python/bin/activate

# Set total number of personas and personas per job
TOTAL_PERSONAS=1000
PERSONAS_PER_JOB=50

# Calculate start and end indices for this job
START_IDX=$((SLURM_ARRAY_TASK_ID * PERSONAS_PER_JOB))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * PERSONAS_PER_JOB - 1))

# Cap the end index to total personas
if [ $END_IDX -ge $TOTAL_PERSONAS ]; then
    END_IDX=$((TOTAL_PERSONAS - 1))
fi

# Create output filename based on indices
OUTPUT_FILE="results/fixed_three/chunked_outputs/fixed_personas_${START_IDX}_to_${END_IDX}.csv"

echo "Running fixed personas comparison study for personas $START_IDX to $END_IDX"
echo "Output will be saved to $OUTPUT_FILE"

# Run the fixed comparisons study script
python persona_size/fixed_personas_study.py \
    --output $OUTPUT_FILE \
    --model llama3-8b-instruct \
    --fixed_comparisons_file ../data/fixed_comparisons.json \
    --start_index $START_IDX \
    --end_index $END_IDX \
    --batch_size 10

echo "Job complete for personas $START_IDX to $END_IDX"