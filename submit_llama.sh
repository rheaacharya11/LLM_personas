#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-02:00
#SBATCH -p gpu_requeue
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o jobs/%j/llama_output.out
#SBATCH -e jobs/%j/llama_error.err

module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01

source ~/my_llama_python/bin/activate

# Get SLURM Job ID
JOB_ID=$SLURM_JOB_ID
OUTPUT_DIR="jobs/$JOB_ID"

# Create the job-specific folder
mkdir -p $OUTPUT_DIR

# Define choices
OPTION1="Adopt a stray puppy and take care of it"
OPTION2="Go on a fun amusement park trip with friends"

MODEL_PATH="/n/holylabs/LABS/dwork_lab/Lab/rheaacharya/models/Llama-3-8B/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"

# Run the model and save output inside the job-specific folder
python run_llama.py --model_path $MODEL_PATH --option1 "$OPTION1" --option2 "$OPTION2" --output "$OUTPUT_DIR/llama_response.txt"
