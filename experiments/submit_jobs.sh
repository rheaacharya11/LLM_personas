#!/bin/bash
# Abstracted Script for Submitting Jobs for Experiments with Batching
# This script submits jobs in batches (e.g., 10 personas at a time).
# Activate virtual environment

# Function to print usage information
print_usage() {
    echo "Usage: $0 <experiment_name> <config_file> <persona_start> <persona_end> <batch_size>"
    echo ""
    echo "Arguments:"
    echo "  experiment_name: Name for the experiment batch (e.g., binary_judgment)"
    echo "  config_file: Path to the configuration JSON file"
    echo "  persona_start: The starting persona number (inclusive)"
    echo "  persona_end: The ending persona number (inclusive)"
    echo "  batch_size: The number of personas to process per job (e.g., 10)"
    echo ""
    echo "Example: $0 binary_judgment configs/binary_config.json 1 1000 10"
}

# Function to create necessary directories for jobs, logs, and results
create_directories() {
    mkdir -p jobs/${EXPERIMENT_NAME}
    mkdir -p logs/${EXPERIMENT_NAME}
    mkdir -p ../results/${EXPERIMENT_NAME}
}

# Function to parse experiment configuration file
parse_config() {
    local config_file=$1
    local param_str=""
    # Check if file exists
    if [ ! -f "${config_file}" ]; then
        echo "Error: Configuration file ${config_file} not found"
        exit 1
    fi
    # Extract parameters from JSON
    params=$(python -c "
import json
with open('${config_file}', 'r') as f:
    config = json.load(f)
for name, experiment in enumerate(config):
    if '_name' in experiment and experiment['_name'] == '${EXPERIMENT_NAME}':
        print(' '.join([f'--{k} {v}' for k, v in experiment.items() if not k.startswith('_')]))
        break
")
    echo ${params}
}

# Function to submit a single job for a range of personas
submit_job() {
    local persona_start=$1
    local persona_end=$2
    local cmd=$3
    local job_name="${EXPERIMENT_NAME}_personas_${persona_start}_${persona_end}"
    local job_file="jobs/${EXPERIMENT_NAME}/${job_name}.sh"
    
    # Create the job script
    cat > ${job_file} << EOL
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${EXPERIMENT_NAME}/${job_name}.out
#SBATCH --error=logs/${EXPERIMENT_NAME}/${job_name}.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_requeue

# Load required modules
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
# Activate virtual environment
source ~/my_llama_python/bin/activate

# Print job information
echo "Running experiment: ${job_name}"
echo "Command: ${cmd}"
echo "Started at: \$(date)"

# Execute command
${cmd}

echo "Finished at: \$(date)"
EOL

    # Submit the job and get job ID
    job_id=$(sbatch ${job_file} | awk '{print $4}')
    echo "Submitted job ${job_name} with ID ${job_id}"
    echo ${job_id}
}

# Function to submit jobs for a batch of personas
submit_jobs_for_batches() {
    local start=$1
    local end=$2
    local batch_size=$3

    # Loop through the range of personas in steps of batch_size
    for (( persona_start=${start}; persona_start<=${end}; persona_start+=${batch_size} )); do
        # Calculate the end index for this batch
        persona_end=$(( persona_start + batch_size - 1 ))
        if (( persona_end > end )); then
            persona_end=${end}
        fi
        # Create command with specific persona range
        cmd="${BASE_CMD} ${PARAMS} --persona_start ${persona_start} --persona_end ${persona_end}"
        submit_job "${persona_start}" "${persona_end}" "${cmd}"
    done
}

# Main script execution
if [ "$#" -lt 5 ]; then
    print_usage
    exit 1
fi

# Get parameters from the command line
EXPERIMENT_NAME=$1
CONFIG_FILE=$2
PERSONA_START=$3
PERSONA_END=$4
BATCH_SIZE=$5

# Create necessary directories for jobs, logs, and results
create_directories

# Base command for the experiment
BASE_CMD="python ../src/comparison_elicitation.py"

# Parse the configuration file for experiment parameters
PARAMS=$(parse_config ${CONFIG_FILE})
if [ -z "${PARAMS}" ]; then
    echo "Error: Could not find experiment '${EXPERIMENT_NAME}' in config file"
    exit 1
fi

# Submit jobs for the specified range of personas and batch size
submit_jobs_for_batches ${PERSONA_START} ${PERSONA_END} ${BATCH_SIZE}

echo "Job submission complete for experiment: ${EXPERIMENT_NAME}"


# Usage Example
# To run the script for 10 personas at a time:
# $ ./submit_experiment.sh binary_judgment configs/binary_config.json 1 1000 50

