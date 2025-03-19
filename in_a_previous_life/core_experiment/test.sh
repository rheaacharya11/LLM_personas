#!/bin/bash
# Test script to run a single fairness evaluation job
# This script creates and runs one small job to test the framework

# Setup directory structure
CORE_DIR="core_experiment"
JOBS_DIR="jobs/core_experiment"
RESULTS_DIR="results/core_experiment/chunked_outputs"

# Create necessary directories
mkdir -p $JOBS_DIR/test_job
mkdir -p $RESULTS_DIR

# Define test parameters
START_PERSONA=0
END_PERSONA=0            # Just test the first persona
START_COMPARISON=0
END_COMPARISON=2         # Just test the first 3 comparisons (0, 1, 2)
MODEL="llama3-8b-instruct"
OUTPUT_FILE="${RESULTS_DIR}/test_job_p${START_PERSONA}-${END_PERSONA}_c${START_COMPARISON}-${END_COMPARISON}.csv"

echo "Setting up test job for:"
echo "- Persona range: $START_PERSONA to $END_PERSONA"
echo "- Comparison range: $START_COMPARISON to $END_COMPARISON"
echo "- Model: $MODEL"
echo "- Output will be saved to: $OUTPUT_FILE"

# Create the job script
cat > $JOBS_DIR/test_job/job.sh << EOL
#!/bin/bash
#SBATCH --job-name=fairness_test
#SBATCH --output=${JOBS_DIR}/test_job/test_job.out
#SBATCH --error=${JOBS_DIR}/test_job/test_job.err
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_test

# Load required modules
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
source ~/my_llama_python/bin/activate

# Print job information
echo "Running test job"
echo "Processing personas from $START_PERSONA to $END_PERSONA"
echo "Processing comparisons from $START_COMPARISON to $END_COMPARISON"
echo "Output file: $OUTPUT_FILE"

# Navigate to core experiment directory
cd ${CORE_DIR}

# Run the experiment script with specific persona and comparison ranges
python generate_fairness_constraints.py \\
    --output $OUTPUT_FILE \\
    --model $MODEL \\
    --start_index $START_PERSONA \\
    --end_index $END_PERSONA \\
    --start_comparison $START_COMPARISON \\
    --end_comparison $END_COMPARISON \\
    --pairs_per_persona 50

echo "Test job complete"
EOL

echo "Test job script created at $JOBS_DIR/test_job/job.sh"
echo "Submitting test job..."

# Submit the job
job_id=$(sbatch $JOBS_DIR/test_job/job.sh | awk '{print $4}')

echo "Test job submitted with ID: $job_id"
echo "You can monitor the job with: squeue -j $job_id"
echo "And check the output with: tail -f ${JOBS_DIR}/test_job/test_job.out"
echo ""
echo "After the job completes, check the results with:"
echo "  cat $OUTPUT_FILE"
echo "And examine the output log with:"
echo "  cat ${JOBS_DIR}/test_job/test_job.out"