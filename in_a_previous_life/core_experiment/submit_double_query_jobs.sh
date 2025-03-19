#!/bin/bash

# SLURM Job Manager for double_query.py
# This script submits jobs in batches with dependencies to avoid overloading the queue

# Usage information
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <output_name> <num_personas> <pairs_per_persona> <max_concurrent_jobs> [model=llama3-8b-instruct]"
    echo ""
    echo "Example: $0 double_query_study 1000 50 20 llama3-8b-instruct"
    echo "This will process 1000 personas, with 50 comparison pairs per persona, "
    echo "maintaining at most 20 concurrent jobs, using llama3-8b-instruct model"
    exit 1
fi

# Get parameters from command line
OUTPUT_NAME=$1
TOTAL_PERSONAS=$2
PAIRS_PER_PERSONA=$3
MAX_CONCURRENT_JOBS=$4
MODEL=${5:-llama3-8b-instruct}  # Default to llama3-8b-instruct if not specified

# Create a log file
LOG_FILE="job_manager_${OUTPUT_NAME}.log"
echo "Starting job manager at $(date)" > $LOG_FILE
echo "Parameters: output_name=$OUTPUT_NAME, personas=$TOTAL_PERSONAS, pairs_per_persona=$PAIRS_PER_PERSONA, max_concurrent=$MAX_CONCURRENT_JOBS, model=$MODEL" >> $LOG_FILE

# Max personas per job - each job will process this many personas
PERSONAS_PER_JOB=10

# Fixed number of comparisons per persona - each will be queried twice (X vs Y and Y vs X)
COMPARISONS_PER_JOB=$PAIRS_PER_PERSONA  # Process all comparisons for each persona in a single job

# Calculate the dimensions of our job grid
# With this configuration, we only split by personas, not by comparisons
NUM_PERSONA_BLOCKS=$(( ($TOTAL_PERSONAS + $PERSONAS_PER_JOB - 1) / $PERSONAS_PER_JOB ))
NUM_COMPARISON_BLOCKS=1  # No splitting by comparisons
TOTAL_JOBS=$NUM_PERSONA_BLOCKS

echo "Creating $TOTAL_JOBS jobs, each processing $PERSONAS_PER_JOB personas with all $PAIRS_PER_PERSONA comparisons each" | tee -a $LOG_FILE
echo "Each persona will be queried twice (X vs Y and Y vs X) for a total of $(($PAIRS_PER_PERSONA * 2)) judgments per persona" | tee -a $LOG_FILE
echo "Using model: $MODEL" | tee -a $LOG_FILE
echo "Maximum concurrent jobs: $MAX_CONCURRENT_JOBS" | tee -a $LOG_FILE

# Create necessary directories
mkdir -p ../jobs
mkdir -p ../results/chunked_outputs

# Create an array to track job IDs for dependencies
declare -a JOB_IDS

# Tracking counters
JOB_COUNTER=0
WAVE_COUNTER=1
ACTIVE_JOBS=0

echo "Starting wave $WAVE_COUNTER of job submissions" | tee -a $LOG_FILE

# Loop over persona blocks only
for ((p=0; p<$NUM_PERSONA_BLOCKS; p++)); do
    # Calculate persona range for this block
    START_PERSONA=$(($p * $PERSONAS_PER_JOB))
    END_PERSONA=$((($p + 1) * $PERSONAS_PER_JOB - 1))
    
    # Cap the end persona to the total number of personas
    if [ $END_PERSONA -ge $TOTAL_PERSONAS ]; then
        END_PERSONA=$(($TOTAL_PERSONAS - 1))
    fi
    
    # Increment job counter
    JOB_COUNTER=$((JOB_COUNTER + 1))
    
    # Create job-specific output directory
    JOB_DIR="../jobs/cot/p${p}"
    mkdir -p $JOB_DIR
    
    # Define output file path
    OUTPUT_FILE="../results/chunked_outputs/${OUTPUT_NAME}_p${START_PERSONA}-${END_PERSONA}.csv"
    
    # Create the job script
    cat > $JOB_DIR/job.sh << EOL
#!/bin/bash
#SBATCH --job-name=${OUTPUT_NAME}_p${p}
#SBATCH --output=${JOB_DIR}/${OUTPUT_NAME}_p${p}.out
#SBATCH --error=${JOB_DIR}/${OUTPUT_NAME}_p${p}.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_requeue

# Load required modules
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
source ~/my_llama_python/bin/activate

# Print job information
echo "Processing personas from $START_PERSONA to $END_PERSONA"
echo "Processing all $PAIRS_PER_PERSONA comparisons"
echo "Output file: $OUTPUT_FILE"

# Run the double query script with specific persona range
python double_query_cot.py \\
    --output $OUTPUT_FILE \\
    --model $MODEL \\
    --start_index $START_PERSONA \\
    --end_index $END_PERSONA \\
    --pairs_per_persona $PAIRS_PER_PERSONA

echo "Processing complete for job ${JOB_COUNTER}/$TOTAL_JOBS"
EOL
        
        # Check if we need dependencies
        if [ ${#JOB_IDS[@]} -ge $MAX_CONCURRENT_JOBS ]; then
            # Calculate how many old jobs to wait for
            JOBS_TO_WAIT=$((${#JOB_IDS[@]} - $MAX_CONCURRENT_JOBS + 1))
            
            # Create dependency string for the oldest JOBS_TO_WAIT jobs
            DEPENDENCY=""
            for ((i=0; i<$JOBS_TO_WAIT; i++)); do
                if [ -n "$DEPENDENCY" ]; then
                    DEPENDENCY="${DEPENDENCY},"
                fi
                DEPENDENCY="${DEPENDENCY}${JOB_IDS[$i]}"
            done
            
            # Submit job with dependency
            JOB_ID=$(sbatch --dependency=afterany:$DEPENDENCY $JOB_DIR/job.sh | awk '{print $4}')
            
            # Remove the jobs we've waited for
            JOB_IDS=("${JOB_IDS[@]:$JOBS_TO_WAIT}")
            
            echo "Submitted job ${JOB_COUNTER}/$TOTAL_JOBS (ID: $JOB_ID) with dependency on $DEPENDENCY" | tee -a $LOG_FILE
        else
            # Submit job without dependency
            JOB_ID=$(sbatch $JOB_DIR/job.sh | awk '{print $4}')
            echo "Submitted job ${JOB_COUNTER}/$TOTAL_JOBS (ID: $JOB_ID) without dependency" | tee -a $LOG_FILE
        fi
        
        # Add this job ID to our tracking array
        JOB_IDS+=("$JOB_ID")
        
        # Increment active jobs counter
        ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
        
        # Check if we've reached MAX_CONCURRENT_JOBS for this wave
        if [ $ACTIVE_JOBS -ge $MAX_CONCURRENT_JOBS ]; then
            WAVE_COUNTER=$((WAVE_COUNTER + 1))
            echo "Starting wave $WAVE_COUNTER of job submissions" | tee -a $LOG_FILE
            ACTIVE_JOBS=0
        fi
        
        # Small delay to prevent overwhelming the scheduler
        sleep 0.5
done

echo "All $TOTAL_JOBS jobs submitted in $WAVE_COUNTER waves!" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE"

# Create a monitor script
cat > monitor_jobs.sh << EOF
#!/bin/bash
# Check status of all submitted jobs
squeue -u \$USER | grep "$OUTPUT_NAME"

# Count completed and remaining jobs
echo ""
echo "Progress summary:"
COMPLETED=\$(ls -1 ../results/cot/chunked_outputs/${OUTPUT_NAME}_*.csv 2>/dev/null | wc -l)
echo "Completed job files: \$COMPLETED / $TOTAL_JOBS"
EOF

chmod +x monitor_jobs.sh
echo "Created monitor_jobs.sh script to check progress"