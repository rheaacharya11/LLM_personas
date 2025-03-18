#!/bin/bash
# Maximum parallelization SLURM job manager for fairness constraint generation
# This script creates a 2D grid of jobs, splitting both by personas and comparison ranges

# Usage information
if [ "$#" -lt 6 ]; then
    echo "Usage: $0 <output_name> <total_personas> <personas_per_job> <comparisons_per_job> <max_concurrent> [model=llama3-8b-instruct]"
    echo ""
    echo "Example: $0 fairness_constraints 1000 10 10 200 llama3-8b-instruct"
    echo "This will process 1000 personas with each job handling 10 personas and 10 comparisons,"
    echo "maintaining at most 200 concurrent jobs, using llama3-8b-instruct model"
    exit 1
fi

# Get parameters from command line
OUTPUT_NAME=$1
TOTAL_PERSONAS=$2
PERSONAS_PER_JOB=$3
COMPARISONS_PER_JOB=$4
MAX_CONCURRENT=$5
MODEL=${6:-llama3-8b-instruct}  # Default to llama3-8b-instruct if not specified
PAIRS_PER_PERSONA=50           # Each persona gets 50 unique comparison pairs

# Setup directory structure
#CORE_DIR="core_experiment"
JOBS_DIR="jobs/core_experiment_requeue"
RESULTS_DIR="results/core_experiment_requeue/chunked_outputs"

# Create a log file
LOG_FILE="${JOBS_DIR}/job_manager_${OUTPUT_NAME}.log"
mkdir -p $(dirname $LOG_FILE)
echo "Starting job manager at $(date)" > $LOG_FILE
echo "Parameters: output_name=$OUTPUT_NAME, personas=$TOTAL_PERSONAS, personas_per_job=$PERSONAS_PER_JOB, comparisons_per_job=$COMPARISONS_PER_JOB, max_concurrent=$MAX_CONCURRENT, model=$MODEL" >> $LOG_FILE
echo "Each persona will evaluate $PAIRS_PER_PERSONA unique comparison pairs" >> $LOG_FILE

# Calculate the dimensions of our job grid
NUM_PERSONA_BLOCKS=$(( ($TOTAL_PERSONAS + $PERSONAS_PER_JOB - 1) / $PERSONAS_PER_JOB ))
NUM_COMPARISON_BLOCKS=$(( ($PAIRS_PER_PERSONA + $COMPARISONS_PER_JOB - 1) / $COMPARISONS_PER_JOB ))
TOTAL_JOBS=$(( $NUM_PERSONA_BLOCKS * $NUM_COMPARISON_BLOCKS ))

echo "Creating a grid of $NUM_PERSONA_BLOCKS persona blocks × $NUM_COMPARISON_BLOCKS comparison blocks = $TOTAL_JOBS jobs" | tee -a $LOG_FILE
echo "Each job will process up to $PERSONAS_PER_JOB personas and $COMPARISONS_PER_JOB comparisons" | tee -a $LOG_FILE
echo "Using model: $MODEL" | tee -a $LOG_FILE
echo "Maximum concurrent jobs: $MAX_CONCURRENT" | tee -a $LOG_FILE

# Create necessary directories
mkdir -p $JOBS_DIR
mkdir -p $RESULTS_DIR

# Create an array to track job IDs for dependencies
declare -a JOB_IDS

# Tracking counters
JOB_COUNTER=0
WAVE_COUNTER=1
ACTIVE_JOBS=0

echo "Starting wave $WAVE_COUNTER of job submissions" | tee -a $LOG_FILE

# Loop over persona blocks
for ((p=0; p<$NUM_PERSONA_BLOCKS; p++)); do
    # Calculate persona range for this block
    START_PERSONA=$(($p * $PERSONAS_PER_JOB))
    END_PERSONA=$((($p + 1) * $PERSONAS_PER_JOB - 1))
    
    # Cap the end persona to the total number of personas
    if [ $END_PERSONA -ge $TOTAL_PERSONAS ]; then
        END_PERSONA=$(($TOTAL_PERSONAS - 1))
    fi
    
    # Loop over comparison blocks
    for ((c=0; c<$NUM_COMPARISON_BLOCKS; c++)); do
        # Calculate comparison range for this block
        START_COMPARISON=$(($c * $COMPARISONS_PER_JOB))
        END_COMPARISON=$((($c + 1) * $COMPARISONS_PER_JOB - 1))
        
        # Cap the end comparison to the total number of comparisons per persona
        if [ $END_COMPARISON -ge $PAIRS_PER_PERSONA ]; then
            END_COMPARISON=$(($PAIRS_PER_PERSONA - 1))
        fi
        
        # Increment job counter
        JOB_COUNTER=$((JOB_COUNTER + 1))
        
        # Create job-specific output directory
        JOB_DIR="${JOBS_DIR}/p${p}_c${c}"
        mkdir -p $JOB_DIR
        
        # Define output file path
        OUTPUT_FILE="${RESULTS_DIR}/${OUTPUT_NAME}_p${START_PERSONA}-${END_PERSONA}_c${START_COMPARISON}-${END_COMPARISON}.csv"
        
        # Create the job script
        cat > $JOB_DIR/job.sh << EOL
#!/bin/bash
#SBATCH --job-name=${OUTPUT_NAME}_p${p}_c${c}
#SBATCH --output=${JOB_DIR}/${OUTPUT_NAME}_p${p}_c${c}.out
#SBATCH --error=${JOB_DIR}/${OUTPUT_NAME}_p${p}_c${c}.err
#SBATCH --time=2:00:00
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
echo "Processing comparisons from $START_COMPARISON to $END_COMPARISON"
echo "Output file: $OUTPUT_FILE"

# Navigate to core experiment directory

# Run the experiment script with specific persona and comparison ranges
python core_experiment/generate_fairness_constraints.py \\
    --output $OUTPUT_FILE \\
    --model $MODEL \\
    --start_index $START_PERSONA \\
    --end_index $END_PERSONA \\
    --start_comparison $START_COMPARISON \\
    --end_comparison $END_COMPARISON \\
    --pairs_per_persona $PAIRS_PER_PERSONA

echo "Processing complete for job ${JOB_COUNTER}/$TOTAL_JOBS"
EOL
        
        # Check if we need dependencies
        if [ ${#JOB_IDS[@]} -ge $MAX_CONCURRENT ]; then
            # Calculate how many old jobs to wait for
            JOBS_TO_WAIT=$((${#JOB_IDS[@]} - $MAX_CONCURRENT + 1))
            
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
        if [ $ACTIVE_JOBS -ge $MAX_CONCURRENT ]; then
            WAVE_COUNTER=$((WAVE_COUNTER + 1))
            echo "Starting wave $WAVE_COUNTER of job submissions" | tee -a $LOG_FILE
            ACTIVE_JOBS=0
        fi
        
        # Small delay to prevent overwhelming the scheduler
        sleep 0.5
    done
done

echo "All $TOTAL_JOBS jobs submitted in $WAVE_COUNTER waves!" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE"

# Create a monitor script
cat > ${JOBS_DIR}/monitor_jobs.sh << EOF
#!/bin/bash
# Check status of all submitted jobs
squeue -u \$USER | grep "$OUTPUT_NAME"

# Count completed and remaining jobs
echo ""
echo "Progress summary:"
COMPLETED=\$(ls -1 ${RESULTS_DIR}/${OUTPUT_NAME}_*.csv 2>/dev/null | wc -l)
echo "Completed job files: \$COMPLETED / $TOTAL_JOBS"
EOF

chmod +x ${JOBS_DIR}/monitor_jobs.sh
echo "Created monitor_jobs.sh script in ${JOBS_DIR} to check progress"

echo ""
echo "After all jobs complete, combine results with:"
echo "python ${CORE_DIR}/combine_fairness_results.py \\"
echo "  --input_pattern \"${RESULTS_DIR}/${OUTPUT_NAME}_*.csv\" \\"
echo "  --output_file \"results/core_experiment/combined_fairness_judgments.csv\" \\"
echo "  --constraint_dir \"results/core_experiment/fairness_constraints\""