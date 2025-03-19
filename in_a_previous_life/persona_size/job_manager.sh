#!/bin/bash

# Advanced SLURM Job Manager with Dependency Tracking
# This script submits jobs in batches with dependencies to avoid overloading the queue

# Usage information
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <output_name> <num_personas> <comparisons_per_job> <max_concurrent_jobs> [model=llama3-8b-instruct]"
    echo ""
    echo "Example: $0 fixed_compas_study 1000 10 20 llama3-8b-instruct"
    echo "This will process 1000 personas, with each job handling 10 comparisons, "
    echo "maintaining at most 20 concurrent jobs, using llama3-8b-instruct"
    exit 1
fi

# Get parameters from command line
OUTPUT_NAME=$1
TOTAL_PERSONAS=$2
COMPARISONS_PER_JOB=$3
MAX_CONCURRENT_JOBS=$4
MODEL=${5:-llama3-8b-instruct}  # Default to llama3-8b-instruct if not specified

# Create a log file
LOG_FILE="job_manager_${OUTPUT_NAME}.log"
echo "Starting job manager at $(date)" > $LOG_FILE
echo "Parameters: output_name=$OUTPUT_NAME, personas=$TOTAL_PERSONAS, comparisons_per_job=$COMPARISONS_PER_JOB, max_concurrent=$MAX_CONCURRENT_JOBS, model=$MODEL" >> $LOG_FILE

# Load the fixed comparisons file to determine how many comparisons exist
if [ -f "data/fixed_comparisons.json" ]; then
    NUM_COMPARISONS=$(grep -o "comparison_id" data/fixed_comparisons.json | wc -l)
    echo "Found $NUM_COMPARISONS fixed comparisons" | tee -a $LOG_FILE
else
    NUM_COMPARISONS=100
    echo "Couldn't find fixed_comparisons.json, assuming $NUM_COMPARISONS comparisons" | tee -a $LOG_FILE
    echo "Generating fixed comparisons file..." | tee -a $LOG_FILE
    python generate_fixed_comparisons.py --comparisons $NUM_COMPARISONS
fi

# Max personas per job - each job will process this many personas for its assigned comparisons
PERSONAS_PER_JOB=50

# Calculate the dimensions of our job grid
NUM_PERSONA_BLOCKS=$(( ($TOTAL_PERSONAS + $PERSONAS_PER_JOB - 1) / $PERSONAS_PER_JOB ))
NUM_COMPARISON_BLOCKS=$(( ($NUM_COMPARISONS + $COMPARISONS_PER_JOB - 1) / $COMPARISONS_PER_JOB ))
TOTAL_JOBS=$(( $NUM_PERSONA_BLOCKS * $NUM_COMPARISON_BLOCKS ))

echo "Creating a grid of $NUM_PERSONA_BLOCKS persona blocks × $NUM_COMPARISON_BLOCKS comparison blocks = $TOTAL_JOBS jobs" | tee -a $LOG_FILE
echo "Each job will process up to $PERSONAS_PER_JOB personas evaluating up to $COMPARISONS_PER_JOB comparisons" | tee -a $LOG_FILE
echo "Using model: $MODEL" | tee -a $LOG_FILE
echo "Maximum concurrent jobs: $MAX_CONCURRENT_JOBS" | tee -a $LOG_FILE

# Create necessary directories
mkdir -p jobs
mkdir -p results/chunked_outputs

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
        
        # Cap the end comparison to the total number of comparisons
        if [ $END_COMPARISON -ge $NUM_COMPARISONS ]; then
            END_COMPARISON=$(($NUM_COMPARISONS - 1))
        fi
        
        # Increment job counter
        JOB_COUNTER=$((JOB_COUNTER + 1))
        
        # Create job-specific output directory
        JOB_DIR="jobs/persona_size/p${p}_c${c}"
        mkdir -p $JOB_DIR
        
        # Define output file path
        OUTPUT_FILE="results/persona_size/chunked_outputs/${OUTPUT_NAME}_p${START_PERSONA}-${END_PERSONA}_c${START_COMPARISON}-${END_COMPARISON}.csv"
        
        # Create the job script
        cat > $JOB_DIR/job.sh << EOL
#!/bin/bash
#SBATCH --job-name=${OUTPUT_NAME}_p${p}_c${c}
#SBATCH --output=${JOB_DIR}/${OUTPUT_NAME}_p${p}_c${c}.out
#SBATCH --error=${JOB_DIR}/${OUTPUT_NAME}_p${p}_c${c}.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

# Load required modules
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
source ~/my_llama_python/bin/activate

# Print job information
echo "Processing personas from $START_PERSONA to $END_PERSONA"
echo "Processing comparisons from $START_COMPARISON to $END_COMPARISON"
echo "Output file: $OUTPUT_FILE"

# Run the experiment script with specific persona and comparison ranges
python persona_size/fixed_comparisons_study.py \\
    --output $OUTPUT_FILE \\
    --model $MODEL \\
    --start_index $START_PERSONA \\
    --end_index $END_PERSONA \\
    --start_comparison $START_COMPARISON \\
    --end_comparison $END_COMPARISON

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
COMPLETED=\$(ls -1 results/chunked_outputs/${OUTPUT_NAME}_*.csv 2>/dev/null | wc -l)
echo "Completed job files: \$COMPLETED / $TOTAL_JOBS"
EOF

chmod +x monitor_jobs.sh
echo "Created monitor_jobs.sh script to check progress"