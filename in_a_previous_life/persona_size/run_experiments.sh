#!/bin/bash

# Usage information
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <output_name> <num_personas> <personas_per_job> [comparisons_per_persona=100] [model=llama3-8b-instruct]"
    echo ""
    echo "Example: $0 fixed_compas_study 1000 50 100 llama3-8b-instruct"
    echo "This will process 1000 personas with 50 personas per job, 100 comparisons each, using llama3-8b-instruct"
    exit 1
fi

# Get parameters from command line
OUTPUT_NAME=$1
TOTAL_PERSONAS=$2
PERSONAS_PER_JOB=$3
COMPARISONS_PER_PERSONA=${4:-100}  # Default to 100 if not specified
MODEL=${5:-llama3-8b-instruct}     # Default to llama3-8b-instruct if not specified

# Calculate number of jobs needed
NUM_JOBS=$(( ($TOTAL_PERSONAS + $PERSONAS_PER_JOB - 1) / $PERSONAS_PER_JOB ))

echo "Launching $NUM_JOBS SLURM jobs to process $TOTAL_PERSONAS personas"
echo "Each job will process $PERSONAS_PER_JOB personas, each evaluating $COMPARISONS_PER_PERSONA comparisons"
echo "Using model: $MODEL"
echo "Output files will be named: results/chunked_outputs/${OUTPUT_NAME}_<start>_to_<end>.csv"

# Create necessary directories
mkdir -p jobs
mkdir -p results/chunked_outputs

# Generate fixed comparisons first if they don't exist
if [ ! -f "data/fixed_comparisons.json" ]; then
    echo "Generating fixed comparisons file..."
    python generate_fixed_comparisons.py --comparisons $COMPARISONS_PER_PERSONA
fi

# Loop to submit all jobs
for ((i=0; i<$NUM_JOBS; i++)); do
    # Calculate start and end indices for this job
    START_INDEX=$(($i * $PERSONAS_PER_JOB))
    END_INDEX=$((($i + 1) * $PERSONAS_PER_JOB - 1))
    
    # Cap the end index to the total number of personas
    if [ $END_INDEX -ge $TOTAL_PERSONAS ]; then
        END_INDEX=$(($TOTAL_PERSONAS - 1))
    fi
    
    # Create job-specific output directory
    mkdir -p jobs/persona_size/${i}
    
    # Define output file path
OUTPUT_FILE="results/persona_size/chunked_outputs/${OUTPUT_NAME}_p${START_PERSONA}-${END_PERSONA}_c${START_COMPARISON}-${END_COMPARISON}.csv"
    
    # Create the job script
    cat > jobs/persona_size/${i}/job_${i}.sh << EOL
#!/bin/bash
#SBATCH --job-name=${OUTPUT_NAME}_${i}
#SBATCH --output=jobs/persona_size/${i}/${OUTPUT_NAME}_${i}.out
#SBATCH --error=jobs/persona_size/${i}/${OUTPUT_NAME}_${i}.err
#SBATCH --time=20:00:00
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
echo "Processing personas from $START_INDEX to $END_INDEX"
echo "Output file: $OUTPUT_FILE"

# Run the experiment script
python fixed_comparisons_study.py \\
    --output $OUTPUT_FILE \\
    --model $MODEL \\
    --comparisons $COMPARISONS_PER_PERSONA \\
    --start_index $START_INDEX \\
    --end_index $END_INDEX

echo "Processing complete for personas $START_INDEX to $END_INDEX"
EOL
    
    # Submit the job
    sbatch jobs/persona_size/${i}/job_${i}.sh
    
    echo "Submitted job $i for personas $START_INDEX to $END_INDEX"
done

echo "All $NUM_JOBS jobs submitted!"