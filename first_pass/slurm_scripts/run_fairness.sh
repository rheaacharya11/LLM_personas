#!/bin/bash
#SBATCH --job-name=fairness_elicitation
#SBATCH --output=fairness_output_%j.log
#SBATCH --error=fairness_error_%j.log
#SBATCH --time=10:00:00 # Set the time limit for the job
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --mem=32G # Allocate 32GB of memory
#SBATCH --cpus-per-task=4 # Allocate 4 CPUs
#SBATCH --partition=gpu # Specify the GPU partition

# Load any required modules
module load python/3.8

# Activate your existing Python environment
source ~/my_llama_python/bin/activate

# Install required packages (if needed)
# pip install pandas numpy scikit-learn requests

# Set environment variables
export PYTHONUNBUFFERED=1

# Define paths
COMPAS_URL="https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
STAKEHOLDER_FILE="cleaned_results/combined_similar.csv"
OUTPUT_DIR="results_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the script with added logging
echo "Starting fairness elicitation job at $(date)"
echo "Stakeholder file: $STAKEHOLDER_FILE"
echo "Output directory: $OUTPUT_DIR"

# Modify your script to include these arguments
python -u << EOF
import sys
import os
import time
from datetime import datetime

# Add the directory containing your script to the Python path
sys.path.append("$(dirname $(realpath $0))")

# Import your module
from paste import run_fairness_elicitation

# Log start time
start_time = time.time()
print(f"Job started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Processing stakeholder file: $STAKEHOLDER_FILE")

# Run your function
try:
    fairness, D, X_test, y_test, test_idx = run_fairness_elicitation(
        "$COMPAS_URL", 
        "$STAKEHOLDER_FILE"
    )
    
    # Evaluate on test set (add this if needed)
    test_accuracy = fairness.evaluate_accuracy(D, X_test, y_test)
    test_violation, test_violations_by_pair = fairness.evaluate_fairness_violation(D, X_test, y_test)
    
    print(f"\nTest Set Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Fairness Violation: {test_violation:.4f}")
    
    # Save results
    import pickle
    with open(os.path.join("$OUTPUT_DIR", "fairness_model.pkl"), "wb") as f:
        pickle.dump({"fairness": fairness, "D": D}, f)
        
    print(f"Results saved to {os.path.join('$OUTPUT_DIR', 'fairness_model.pkl')}")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

# Log end time
end_time = time.time()
print(f"Job completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {(end_time - start_time) / 60:.2f} minutes")
EOF

echo "Job completed at $(date)"