#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-00:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=8G            # Adjust memory based on model size
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# Load required modules
module load python/3.10.9-fasrc01
source activate my_llama_env  # Ensure you have a virtual environment with required packages

# Run the Python script
python run_llama.py
