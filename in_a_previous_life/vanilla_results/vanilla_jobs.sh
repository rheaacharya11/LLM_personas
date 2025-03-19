#!/bin/bash

TOTAL_CHUNKS=50

for chunk_id in $(seq 0 $((TOTAL_CHUNKS-1))); do
    # Submit a job for each chunk
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fairness_$chunk_id
#SBATCH --output=fairness_$chunk_id.out
#SBATCH --error=fairness_$chunk_id.err
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_requeue

# Load required modules
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
source ~/my_llama_python/bin/activate

python vanilla_query.py \
    --total_comparisons 50000 \
    --chunks 50 \
    --chunk_id $chunk_id \
    --model llama3-8b-instruct
EOF

    echo "Submitted job for chunk $chunk_id"
done