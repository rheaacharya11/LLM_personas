#!/bin/bash
# Check status of all submitted jobs
squeue -u $USER | grep "double_query_study"

# Count completed and remaining jobs
echo ""
echo "Progress summary:"
COMPLETED=$(ls -1 ../results/cot/chunked_outputs/double_query_study_*.csv 2>/dev/null | wc -l)
echo "Completed job files: $COMPLETED / 100"
