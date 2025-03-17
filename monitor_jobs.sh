#!/bin/bash
# Check status of all submitted jobs
squeue -u $USER | grep "fixed_compas_study"

# Count completed and remaining jobs
echo ""
echo "Progress summary:"
COMPLETED=$(ls -1 results/persona_size/chunked_outputs/fixed_compas_study_*.csv 2>/dev/null | wc -l)
echo "Completed job files: $COMPLETED / 200"
