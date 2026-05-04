#!/bin/bash

# Define the script path
PY_SCRIPT="gptoss_finetune_gen_dis.py"

# Check if script exists
if [ ! -f "$PY_SCRIPT" ]; then
    echo "Error: $PY_SCRIPT not found in the current directory."
    exit 1
fi

# Loop through all prompt types
for p in p1 p2 p3 p4; do
    # Loop through shot configurations
    for s in zero-shot 3-shot; do
        echo "------------------------------------------------"
        echo "Running Experiment: Type=$p, Shots=$s"
        echo "------------------------------------------------"
        
        python "$PY_SCRIPT" \
            --p_type "$p" \
            --shots "$s" \
            --max-rows 1000 \
            --output-tag "full_run"
            
        echo "Finished $p $s. Moving to next..."
        echo ""
    done
done

echo "All experiments completed!"
