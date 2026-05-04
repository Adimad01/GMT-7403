import json
import os
import glob

# Use glob to dynamically find ALL evaluation files in the current directory
# This will catch eval_cot_outputs, eval_cot_no_region_outputs, etc.
jsonl_files = glob.glob('eval_outputs_real/eval_*_outputs.jsonl')

if not jsonl_files:
    print("No evaluation files found matching 'eval_*_outputs.jsonl'")

for file in jsonl_files:
    # Extract a clean variant name for tracking (e.g., 'COT' or 'COT_NO_REGION')
    basename = os.path.basename(file)
    variant = basename.replace('eval_', '').replace('_outputs.jsonl', '').upper()
    
    total = 0
    valid_outputs = 0
    correct = 0
    sum_error_km = 0.0

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            data = json.loads(line)
            
            # Check if prediction and error calculation are valid
            if data.get('pred_norm') is not None and data.get('error_km') is not None:
                valid_outputs += 1
                sum_error_km += data['error_km']
                
                # A prediction is considered correct if the error is within 100km
                if data['error_km'] <= 100.0:
                    correct += 1

    # Calculate final averages
    accuracy_100km = correct / valid_outputs if valid_outputs > 0 else 0
    average_error_km = sum_error_km / valid_outputs if valid_outputs > 0 else 0

    metrics = {
        "variant": variant,
        "accuracy_100km": accuracy_100km,
        "correct": correct,
        "valid_outputs": valid_outputs,
        "total": total,
        "average_error_km": average_error_km
    }

    # Format the new filename
    output_file = basename.replace('eval_', 'metrics_eval_').replace('.jsonl', '.json')
    
    # Save the JSON locally
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Generated {output_file} for variant [{variant}]")