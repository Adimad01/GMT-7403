import subprocess
import sys
import json
import os
import pandas as pd

# 1. Define combinations
shot_types = ['zero-shot', '3-shot']

# Tag for 30% subset runs to isolate checkpoint and outputs
output_tag = "30percentage"
checkpoint_file = f"experiment_checkpoint_{output_tag}.json"

# Dataset configuration
dataset_path = "cities_opt_with_coords.pkl"
subset_percent = 0.30
subset_ids_out = os.path.join("outputs", f"subset_ids_cities_opt_{int(subset_percent*100)}.json")

# Determine available prompt types based on dataset columns
try:
    df_cols = pd.read_pickle(dataset_path).columns
    has_coords = {'a_lat', 'a_lon', 'b_lat', 'b_lon'}.issubset(df_cols)
except Exception:
    has_coords = False

prompt_types = ['p1', 'p2', 'p3', 'p4'] if has_coords else ['p1', 'p3']

# 2. Load or initialize checkpoint
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        completed = json.load(f)
    print(f"--> Found checkpoint. {len(completed)} combinations already done.")
else:
    completed = []
    print("--> Starting fresh experiment.")

# Path to your main processing script
script_to_run = "gen_dis.py"

for p in prompt_types:
    for s in shot_types:
        combo_id = f"{p}_{s}"
        
        # Skip if already in checkpoint
        if combo_id in completed:
            print(f"SKIPPING: {combo_id} (Already completed)")
            continue

        print(f"\n" + "="*50)
        print(f"RUNNING: Type={p}, Shots={s}")
        print("="*50)
        
        cmd = [
            sys.executable, script_to_run,
            "--p_type", p,
            "--shots", s,
            "--dataset", dataset_path,
            "--subset-percent", str(subset_percent),
            "--subset-ids-output", subset_ids_out,
            "--output-tag", output_tag,
        ]
        
        try:
            # Run the experiment
            result = subprocess.run(cmd, check=True)
            
            if result.returncode == 0:
                # 3. Update checkpoint on success
                completed.append(combo_id)
                with open(checkpoint_file, 'w') as f:
                    json.dump(completed, f)
                print(f"SUCCESS: {combo_id} saved to checkpoint.")
                
        except subprocess.CalledProcessError as e:
            print(f"CRITICAL ERROR on {combo_id}: {e}")
            print("Stopping runner to prevent corrupted data. Restart the script to resume.")
            sys.exit(1)

print("\nAll experiments finished successfully!")