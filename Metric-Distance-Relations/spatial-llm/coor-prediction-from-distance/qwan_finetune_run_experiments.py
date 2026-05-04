import subprocess
import sys
import json
import os
import pickle
import pandas as pd


# --- CUSTOM UNPICKLER FOR NUMPY 2.0 -> 1.x COMPATIBILITY ---
class Numpy2to1Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_pickle(path):
    with open(path, 'rb') as f:
        return Numpy2to1Unpickler(f).load()
# ------------------------------------------------------------


# 1. Define combinations
shot_types = ['zero-shot', '3-shot']

# Tag clearly marks these as fine-tuned results
output_tag      = "qwen14b_finetuned_30percentage"
checkpoint_file = f"experiment_checkpoint_{output_tag}.json"

# Dataset configuration — same 30% split used for base-model experiments
dataset_path    = "cities_with_coords.pkl"
subset_percent  = 0.30
subset_ids_out  = os.path.join("outputs", f"subset_ids_cities_opt_{output_tag}.json")

# Path to the LoRA adapter produced by qwan_finetune.py
adapter_path    = "finetuned_qwen14b/final_adapter"

# Determine available prompt types
try:
    df_cols    = load_pickle(dataset_path).columns
    has_coords = {'a_lat', 'a_lon', 'b_lat', 'b_lon'}.issubset(df_cols)
    print(f"--> Dataset loaded. Columns: {list(df_cols)}")
    print(f"--> Coordinates available: {has_coords}")
except Exception as e:
    print(f"--> Warning: Could not inspect dataset columns: {e}")
    has_coords = False

prompt_types = ['p1', 'p2', 'p3', 'p4'] if has_coords else ['p1', 'p3']
print(f"--> Prompt types to run: {prompt_types}")

# 2. Load or initialize checkpoint
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        completed = json.load(f)
    print(f"--> Found checkpoint. {len(completed)} combinations already done: {completed}")
else:
    completed = []
    print("--> Starting fresh experiment.")

os.makedirs("outputs", exist_ok=True)

# ✅ Points to the fine-tuned inference script
script_to_run = "qwan_finetune_gen_dis.py"

for p in prompt_types:
    for s in shot_types:
        combo_id = f"{p}_{s}"

        if combo_id in completed:
            print(f"\nSKIPPING: {combo_id} (Already completed)")
            continue

        print(f"\n{'='*50}")
        print(f"RUNNING (fine-tuned): Type={p}, Shots={s}")
        print(f"{'='*50}")

        cmd = [
            sys.executable, script_to_run,
            "--p_type",            p,
            "--shots",             s,
            "--dataset",           dataset_path,
            "--adapter-path",      adapter_path,
            "--subset-percent",    str(subset_percent),
            "--subset-ids-output", subset_ids_out,
            "--output-tag",        output_tag,
        ]

        try:
            result = subprocess.run(cmd, check=True)

            if result.returncode == 0:
                completed.append(combo_id)
                with open(checkpoint_file, 'w') as f:
                    json.dump(completed, f, indent=2)
                print(f"SUCCESS: {combo_id} saved to checkpoint.")

        except subprocess.CalledProcessError as e:
            print(f"\nCRITICAL ERROR on {combo_id}: {e}")
            print("Stopping runner to prevent corrupted data. Restart the script to resume.")
            sys.exit(1)

print("\nAll fine-tuned experiments finished successfully!")