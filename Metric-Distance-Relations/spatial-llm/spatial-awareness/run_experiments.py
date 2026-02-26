import subprocess
import time
import os
from pathlib import Path

# --- Configuration ---
DATA_FILE = "cities_30.pkl"      # Your input data file
PYTHON_CMD = "python"            # Or "python3"
OUTPUT_DIR = "outputs_gpt_oss_30"   # The folder where main.py saves results

# --- Experiment Parameters ---
P_TYPES = ["near", "far", "close", "and"]
P_LENGTHS = ["zero-shot", "3-shot"]
STATES = ["true", "false"]

def get_expected_filename(p_type, p_length, state):
    """
    Reconstructs the exact filename that main.py creates.
    Based on your file list: gen_sen-{p_length}-{p_type}-{state_str}.json
    """
    state_str = "state" if state == "true" else "no-state"
    # Note: Ensure this order matches exactly how main.py saves it.
    # Looking at your file list: gen_sen-zero-shot-near-no-state.json
    return f"gen_sen-{p_length}-{p_type}-{state_str}.json"

def run_experiment():
    # Ensure output directory exists to avoid errors checking paths
    if not os.path.exists(OUTPUT_DIR):
        print(f"[*] Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    total_experiments = len(P_TYPES) * len(P_LENGTHS) * len(STATES)
    current_count = 0
    skipped_count = 0
    run_count = 0

    print(f"[*] Starting batch execution check for {total_experiments} configurations...")
    print(f"[*] Looking for existing files in: {os.path.abspath(OUTPUT_DIR)}")

    for p_type in P_TYPES:
        for p_length in P_LENGTHS:
            for state in STATES:
                current_count += 1
                
                # 1. Calculate the Target File Path
                filename = get_expected_filename(p_type, p_length, state)
                filepath = os.path.join(OUTPUT_DIR, filename)

                print(f"\n==================================================")
                print(f"[*] Experiment {current_count}/{total_experiments}")
                print(f"[*] Config: Type={p_type}, Length={p_length}, State={state}")
                
                # 2. THE CHECKPOINT: Does the file exist?
                if os.path.exists(filepath):
                    # Optional: Check if file is empty or corrupted
                    if os.path.getsize(filepath) > 100: # Assuming a valid JSON is > 100 bytes
                        print(f"[✓] SKIPPED: File already exists ({filename})")
                        skipped_count += 1
                        continue
                    else:
                        print(f"[!] Found empty/corrupt file. Re-running...")
                
                # 3. If we are here, we need to run it
                print(f"[*] Status: Missing or incomplete. executing...")
                run_count += 1
                
                cmd = [
                    PYTHON_CMD, "main.py",
                    "--p_type", p_type,
                    "--p_length", p_length,
                    "--state", state,
                    "--data", DATA_FILE,
                    "--out_dir", OUTPUT_DIR
                ]

                try:
                    # Run the command
                    subprocess.run(cmd, check=True)
                    print(f"[*] Success: Experiment finished.")
                    
                    # Sleep briefly to allow file system I/O to settle and API to cool down
                    time.sleep(2)
                    
                except subprocess.CalledProcessError as e:
                    print(f"[!] FAILED: Experiment crashed with error: {e}")
                    # We continue to the next loop even if one fails
                except KeyboardInterrupt:
                    print("\n[!] Batch execution interrupted by user.")
                    return

    print(f"\n==================================================")
    print(f"[*] Batch Completed.")
    print(f"[*] Total Scanned: {total_experiments}")
    print(f"[*] Skipped (Already Done): {skipped_count}")
    print(f"[*] Newly Executed: {run_count}")

if __name__ == "__main__":
    run_experiment()