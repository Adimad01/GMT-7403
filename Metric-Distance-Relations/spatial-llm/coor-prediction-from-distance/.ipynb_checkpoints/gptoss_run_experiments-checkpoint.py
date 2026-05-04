import subprocess
import sys

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT = "gptoss_gen_dis.py"
MODEL_ID = "openai/gpt-oss-20b"   
DATASET = "cities_with_coords.json"
INPUT_IDS = "100_target_ids.json"  # <-- Added the new 100-item list
MAX_ROWS = 1000
# ──────────────────────────────────────────────────────────────────────────────

P_TYPES = ["p1", "p2", "p3", "p4"]
SHOTS = ["zero-shot", "3-shot"]

configs = [(p, s) for p in P_TYPES for s in SHOTS]
total = len(configs)
failed = []

print("=" * 50)
print(f" Running all {total} configurations")
print("=" * 50)

for idx, (p_type, shots) in enumerate(configs, start=1):
    print(f"\n[{idx}/{total}] Starting: --p_type {p_type} --shots {shots}")
    print("-" * 50)

    cmd = [
        sys.executable, SCRIPT,
        "--p_type", p_type,
        "--shots", shots,
        "--dataset", DATASET,
        "--model-id", MODEL_ID, 
        "--input-ids", INPUT_IDS,  # <-- Added this line to force the exact 100 IDs
        "--max-rows", str(MAX_ROWS),
    ]

    try:
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print(f"[{idx}/{total}] ✓ Done: {p_type} / {shots}")
        else:
            print(f"[{idx}/{total}] ✗ FAILED: {p_type} / {shots} (Exit code: {result.returncode})")
            failed.append((p_type, shots))
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"[{idx}/{total}] ✗ ERROR executing script: {e}")
        failed.append((p_type, shots))

print("\n" + "=" * 50)
print(f" Finished {total - len(failed)}/{total} configurations successfully.")
if failed:
    print(" Failed runs:")
    for p, s in failed:
        print(f"   - {p} / {s}")
print(" Results saved in: outputs/")
print("=" * 50)