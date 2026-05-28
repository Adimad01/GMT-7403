"""
Experiment 1 — GPTOSS Base
================================================================================
Evaluates the base GPT-OSS-20B model (no fine-tuning, no KG evidence) on the
96 balanced test examples (16 per DE-9IM predicate).

Model     : openai/gpt-oss-20b  (base, no adapter)
KG        : none
Prompt    : vernacular relation + geometry types only (zero-shot)
Eval set  : 96 balanced examples from triplet_update_v3_30.csv
Output    : results/kg_eval_gptoss_base_96_none.csv

Run:
    python exp1_gptoss_base.py
"""

import os
import sys
import json
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATASET        = "../dataset/triplet_update_v3_30.csv"
INDICES_FILE   = "../dataset/eval_96_balanced_indices.json"
MODEL_ID       = "openai/gpt-oss-20b"
ADAPTER_PATH   = None                   # base model — no adapter
KG_SOURCE      = "none"
MODEL_TAG      = "gptoss_base_96"
OUTPUT_DIR     = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 150

OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"kg_eval_{MODEL_TAG}_{KG_SOURCE}.csv")
# ---------------------------------------------------------------------------


def preflight():
    """Verify required files exist before loading the model."""
    ok = True
    for path in [DATASET, INDICES_FILE]:
        if not os.path.exists(path):
            print(f"[ERROR] Required file not found: {path}")
            ok = False
    if not ok:
        sys.exit(1)
    print(f"[OK] Dataset : {DATASET}")
    print(f"[OK] Indices : {INDICES_FILE}")


def check_existing_results():
    """Report progress if a partial result file already exists."""
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        done = len(df)
        print(f"[RESUME] Found {done} rows already evaluated in {OUTPUT_CSV}")
        if done >= 96:
            acc = df["match"].astype(str).str.lower().eq("true").mean()
            print(f"[DONE]   All 96 rows complete. Accuracy = {acc:.2%}")
            print("         Delete the CSV to force a re-run.")
            sys.exit(0)
    else:
        print(f"[NEW]  Starting fresh — output will be saved to {OUTPUT_CSV}")


def run():
    preflight()
    check_existing_results()

    print("\n" + "=" * 70)
    print("  EXPERIMENT 1 — GPTOSS Base (no adapter, no KG)")
    print("=" * 70)
    print(f"  Model  : {MODEL_ID}")
    print(f"  Adapter: none")
    print(f"  KG     : {KG_SOURCE}")
    print(f"  Output : {OUTPUT_CSV}")
    print("=" * 70 + "\n")

    # Delegate to eval_kg_instruction_finetuned.main() via sys.argv injection
    sys.argv = [
        "eval_kg_instruction_finetuned.py",
        "--dataset",        DATASET,
        "--model-id",       MODEL_ID,
        "--kg-source",      KG_SOURCE,
        "--model-tag",      MODEL_TAG,
        "--filter-indices", INDICES_FILE,
        "--output-dir",     OUTPUT_DIR,
        "--temperature",    str(TEMPERATURE),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
    ]

    from eval_kg_instruction_finetuned import main
    main()


if __name__ == "__main__":
    run()
