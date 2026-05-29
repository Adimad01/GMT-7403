"""
Fine-tuning 1 — GPTOSS Fine-tuné (raw topological data)
================================================================================
Fine-tunes GPT-OSS-20B with LoRA on the raw topological dataset
(triplet_update_v3_70.csv — 756 training rows).

No KG evidence is included in the training prompts.  The model learns
purely from (vernacular relation, geometry types) → predicate pairs.

STATUS: ✅ ALREADY DONE — adapter exists at finetuned_gptoss_topological/final_adapter
This script is kept for reproducibility.  It will skip automatically if the
final_adapter directory already exists.

Used by: Experiments 2, 3, 5

Output: finetuned_gptoss_topological/final_adapter

Run:
    python ft1_topological.py
"""

import os
import sys

# ---------------------------------------------------------------------------
DATASET     = "../dataset/triplet_update_v3_70.csv"
OUTPUT_DIR  = "finetuned_gptoss_topological"
MODEL_ID    = "openai/gpt-oss-20b"
# ---------------------------------------------------------------------------


def check_done() -> bool:
    final = os.path.join(OUTPUT_DIR, "final_adapter", "adapter_model.safetensors")
    if os.path.exists(final):
        print(f"[DONE] ✅  Adapter already exists: {final}")
        print("         Delete the directory to re-run fine-tuning.")
        return True
    return False


def preflight():
    if not os.path.exists(DATASET):
        print(f"[ERROR] Training dataset not found: {DATASET}")
        sys.exit(1)
    print(f"[OK] Dataset: {DATASET}  ({sum(1 for _ in open(DATASET))-1} rows)")


def run():
    print("\n" + "=" * 70)
    print("  FINE-TUNING 1 — Raw Topological Data (no KG)")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Dataset    : {DATASET}")
    print(f"  Output dir : {OUTPUT_DIR}/final_adapter")
    print("=" * 70 + "\n")

    if check_done():
        sys.exit(0)

    preflight()

    sys.argv = [
        "finetune_gptoss.py",
        "--dataset",    DATASET,
        "--model-id",   MODEL_ID,
        "--output-dir", OUTPUT_DIR,
    ]

    from finetune_gptoss import main
    main()


if __name__ == "__main__":
    run()
