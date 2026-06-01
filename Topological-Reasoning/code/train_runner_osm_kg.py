"""
Fine-tuning 4 — GPTOSS Fine-tuné sur KG OSM (instruction tuning, balanced)
================================================================================
Fine-tunes GPT-OSS-20B with LoRA on the BALANCED OSM KG instruction dataset
(osm_kg_balanced_train.jsonl — 234 rows, 39 per DE-9IM predicate).

Original dataset (osm_kg_train.jsonl, 754 rows) was imbalanced:
  touches 223, within 186, disjoint 183, overlaps 83, crosses 40, contains 39.
The balanced version keeps 39 examples per predicate (minimum class count).

Each training example includes a structured KG context (coordinates, bounding
box, admin hierarchy) so the model learns to ground topological reasoning on
real geographic evidence.

Generate balanced dataset with:
    python build_balanced_training_data.py

Output: finetuned_gptoss_osm_kg/final_adapter

Run:
    python train_runner_osm_kg.py
"""

import os
import sys

# ---------------------------------------------------------------------------
DATASET     = "../dataset/osm_kg_balanced_train.jsonl"
OUTPUT_DIR  = "finetuned_gptoss_osm_kg"
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
        print("        Build it first with: python build_kg_instruction_dataset_osm.py")
        sys.exit(1)
    print(f"[OK] Dataset: {DATASET}  ({sum(1 for _ in open(DATASET))} lines)")


def run():
    print("\n" + "=" * 70)
    print("  FINE-TUNING 4 — OSM KG Instruction Dataset")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Dataset    : {DATASET}")
    print(f"  Output dir : {OUTPUT_DIR}/final_adapter")
    print("=" * 70 + "\n")

    if check_done():
        sys.exit(0)

    preflight()

    sys.argv = [
        "train_lora_adapter_kg.py",
        "--dataset",    DATASET,
        "--model-id",   MODEL_ID,
        "--output-dir", OUTPUT_DIR,
    ]

    from train_lora_adapter_kg import main
    main()


if __name__ == "__main__":
    run()
