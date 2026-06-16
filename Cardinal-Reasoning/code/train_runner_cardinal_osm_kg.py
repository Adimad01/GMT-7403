"""
train_runner_cardinal_osm_kg.py
================================================================================
OSM-KG cardinal LoRA for the unified 6-experiment design (Exp 3 adapter).
Trains on cardinal_osm_kg_train.jsonl — each example embeds OpenStreetMap
evidence (coordinates / bearing / offset) so the geographic facts are baked
into the adapter weights.  Evaluated WITHOUT KG at inference (kg-mode none).

  Dataset : ../dataset/cardinal_osm_kg_train.jsonl  (build_cardinal_train_data.py)
  Output  : finetuned_gptoss_cardinal_osm_kg/final_adapter

Run:
    python build_cardinal_train_data.py            # build data (locally, warms OSM)
    python train_runner_cardinal_osm_kg.py
"""
import os
import sys

DATASET    = "../dataset/cardinal_osm_kg_train.jsonl"
OUTPUT_DIR = "finetuned_gptoss_cardinal_osm_kg"
MODEL_ID   = "openai/gpt-oss-20b"


def main():
    final = os.path.join(OUTPUT_DIR, "final_adapter", "adapter_model.safetensors")
    if os.path.exists(final):
        print(f"[DONE] Adapter already exists: {final} (delete dir to re-run)")
        return
    if not os.path.exists(DATASET):
        print(f"[ERROR] Training data not found: {DATASET}")
        print("        Build it first: python build_cardinal_train_data.py")
        sys.exit(1)

    print("=" * 70)
    print("  FINE-TUNING — Cardinal OSM-KG LoRA  (KG @ training)")
    print(f"  dataset={DATASET}  →  {OUTPUT_DIR}/final_adapter")
    print("=" * 70)

    sys.argv = [
        "train_lora_adapter_cardinal.py",
        "--dataset",    DATASET,
        "--run-name",   "cardinal_osm_kg",
        "--model-id",   MODEL_ID,
        "--output-dir", OUTPUT_DIR,
    ]
    from train_lora_adapter_cardinal import main as train_main
    train_main()


if __name__ == "__main__":
    main()
