"""
train_runner_cardinal_nokg.py
================================================================================
No-KG cardinal LoRA for the unified 6-experiment design (Exp 2 / Exp 5 adapter).
Trains on cardinal_nokg_train.csv (corpus → relation_label, no KG evidence).

  Dataset : ../dataset/cardinal_nokg_train.csv   (build_cardinal_train_data.py)
  Output  : finetuned_gptoss_cardinal/final_adapter

NB: supersedes the legacy train_runner_cardinal.py (shore/compass task).

Run:
    python build_cardinal_train_data.py      # build data first
    python train_runner_cardinal_nokg.py
"""
import os
import sys

DATASET    = "../dataset/cardinal_nokg_train.csv"
OUTPUT_DIR = "finetuned_gptoss_cardinal"
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
    print("  FINE-TUNING — Cardinal no-KG LoRA  (unified task)")
    print(f"  dataset={DATASET}  →  {OUTPUT_DIR}/final_adapter")
    print("=" * 70)

    sys.argv = [
        "train_lora_adapter_cardinal.py",
        "--dataset",    DATASET,
        "--run-name",   "cardinal_nokg",
        "--model-id",   MODEL_ID,
        "--output-dir", OUTPUT_DIR,
    ]
    from train_lora_adapter_cardinal import main as train_main
    train_main()


if __name__ == "__main__":
    main()
