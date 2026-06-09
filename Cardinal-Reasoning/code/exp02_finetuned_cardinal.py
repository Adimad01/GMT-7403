"""
Experiment 2 — Cardinal-LoRA (no KG)
================================================================================
GPT-OSS-20B fine-tuned on cardinal_train.jsonl (5680 plain question→direction
examples).  No KG evidence at training or inference time.

Model     : openai/gpt-oss-20b + Cardinal-LoRA adapter
KG        : none
Strategies: CoT, ToT, GoT
Eval set  : 80 examples — 10 per direction × 8 directions
Outputs   :
  results/voletc_exp2_card_topo_gpu_{strategy}_cardinal_direction_80_sample_ckpt.json

Run:
    python exp02_finetuned_cardinal.py
"""

import os
import sys
import json
import argparse

DATASET      = "../dataset/cardinal_eval_80.csv"
MODEL_ID     = "openai/gpt-oss-20b"
ADAPTER_PATH = "finetuned_gptoss_cardinal/final_adapter"
MODEL_TAG    = "exp2_card_topo_gpu"
OUTPUT_DIR   = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 512
STRATEGIES = ["cot", "tot", "got"]
SUFFIX     = "cardinal_direction_80_sample"


def preflight():
    ok = True
    if not os.path.exists(DATASET):
        print(f"[ERROR] Eval dataset not found: {DATASET}")
        ok = False
    if not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_model.safetensors")):
        print(f"[ERROR] Cardinal-LoRA adapter not found: {ADAPTER_PATH}")
        print("        Train it first: python train_runner_cardinal.py")
        ok = False
    if not ok:
        sys.exit(1)
    print(f"[OK] Dataset : {DATASET}")
    print(f"[OK] Adapter : {ADAPTER_PATH}")


def check_strategy_status(strategies: list) -> bool:
    print("\n[STATUS] Checkpoint summary:")
    all_done = True
    for strat in strategies:
        ckpt = os.path.join(OUTPUT_DIR, f"voletc_{MODEL_TAG}_{strat}_{SUFFIX}_ckpt.json")
        if os.path.exists(ckpt):
            data    = json.load(open(ckpt))
            results = data.get("results", [])
            done    = len(data.get("processed_indices", []))
            if done >= 80 and results:
                acc = sum(1 for r in results if r.get("match")) / len(results) * 100
                print(f"  {strat.upper():3s} : COMPLETE  ({done}/80, acc={acc:.1f}%)  ✅")
            else:
                print(f"  {strat.upper():3s} : PARTIAL   ({done}/80) — will resume")
                all_done = False
        else:
            print(f"  {strat.upper():3s} : NOT STARTED")
            all_done = False
    return all_done


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=STRATEGIES + ["all"], default="all")
    args = parser.parse_args()
    target = STRATEGIES if args.strategy == "all" else [args.strategy]

    preflight()

    print("\n" + "=" * 70)
    print("  CARDINAL EXPERIMENT 2 — Cardinal-LoRA · CoT/ToT/GoT")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Adapter    : {ADAPTER_PATH}")
    print(f"  Strategies : {', '.join(s.upper() for s in target)}")
    print(f"  Output tag : {MODEL_TAG}")
    print("=" * 70)

    if check_strategy_status(target):
        print("\n[DONE] All strategies complete. Delete checkpoints to re-run.")
        sys.exit(0)

    sys.argv = [
        "eval_engine_cardinal.py",
        "--dataset",        DATASET,
        "--model-id",       MODEL_ID,
        "--adapter-path",   ADAPTER_PATH,
        "--strategy",       args.strategy,
        "--output-dir",     OUTPUT_DIR,
        "--model-tag",      MODEL_TAG,
        "--temperature",    str(TEMPERATURE),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
    ]

    from eval_engine_cardinal import main
    main()


if __name__ == "__main__":
    run()
