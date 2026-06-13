"""
Experiment 1 — Base GPT-OSS-20B (Cardinal)
================================================================================
Base GPT-OSS-20B (no fine-tuning, no adapter) evaluated on 440 balanced cardinal
direction examples using CoT, ToT, and GoT reasoning strategies.

Model     : openai/gpt-oss-20b  (base, no adapter)
KG        : none
Strategies: CoT, ToT, GoT
Eval set  : 440 examples — 55 per direction × 8 directions
Outputs   :
  results/voletc_exp1_card_base_gpu_cot_cardinal_direction_440_sample_ckpt.json
  results/voletc_exp1_card_base_gpu_tot_cardinal_direction_440_sample_ckpt.json
  results/voletc_exp1_card_base_gpu_got_cardinal_direction_440_sample_ckpt.json

Run:
    python exp01_base_cardinal.py
    python exp01_base_cardinal.py --strategy cot
"""

import os
import sys
import json
import argparse

DATASET      = "../dataset/cardinal_direction_relations.csv"
INDICES_FILE = "../dataset/eval_32_balanced_indices.json"
MODEL_ID     = "openai/gpt-oss-20b"
ADAPTER_PATH = None
MODEL_TAG    = "exp1_card_base_gpu"
OUTPUT_DIR   = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 512
N_EVAL     = 32
STRATEGIES = ["cot", "tot", "got"]
SUFFIX     = "cardinal_dir_32_sample"


def preflight():
    if not os.path.exists(DATASET):
        print(f"[ERROR] Eval dataset not found: {DATASET}")
        sys.exit(1)
    print(f"[OK] Dataset : {DATASET}")


def check_strategy_status(strategies: list) -> bool:
    print("\n[STATUS] Checkpoint summary:")
    all_done = True
    for strat in strategies:
        ckpt = os.path.join(OUTPUT_DIR, f"voletc_{MODEL_TAG}_{strat}_{SUFFIX}_ckpt.json")
        if os.path.exists(ckpt):
            data    = json.load(open(ckpt))
            results = data.get("results", [])
            done    = len(data.get("processed_indices", []))
            hits = sum(1 for r in results if r.get("match"))
            acc  = hits / len(results) * 100 if results else 0.0
            if results and done <= 5 and hits == 0:
                print(f"  {strat.upper():3s} : STALE ({done} rows, acc=0.0%) — auto-deleting ⚠️")
                os.remove(ckpt)
                all_done = False
            elif done >= N_EVAL and results:
                print(f"  {strat.upper():3s} : COMPLETE  ({done}/{N_EVAL}, acc={acc:.1f}%)  ✅")
            else:
                print(f"  {strat.upper():3s} : PARTIAL   ({done}/{N_EVAL}) — will resume")
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
    print("  CARDINAL EXPERIMENT 1 — Base GPT-OSS-20B · CoT/ToT/GoT")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Adapter    : none (base model)")
    print(f"  Strategies : {', '.join(s.upper() for s in target)}")
    print(f"  Output tag : {MODEL_TAG}")
    print("=" * 70)

    if check_strategy_status(target):
        print("\n[DONE] All strategies complete. Delete checkpoints to re-run.")
        sys.exit(0)

    sys.argv = [
        "eval_engine_cardinal.py",
        "--dataset",         DATASET,
        "--filter-indices",  INDICES_FILE,
        "--model-id",        MODEL_ID,
        "--strategy",        args.strategy,
        "--output-dir",      OUTPUT_DIR,
        "--model-tag",       MODEL_TAG,
        "--temperature",     str(TEMPERATURE),
        "--max-new-tokens",  str(MAX_NEW_TOKENS),
    ]

    from eval_engine_cardinal import main
    main()


if __name__ == "__main__":
    run()
