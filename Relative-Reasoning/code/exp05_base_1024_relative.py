"""
Experiment 5 — Base model + extended reasoning budget (ablation, no fine-tuning)
================================================================================
Base GPT-OSS-20B (no adapter) with 1024 max tokens.
Ablation: does more reasoning budget alone improve navigation without fine-tuning?

Model     : openai/gpt-oss-20b  (base, no adapter)
Strategies: CoT, ToT, GoT
Eval set  : 270 examples (ring / square / tree navigation)
Max tokens: 1024  (extended budget)
Outputs   :
  results/voletc_exp5_rel_base_ext_gpu_{strategy}_relative_nav_270_sample_ckpt.json

Run:
    python exp05_base_1024_relative.py
"""

import os
import sys
import json
import argparse

DATASET      = "../dataset/relative_eval.jsonl"
MODEL_ID     = "openai/gpt-oss-20b"
ADAPTER_PATH = None
MODEL_TAG    = "exp5_rel_base_ext_gpu"
OUTPUT_DIR   = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 1024
N_EVAL     = 270
STRATEGIES = ["cot", "tot", "got"]
SUFFIX     = "relative_nav_270_sample"


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
            if results and done <= 80 and hits == 0:
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
    print("  RELATIVE EXPERIMENT 5 — Base GPT-OSS-20B + extended budget (no FT)")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Adapter    : none (base model)")
    print(f"  Max tokens : {MAX_NEW_TOKENS}  (extended budget)")
    print(f"  Strategies : {', '.join(s.upper() for s in target)}")
    print(f"  Output tag : {MODEL_TAG}")
    print("=" * 70)

    if check_strategy_status(target):
        print("\n[DONE] All strategies complete. Delete checkpoints to re-run.")
        sys.exit(0)

    sys.argv = [
        "eval_engine_relative.py",
        "--dataset",        DATASET,
        "--model-id",       MODEL_ID,
        "--strategy",       args.strategy,
        "--output-dir",     OUTPUT_DIR,
        "--model-tag",      MODEL_TAG,
        "--temperature",    str(TEMPERATURE),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
    ]

    from eval_engine_relative import main
    main()


if __name__ == "__main__":
    run()
