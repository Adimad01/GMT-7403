"""
Experiment 2 — Relative-LoRA (fine-tuned on relative_balanced_train.csv)
================================================================================
GPT-OSS-20B fine-tuned on 55 relative direction examples (11/class × 5 classes).
No extended token budget.

Model     : openai/gpt-oss-20b + Relative-LoRA adapter
Strategies: CoT, ToT, GoT
Eval set  : 20 examples (4/class × 5 classes, balanced across ambiguity levels)
Max tokens: 512
Outputs   :
  results/voletc_exp2_rel_lora_gpu_{strategy}_relative_dir_20_sample_ckpt.json

Run:
    python exp02_finetuned_relative.py
"""

import os
import sys
import json
import argparse

DATASET      = "../../Topological-Reasoning/dataset/relative_direction_relations.csv"
INDICES_FILE = "../dataset/eval_20_balanced_indices.json"
MODEL_ID     = "openai/gpt-oss-20b"
ADAPTER_PATH = "finetuned_gptoss_relative/final_adapter"
MODEL_TAG    = "exp2_rel_lora_gpu"
OUTPUT_DIR   = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 512
N_EVAL     = 20
STRATEGIES = ["cot", "tot", "got"]
SUFFIX     = "relative_dir_20_sample"


def preflight():
    ok = True
    if not os.path.exists(DATASET):
        print(f"[ERROR] Eval dataset not found: {DATASET}")
        ok = False
    if not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_model.safetensors")):
        print(f"[ERROR] Relative-LoRA adapter not found: {ADAPTER_PATH}")
        print("        Train it first: python train_runner_relative.py")
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
            hits = sum(1 for r in results if r.get("match"))
            acc  = hits / len(results) * 100 if results else 0.0
            if results and done <= 3 and hits == 0:
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
    print("  RELATIVE EXPERIMENT 2 — Relative-LoRA · CoT/ToT/GoT")
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
        "eval_engine_relative.py",
        "--dataset",        DATASET,
        "--filter-indices", INDICES_FILE,
        "--model-id",       MODEL_ID,
        "--adapter-path",   ADAPTER_PATH,
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
