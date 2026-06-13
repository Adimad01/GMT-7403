"""
Experiment 4 — Wikidata-KG LoRA (cross-task transfer)
================================================================================
Uses the Wikidata-KG adapter trained on topological data (DE-9IM) and evaluates
it on cardinal direction questions.  Tests whether topological spatial reasoning
knowledge transfers to directional reasoning.

The adapter lives in: ../../Topological-Reasoning/code/finetuned_gptoss_wikidata_kg/final_adapter

Model     : openai/gpt-oss-20b + Wikidata-KG LoRA (cross-task)
KG        : CardinalKnowledgeGraph (compass rules at inference)
Strategies: CoT, ToT, GoT
Eval set  : 80 examples — 10 per direction × 8 directions
Max tokens: 1024
Outputs   :
  results/voletc_exp4_card_wikidata_gpu_{strategy}_cardinal_direction_80_sample_ckpt.json

Run:
    python exp04_wikidata_cardinal.py
"""

import os
import sys
import json
import argparse

DATASET      = "../dataset/cardinal_direction_relations.csv"
INDICES_FILE = "../dataset/eval_32_balanced_indices.json"
MODEL_ID     = "openai/gpt-oss-20b"
ADAPTER_PATH = "../../Topological-Reasoning/code/finetuned_gptoss_wikidata_kg/final_adapter"
MODEL_TAG    = "exp4_card_wikidata_gpu"
OUTPUT_DIR   = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 1024
N_EVAL     = 32
STRATEGIES = ["cot", "tot", "got"]
SUFFIX     = "cardinal_dir_32_sample"


def preflight():
    ok = True
    if not os.path.exists(DATASET):
        print(f"[ERROR] Eval dataset not found: {DATASET}")
        ok = False
    adapter_check = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
    if not os.path.exists(adapter_check):
        print(f"[ERROR] Wikidata-KG adapter not found: {ADAPTER_PATH}")
        print("        Train it in Topological-Reasoning/code/:")
        print("          python train_runner_wikidata_kg.py")
        ok = False
    if not ok:
        sys.exit(1)
    print(f"[OK] Dataset : {DATASET}")
    print(f"[OK] Adapter : {ADAPTER_PATH}  (Wikidata-KG, cross-task)")


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
    print("  CARDINAL EXPERIMENT 4 — Wikidata-KG LoRA (cross-task) + KG")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Adapter    : {ADAPTER_PATH}  (trained on DE-9IM Wikidata data)")
    print(f"  KG         : CardinalKnowledgeGraph (compass rules at inference)")
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
        "--adapter-path",    ADAPTER_PATH,
        "--use-kg",
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
