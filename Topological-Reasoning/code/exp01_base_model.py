"""
Experiment 1 — GPTOSS Base
================================================================================
Base GPT-OSS-20B (no fine-tuning, no adapter) evaluated on 385 balanced test
examples using CoT, ToT, and GoT reasoning strategies grounded on OSM KG.

Model     : openai/gpt-oss-20b  (base, no adapter)
KG        : OSM (Nominatim) — fetched dynamically, cached in osm_cache.json
Strategies: CoT, ToT, GoT  (all three run sequentially)
Eval set  : 385 balanced examples — 55 per DE-9IM predicate × 7 predicates
Outputs   :
  results/voletc_exp1_base_gpu_cot_balanced_385_ckpt.json
  results/voletc_exp1_base_gpu_tot_balanced_385_ckpt.json
  results/voletc_exp1_base_gpu_got_balanced_385_ckpt.json

Run:
    python exp01_base_model.py
    python exp01_base_model.py --strategy cot   # single strategy
"""

import os
import sys
import json
import argparse

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATASET        = "../dataset/topological_relations.csv"
INDICES_FILE   = "../dataset/eval_385_balanced_indices.json"
MODEL_ID       = "openai/gpt-oss-20b"
ADAPTER_PATH   = None                   # base model — no adapter
OSM_CACHE      = "results/osm_cache.json"
MODEL_TAG      = "exp1_base_gpu"
OUTPUT_DIR     = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 512

SUFFIX     = "balanced_385"
N_EVAL     = 385
STRATEGIES = ["cot", "tot", "got"]
# ---------------------------------------------------------------------------


def preflight():
    ok = True
    for path in [DATASET, INDICES_FILE]:
        if not os.path.exists(path):
            print(f"[ERROR] Required file not found: {path}")
            ok = False
    if not ok:
        sys.exit(1)
    print(f"[OK] Dataset : {DATASET}")
    print(f"[OK] Indices : {INDICES_FILE}")
    if os.path.exists(OSM_CACHE):
        cache = json.load(open(OSM_CACHE))
        print(f"[OK] OSM cache: {len(cache)} entries ({OSM_CACHE})")
    else:
        print(f"[WARN] OSM cache not found — will query Nominatim live (slower)")


def check_strategy_status(strategies: list) -> bool:
    print("\n[STATUS] Checkpoint summary:")
    all_done = True
    for strat in strategies:
        ckpt = os.path.join(OUTPUT_DIR, f"voletc_{MODEL_TAG}_{strat}_{SUFFIX}_ckpt.json")
        if os.path.exists(ckpt):
            data   = json.load(open(ckpt))
            done    = len(data.get("processed_indices", []))
            results = data.get("results", [])
            if done >= N_EVAL and results:
                acc = sum(1 for r in results if r.get("match")) / len(results) * 100
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
    print("  EXPERIMENT 1 — GPTOSS Base + CoT/ToT/GoT")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Adapter    : none (base model)")
    print(f"  Strategies : {', '.join(s.upper() for s in target)}")
    print(f"  Output tag : {MODEL_TAG}")
    print("=" * 70)

    if check_strategy_status(target):
        print("\n[DONE] All strategies complete. Delete checkpoints to re-run.")
        sys.exit(0)

    print()
    sys.argv = [
        "eval_engine_gpu.py",
        "--dataset",        DATASET,
        "--model-id",       MODEL_ID,
        "--filter-indices", INDICES_FILE,
        "--strategy",       args.strategy,
        "--output-dir",     OUTPUT_DIR,
        "--model-tag",      MODEL_TAG,
        "--temperature",    str(TEMPERATURE),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
    ]

    from eval_engine_gpu import main
    main()


if __name__ == "__main__":
    run()
