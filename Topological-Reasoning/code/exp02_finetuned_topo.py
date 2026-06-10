"""
Experiment 2 — GPTOSS Fine-tuné
================================================================================
GPT-OSS-20B fine-tuned on raw topological data (no KG) evaluated with CoT, ToT,
and GoT reasoning strategies grounded on OSM KG.

The LoRA adapter was trained with SFT on (vernacular, geometry) → predicate
pairs only — no KG evidence was seen during training.  At evaluation, OSM KG
evidence is provided to the CoT/ToT/GoT reasoning strategies.

Model     : openai/gpt-oss-20b + finetuned_gptoss_topological/final_adapter
KG        : OSM (Nominatim)
Strategies: CoT, ToT, GoT
Eval set  : 385 balanced examples — 55 per predicate × 7 predicates
Outputs   :
  results/voletc_exp2_finetuned_topo_gpu_{cot|tot|got}_*_ckpt.json

Run:
    python exp02_finetuned_topo.py
    python exp02_finetuned_topo.py --strategy cot
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
ADAPTER_PATH   = "finetuned_gptoss_topological/final_adapter"
OSM_CACHE      = "results/osm_cache.json"
MODEL_TAG      = "exp2_finetuned_topo_gpu"
OUTPUT_DIR     = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 512

SUFFIX     = "balanced_385"
N_EVAL     = 385
STRATEGIES = ["cot", "tot", "got"]
# ---------------------------------------------------------------------------


def preflight():
    ok = True
    for path in [DATASET, INDICES_FILE, ADAPTER_PATH]:
        if not os.path.exists(path):
            print(f"[ERROR] Required path not found: {path}")
            ok = False
    if not ok:
        sys.exit(1)
    print(f"[OK] Dataset : {DATASET}")
    print(f"[OK] Indices : {INDICES_FILE}")
    print(f"[OK] Adapter : {ADAPTER_PATH}")
    if os.path.exists(OSM_CACHE):
        print(f"[OK] OSM cache: {len(json.load(open(OSM_CACHE)))} entries")
    else:
        print(f"[WARN] OSM cache not found — will query Nominatim live")


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
    print("  EXPERIMENT 2 — GPTOSS Fine-tuné + CoT/ToT/GoT")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Adapter    : {ADAPTER_PATH}")
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
        "--adapter-path",   ADAPTER_PATH,
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
