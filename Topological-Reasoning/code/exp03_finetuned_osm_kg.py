"""
Experiment 3 — GPTOSS Fine-tuné KG-OSM + Inférence enrichie OSM
================================================================================
GPT-OSS-20B fine-tuned on OSM-KG instruction data (osm_kg_train.jsonl), then
evaluated with CoT/ToT/GoT reasoning strategies grounded on OSM KG.

Key distinction from Experiment 2: the adapter was trained on instruction
examples that ALREADY contained OSM KG evidence (coordinates, bounding boxes,
administrative hierarchy).  At inference, the same OSM evidence is provided
via CoT/ToT/GoT — so the model has seen this kind of evidence both during
fine-tuning AND at evaluation.

Model     : openai/gpt-oss-20b + finetuned_gptoss_osm_kg/final_adapter
KG        : OSM (Nominatim) — in training AND at inference via CoT/ToT/GoT
Strategies: CoT, ToT, GoT
Eval set  : 96 balanced examples — 16 per predicate
Outputs   :
  results/voletc_exp3_finetuned_kg_in_gpu_{cot|tot|got}_*_ckpt.json

Run:
    python exp03_finetuned_osm_kg.py
    python exp03_finetuned_osm_kg.py --strategy cot
"""

import os
import sys
import json
import argparse

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATASET        = "../dataset/topological_relations.csv"
INDICES_FILE   = "../dataset/eval_112_balanced_indices.json"
MODEL_ID       = "openai/gpt-oss-20b"
ADAPTER_PATH   = "finetuned_gptoss_osm_kg/final_adapter"
OSM_CACHE      = "results/osm_cache.json"
MODEL_TAG      = "exp3_finetuned_kg_in_gpu"
OUTPUT_DIR     = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 1024

SUFFIX     = "neighborhood_details_spatial_relation_16_sample"
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
            if done >= 96 and results:
                acc = sum(1 for r in results if r.get("match")) / len(results) * 100
                print(f"  {strat.upper():3s} : COMPLETE  ({done}/96, acc={acc:.1f}%)  ✅")
            else:
                print(f"  {strat.upper():3s} : PARTIAL   ({done}/96) — will resume")
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
    print("  EXPERIMENT 3 — GPTOSS FT OSM-KG + Inférence enrichie OSM")
    print("  (OSM-KG adapter, OSM KG in training AND at inference)")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Adapter    : {ADAPTER_PATH}  [trained WITH OSM KG evidence]")
    print(f"  KG         : OSM evidence in training + CoT/ToT/GoT at inference")
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
