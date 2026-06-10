"""
Experiment 4 — GPTOSS OSM-KG LoRA sans KG à l'inférence
================================================================================
GPT-OSS-20B fine-tuned on OSM-KG instruction data (same adapter as Exp 3),
evaluated with CoT/ToT/GoT reasoning strategies WITHOUT OSM KG evidence at
inference time.

Key distinction from Experiment 3: both use the OSM-KG LoRA adapter, but
  Exp 3 — KG in training AND at inference  (KG enriches both phases)
  Exp 4 — KG in training ONLY              (ablates the inference-time KG)

This isolates the contribution of KG-enriched fine-tuning alone, independent
of KG evidence at inference.

Model     : openai/gpt-oss-20b + finetuned_gptoss_osm_kg/final_adapter
KG        : OSM in training only — NO KG at inference (--no-kg)
Strategies: CoT, ToT, GoT
Eval set  : 385 balanced examples — 55 per predicate × 7 predicates
Outputs   :
  results/voletc_exp4_osm_kg_ft_only_gpu_{cot|tot|got}_*_ckpt.json

Run:
    python exp04_finetuned_wikidata_kg.py
    python exp04_finetuned_wikidata_kg.py --strategy got
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
ADAPTER_PATH   = "finetuned_gptoss_osm_kg/final_adapter"
OSM_CACHE      = "results/osm_cache.json"
MODEL_TAG      = "exp4_osm_kg_ft_only_gpu"
OUTPUT_DIR     = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 512

SUFFIX     = "neighborhood_details_spatial_relation_16_sample"
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
    print(f"[OK] Dataset  : {DATASET}")
    print(f"[OK] Indices  : {INDICES_FILE}")
    print(f"[OK] Adapter  : {ADAPTER_PATH}")
    print(f"[INFO] KG at inference : DISABLED (--no-kg)")


def check_strategy_status(strategies: list) -> bool:
    print("\n[STATUS] Checkpoint summary:")
    all_done = True
    for strat in strategies:
        ckpt = os.path.join(OUTPUT_DIR, f"voletc_{MODEL_TAG}_{strat}_{SUFFIX}_ckpt.json")
        if os.path.exists(ckpt):
            data    = json.load(open(ckpt))
            done    = len(data.get("processed_indices", []))
            results = data.get("results", [])
            hits = sum(1 for r in results if r.get("match"))
            acc  = hits / len(results) * 100 if results else 0.0
            if results and done <= 96 and hits == 0:
                print(f"  {strat.upper():3s} : STALE ({done} rows, acc=0.0%) — auto-deleting & restarting ⚠️")
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
    print("  EXPERIMENT 4 — GPTOSS OSM-KG LoRA (KG in training, NO KG at inference)")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Adapter    : {ADAPTER_PATH}  [trained WITH OSM KG evidence]")
    print(f"  KG         : OSM in training only — inference runs WITHOUT evidence")
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
        "--no-kg",
    ]

    from eval_engine_gpu import main
    main()


if __name__ == "__main__":
    run()
