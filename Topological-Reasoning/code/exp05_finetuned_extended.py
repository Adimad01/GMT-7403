"""
Experiment 5 — GPTOSS Fine-tuné topo + Inférence enrichie étendue (GPU)
================================================================================
GPT-OSS-20B fine-tuned on raw topological data (same adapter as Exp 2), evaluated
with an extended CoT/ToT/GoT reasoning budget (max_new_tokens=1024 vs 512).

Key distinction from Experiment 2: the model is given twice the token budget to
develop its chain-of-thought, tree-of-thought, or graph-of-thought reasoning
over the OSM KG evidence.  This tests whether longer reasoning improves accuracy
for the fine-tuned-but-no-KG-trained model.

Model     : openai/gpt-oss-20b + finetuned_gptoss_topological/final_adapter
KG        : OSM (Nominatim) — same as Exp 2
Inference : GPU (local)
Strategies: CoT, ToT, GoT
Max tokens: 1024 (vs 512 in Exp 2)
Eval set  : 96 balanced examples — 16 per predicate
Outputs   :
  results/voletc_exp5_finetuned_enriched_gpu_{cot|tot|got}_*_ckpt.json

Run:
    python exp05_finetuned_extended.py
    python exp05_finetuned_extended.py --strategy cot
"""

import os
import sys
import json
import argparse

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATASET        = "../dataset/triplet_update_v3_30.csv"
INDICES_FILE   = "../dataset/eval_96_balanced_indices.json"
MODEL_ID       = "openai/gpt-oss-20b"
ADAPTER_PATH   = "finetuned_gptoss_topological/final_adapter"
OSM_CACHE      = "results/osm_cache.json"
MODEL_TAG      = "exp5_finetuned_enriched_gpu"
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
    print("  EXPERIMENT 5 — GPTOSS FT topo + Inférence enrichie étendue (GPU)")
    print(f"  [max_new_tokens={MAX_NEW_TOKENS} — extended reasoning vs Exp 2 (512)]")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Adapter    : {ADAPTER_PATH}")
    print(f"  KG         : OSM (dynamic Nominatim, same as Exp 2)")
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
