"""
Experiment 5 — GPTOSS Fine-tuné + Inférence LLM enrichie par KG
================================================================================
The GPT-OSS-20B model (fine-tuned on raw topological data) is evaluated with
enriched LLM inference: CoT, ToT, and GoT reasoning strategies grounded on
dynamic OSM knowledge-graph evidence.

All three strategies are run sequentially.  Results are saved as separate
checkpoint files per strategy and can be resumed if interrupted.

Model     : openai/gpt-oss-20b + finetuned_gptoss_topological/final_adapter
KG        : OSM (Nominatim) — used by CoT/ToT/GoT reasoning strategies
Inference : GPU (local, via PyTorch + PEFT)
Strategies: CoT, ToT, GoT
Eval set  : 96 balanced examples from triplet_update_v3_30.csv
Outputs   :
  results/voletc_dynamic_osm_finetuned_gpu_cot_*_ckpt.json
  results/voletc_dynamic_osm_finetuned_gpu_tot_*_ckpt.json
  results/voletc_dynamic_osm_finetuned_gpu_got_*_ckpt.json

Run:
    python exp5_gptoss_finetuned_enriched_gpu.py
    python exp5_gptoss_finetuned_enriched_gpu.py --strategy cot   # single strategy
"""

import os
import sys
import argparse

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATASET        = "../dataset/triplet_update_v3_30.csv"
INDICES_FILE   = "../dataset/eval_96_balanced_indices.json"
MODEL_ID       = "openai/gpt-oss-20b"
ADAPTER_PATH   = "finetuned_gptoss_topological/final_adapter"
OSM_CACHE      = "results/osm_cache.json"
MODEL_TAG      = "dynamic_osm_finetuned_gpu"
OUTPUT_DIR     = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 512

SUFFIX = "neighborhood_details_spatial_relation_16_sample"
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
        import json
        cache = json.load(open(OSM_CACHE))
        print(f"[OK] OSM cache: {len(cache)} entries ({OSM_CACHE})")
    else:
        print(f"[WARN] OSM cache not found — will query Nominatim live (slower)")


def check_strategy_status(strategies: list):
    """Report which strategies are complete, partial, or missing."""
    import json
    print("\n[STATUS] Checkpoint summary:")
    all_done = True
    for strat in strategies:
        ckpt = os.path.join(OUTPUT_DIR, f"voletc_{MODEL_TAG}_{strat}_{SUFFIX}_ckpt.json")
        if os.path.exists(ckpt):
            with open(ckpt) as f:
                data = json.load(f)
            done = len(data.get("processed_indices", []))
            results = data.get("results", [])
            if done >= 96:
                acc = sum(1 for r in results if r.get("match")) / len(results) * 100
                print(f"  {strat.upper():3s} : COMPLETE  ({done}/96 rows, acc={acc:.1f}%)")
            else:
                print(f"  {strat.upper():3s} : PARTIAL   ({done}/96 rows) — will resume")
                all_done = False
        else:
            print(f"  {strat.upper():3s} : NOT STARTED")
            all_done = False
    return all_done


def run():
    parser = argparse.ArgumentParser(description="Experiment 5 runner")
    parser.add_argument("--strategy", choices=STRATEGIES + ["all"], default="all",
                        help="Which strategy to run (default: all)")
    args = parser.parse_args()

    preflight()

    target_strategies = STRATEGIES if args.strategy == "all" else [args.strategy]

    print("\n" + "=" * 70)
    print("  EXPERIMENT 5 — GPTOSS Fine-tuné + Inférence LLM enrichie par KG")
    print("  (topological adapter + CoT/ToT/GoT via GPU)")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Adapter    : {ADAPTER_PATH}")
    print(f"  KG         : OSM (dynamic Nominatim)")
    print(f"  Strategies : {', '.join(s.upper() for s in target_strategies)}")
    print(f"  Output dir : {OUTPUT_DIR}/")
    print("=" * 70)

    if check_strategy_status(target_strategies):
        print("\n[DONE] All strategies complete for this experiment.")
        print("       Delete the checkpoint files to force a re-run.")
        sys.exit(0)

    print()

    # Delegate to run_eval_osm_gpu.main() via sys.argv injection
    sys.argv = [
        "run_eval_osm_gpu.py",
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

    from run_eval_osm_gpu import main
    main()


if __name__ == "__main__":
    run()
