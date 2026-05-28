"""
Experiment 6 — GPTOSS + Inférence LLM enrichie par KG  (base model, Ollama)
================================================================================
The base GPT-OSS-20B model (no fine-tuning, no adapter) is evaluated with
enriched LLM inference: CoT, ToT, and GoT reasoning strategies grounded on
dynamic OSM knowledge-graph evidence.

Inference is served through the remote Ollama endpoint at:
    http://ollama.apps.crdig.ulaval.ca  (model: gpt-oss)

STATUS: ALL 96 RESULTS ALREADY EXIST in the checkpoint files below.
This script reports the existing results and can re-run if the files are deleted.

Model     : gpt-oss (base, via Ollama remote API)
KG        : OSM (Nominatim) — used by CoT/ToT/GoT reasoning strategies
Inference : Remote Ollama (http://ollama.apps.crdig.ulaval.ca)
Strategies: CoT, ToT, GoT
Eval set  : 96 balanced examples (16 per predicate, random_state=42)
Outputs   :
  results/voletc_dynamic_osm_improved_version_cot_*_ckpt.json  ✅
  results/voletc_dynamic_osm_improved_version_tot_*_ckpt.json  ✅
  results/voletc_dynamic_osm_improved_version_got_*_ckpt.json  ✅

Run (only needed if checkpoints are deleted):
    python exp6_gptoss_enriched_ollama.py
    python exp6_gptoss_enriched_ollama.py --strategy cot
"""

import os
import sys
import json
import argparse

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATASET        = "../dataset/triplet_update_v3_30.csv"
MODEL_TAG      = "dynamic_osm_improved_version"
OUTPUT_DIR     = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 1024

SUFFIX     = "neighborhood_details_spatial_relation_16_sample"
STRATEGIES = ["cot", "tot", "got"]
# ---------------------------------------------------------------------------


def check_strategy_status(strategies: list) -> bool:
    """Report checkpoint status for each strategy. Returns True if all complete."""
    print("\n[STATUS] Checkpoint summary:")
    all_done = True
    for strat in strategies:
        ckpt = os.path.join(OUTPUT_DIR, f"voletc_{MODEL_TAG}_{strat}_{SUFFIX}_ckpt.json")
        if os.path.exists(ckpt):
            with open(ckpt) as f:
                data = json.load(f)
            done = len(data.get("processed_indices", []))
            results = data.get("results", [])
            if done >= 96 and results:
                acc = sum(1 for r in results if r.get("match")) / len(results) * 100
                print(f"  {strat.upper():3s} : COMPLETE  ({done}/96 rows, acc={acc:.1f}%)  ✅")
            else:
                print(f"  {strat.upper():3s} : PARTIAL   ({done}/96 rows) — will resume")
                all_done = False
        else:
            print(f"  {strat.upper():3s} : NOT STARTED")
            all_done = False
    return all_done


def preflight():
    if not os.path.exists(DATASET):
        print(f"[ERROR] Dataset not found: {DATASET}")
        sys.exit(1)
    print(f"[OK] Dataset : {DATASET}")


def run():
    parser = argparse.ArgumentParser(description="Experiment 6 runner (Ollama)")
    parser.add_argument("--strategy", choices=STRATEGIES + ["all"], default="all")
    args = parser.parse_args()

    target_strategies = STRATEGIES if args.strategy == "all" else [args.strategy]

    print("\n" + "=" * 70)
    print("  EXPERIMENT 6 — GPTOSS + Inférence LLM enrichie par KG")
    print("  (base model via Ollama + CoT/ToT/GoT)")
    print("=" * 70)
    print(f"  Model      : gpt-oss (base, no adapter)")
    print(f"  Endpoint   : http://ollama.apps.crdig.ulaval.ca")
    print(f"  KG         : OSM (dynamic Nominatim)")
    print(f"  Strategies : {', '.join(s.upper() for s in target_strategies)}")
    print(f"  Output dir : {OUTPUT_DIR}/")
    print("=" * 70)

    if check_strategy_status(target_strategies):
        print("\n[DONE] All strategies already complete.")
        print("       Delete the checkpoint files to force a re-run.")
        sys.exit(0)

    preflight()
    print()

    # Delegate to run_eval_osm.main() via sys.argv injection
    # run_eval_osm.py does its own stratified sampling (random_state=42),
    # which produces exactly the same 96 indices as eval_96_balanced_indices.json
    sys.argv = [
        "run_eval_osm.py",
        "--dataset",        DATASET,
        "--strategy",       args.strategy,
        "--output-dir",     OUTPUT_DIR,
        "--model-tag",      MODEL_TAG,
        "--temperature",    str(TEMPERATURE),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
    ]

    from run_eval_osm import main
    main()


if __name__ == "__main__":
    run()
