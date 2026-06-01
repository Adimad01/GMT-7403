"""
Experiment 6 — GPTOSS + Inférence LLM enrichie par KG  (base model, Ollama)
================================================================================
Base GPT-OSS-20B (no fine-tuning) evaluated with enriched LLM inference: CoT,
ToT, and GoT reasoning strategies grounded on OSM KG via remote Ollama endpoint.

STATUS: ALL 96 RESULTS ALREADY EXIST in the checkpoint files below.

Model     : gpt-oss (base, no adapter, via Ollama)
Endpoint  : http://ollama.apps.crdig.ulaval.ca
KG        : OSM (Nominatim)
Strategies: CoT, ToT, GoT  — ALL COMPLETE ✅
Eval set  : 96 balanced examples (random_state=42, same as eval_96_balanced_indices.json)
Results   :
  CoT — results/voletc_dynamic_osm_improved_version_cot_*_ckpt.json  ✅ 66.7%
  ToT — results/voletc_dynamic_osm_improved_version_tot_*_ckpt.json  ✅ 69.8%
  GoT — results/voletc_dynamic_osm_improved_version_got_*_ckpt.json  ✅ 74.0%

Run (only if checkpoints are deleted):
    python exp06_base_ollama.py
    python exp06_base_ollama.py --strategy cot
"""

import os
import sys
import json
import argparse

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATASET        = "../dataset/triplet_update_v3_30.csv"
MODEL_TAG      = "dynamic_osm_improved_version"   # keep existing tag (results exist)
OUTPUT_DIR     = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 1024

SUFFIX     = "neighborhood_details_spatial_relation_16_sample"
STRATEGIES = ["cot", "tot", "got"]
# ---------------------------------------------------------------------------


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

    print("\n" + "=" * 70)
    print("  EXPERIMENT 6 — GPTOSS Base + Inférence enrichie (Ollama)")
    print("=" * 70)
    print(f"  Model      : gpt-oss  (base, no adapter)")
    print(f"  Endpoint   : http://ollama.apps.crdig.ulaval.ca")
    print(f"  KG         : OSM (dynamic Nominatim)")
    print(f"  Strategies : {', '.join(s.upper() for s in target)}")
    print(f"  Output tag : {MODEL_TAG}")
    print("=" * 70)

    if check_strategy_status(target):
        print("\n[DONE] All strategies already complete.")
        print("       Delete checkpoint files to force a re-run.")
        sys.exit(0)

    if not os.path.exists(DATASET):
        print(f"[ERROR] Dataset not found: {DATASET}")
        sys.exit(1)

    print()
    # run_eval_osm.py performs stratified sampling with random_state=42,
    # producing the same 96 indices as eval_96_balanced_indices.json
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
