"""
Experiment 3 — GPTOSS Fine-tuné + KG en entrée
================================================================================
The model is fine-tuned on raw topological data (no KG) but at inference time
OSM KG evidence (coordinates, bounding boxes, administrative hierarchy) is
injected into the prompt.

This tests whether a model trained without KG can benefit from KG evidence at
inference — i.e., whether it can generalise to KG-augmented prompts it never
saw during training.

Model     : openai/gpt-oss-20b + finetuned_gptoss_topological/final_adapter
KG        : OSM (Nominatim) — injected into the inference prompt only
Eval set  : 96 balanced examples from triplet_update_v3_30.csv
Output    : results/kg_eval_gptoss_finetuned_kg_input_96_osm.csv

Run:
    python exp3_gptoss_finetuned_kg_input.py
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATASET        = "../dataset/triplet_update_v3_30.csv"
INDICES_FILE   = "../dataset/eval_96_balanced_indices.json"
MODEL_ID       = "openai/gpt-oss-20b"
ADAPTER_PATH   = "finetuned_gptoss_topological/final_adapter"
KG_SOURCE      = "osm"
OSM_CACHE      = "results/osm_cache.json"
MODEL_TAG      = "gptoss_finetuned_kg_input_96"
OUTPUT_DIR     = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 200   # slightly larger to accommodate KG evidence in reasoning

OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"kg_eval_{MODEL_TAG}_{KG_SOURCE}.csv")
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
        print(f"[OK] OSM cache: {len(cache)} entries pre-loaded ({OSM_CACHE})")
    else:
        print(f"[WARN] OSM cache not found at {OSM_CACHE} — will query Nominatim live (slower)")


def check_existing_results():
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        done = len(df)
        print(f"[RESUME] Found {done} rows already evaluated in {OUTPUT_CSV}")
        if done >= 96:
            acc = df["match"].astype(str).str.lower().eq("true").mean()
            print(f"[DONE]   All 96 rows complete. Accuracy = {acc:.2%}")
            print("         Delete the CSV to force a re-run.")
            sys.exit(0)
    else:
        print(f"[NEW]  Starting fresh — output will be saved to {OUTPUT_CSV}")


def run():
    preflight()
    check_existing_results()

    print("\n" + "=" * 70)
    print("  EXPERIMENT 3 — GPTOSS Fine-tuné + KG en entrée")
    print("  (topological adapter + OSM KG evidence at inference)")
    print("=" * 70)
    print(f"  Model  : {MODEL_ID}")
    print(f"  Adapter: {ADAPTER_PATH}")
    print(f"  KG     : OSM (injected at inference, not seen during training)")
    print(f"  Output : {OUTPUT_CSV}")
    print("=" * 70 + "\n")

    sys.argv = [
        "eval_kg_instruction_finetuned.py",
        "--dataset",        DATASET,
        "--model-id",       MODEL_ID,
        "--adapter-path",   ADAPTER_PATH,
        "--kg-source",      KG_SOURCE,
        "--osm-cache",      OSM_CACHE,
        "--model-tag",      MODEL_TAG,
        "--filter-indices", INDICES_FILE,
        "--output-dir",     OUTPUT_DIR,
        "--temperature",    str(TEMPERATURE),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
    ]

    from eval_kg_instruction_finetuned import main
    main()


if __name__ == "__main__":
    run()
