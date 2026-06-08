"""
Experiment v2-04 — GPT-OSS-20B + Wikidata-KG LoRA (topological_relations v2)
================================================================================
Wikidata-KG adapter (trained on Wikidata-based instruction data) evaluated on
the v2 test set: 105 examples · 15/predicate · 3/level (L1–L5).

Key distinction from v2-03: the adapter was fine-tuned on Wikidata semantic
evidence (entity-level, encyclopaedic) rather than OSM geometric evidence.
At inference, OSM KG evidence (coordinates, bounding boxes, hierarchy) is
provided via CoT/ToT/GoT — testing whether Wikidata-KG fine-tuning transfers
to OSM-grounded topological inference.

Model    : openai/gpt-oss-20b + finetuned_gptoss_wikidata_kg/final_adapter
Dataset  : ../dataset/topo_v2_eval.csv
Indices  : ../dataset/topo_v2_eval_indices.json
Outputs  : results/voletc_v2_exp4_wikidata_kg_{cot|tot|got}_..._ckpt.json

Run:
    python exp_v2_04_wikidata_kg.py
    python exp_v2_04_wikidata_kg.py --strategy cot
"""

import os
import sys
import json
import argparse

# ---------------------------------------------------------------------------
DATASET        = "../dataset/topo_v2_eval.csv"
INDICES_FILE   = "../dataset/topo_v2_eval_indices.json"
MODEL_ID       = "openai/gpt-oss-20b"
ADAPTER_PATH   = "finetuned_gptoss_wikidata_kg/final_adapter"
OSM_CACHE      = "results/osm_cache.json"
MODEL_TAG      = "v2_exp4_wikidata_kg"
OUTPUT_DIR     = "results"
TEMPERATURE    = 0.1
MAX_NEW_TOKENS = 1024
N_EVAL         = 105

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
    print(f"[OK] Adapter : {ADAPTER_PATH}  [trained WITH Wikidata KG evidence]")
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
            data    = json.load(open(ckpt))
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
    print("  EXPERIMENT v2-04 — Wikidata-KG LoRA · topological_relations v2")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Adapter    : {ADAPTER_PATH}")
    print(f"  KG         : Wikidata in training / OSM at inference (1024 tok budget)")
    print(f"  Eval set   : {N_EVAL} examples · 15/predicate · 3/level")
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
