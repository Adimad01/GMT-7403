#!/usr/bin/env bash
# =============================================================================
# run_experiments_v2.sh
# Full pipeline for the topological_relations.csv (v2) dataset
#
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 0  Split topological_relations.csv
#             Eval  : 105 examples · 15/predicate · 3 per ambiguity level (L1–L5)
#             Train : 1204 examples
#
#  PHASE 1  Fine-tune Topo-LoRA v2 on the v2 training split
#             Output : finetuned_gptoss_topo_v2/final_adapter
#             Note   : OSM-KG LoRA is reused from the v1 run
#                      (finetuned_gptoss_osm_kg/final_adapter)
#
#  PHASE 2  Evaluate 5 configurations × 3 strategies on 105 v2 test examples
#             v2-01  Base GPT-OSS-20B                           512 tok
#             v2-02  Topo-LoRA v2                                512 tok
#             v2-03  OSM-KG LoRA + KG evidence at inference    1024 tok
#             v2-04  Wikidata-KG LoRA + OSM evidence at inf.   1024 tok
#             v2-05  Topo-LoRA v2 + extended budget             1024 tok
#
#  PHASE 3  Analyse and summarise v2 results
# ─────────────────────────────────────────────────────────────────────────────
# Usage
#   cd /path/to/Topological-Reasoning/code
#   bash run_experiments_v2.sh                         # full run (fresh)
#   PYTHON=/path/to/python bash run_experiments_v2.sh  # custom interpreter
#   bash run_experiments_v2.sh --skip-train            # reuse existing adapter
# =============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
SKIP_TRAIN=0

for arg in "$@"; do
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=1
done

line() { echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
header() { line; printf "  %s\n" "$1"; line; echo ""; }

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — Build v2 dataset split
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 0 — Dataset splits (105 eval · 1204 train)"

if [[ -f "../dataset/topo_v2_eval.csv" && -f "../dataset/topo_v2_train.csv" ]]; then
    echo "  [OK] Splits already present — skipping build_dataset_topological_v2.py"
    echo "       ../dataset/topo_v2_eval.csv  ($(tail -n +2 ../dataset/topo_v2_eval.csv | wc -l | tr -d ' ') rows)"
    echo "       ../dataset/topo_v2_train.csv ($(tail -n +2 ../dataset/topo_v2_train.csv | wc -l | tr -d ' ') rows)"
else
    echo "  Splits not found — running build_dataset_topological_v2.py"
    $PYTHON build_dataset_topological_v2.py
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Fine-tune Topo-LoRA v2
# ─────────────────────────────────────────────────────────────────────────────
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo ""
    header "PHASE 1 — Fine-tuning Topo-LoRA v2 from scratch"

    echo "  Cleaning old v2 adapter..."
    rm -rf finetuned_gptoss_topo_v2/
    rm -f  results/v2_exp*_ckpt.json
    echo "  Done."
    echo ""

    echo "  Topo-LoRA v2 · topo_v2_train.csv (1204 rows · 7 predicates · no KG)"
    line
    $PYTHON train_runner_topo_v2.py

    echo ""
    echo "  OSM-KG LoRA : reusing existing finetuned_gptoss_osm_kg/final_adapter"
    line
    if [[ ! -f "finetuned_gptoss_osm_kg/final_adapter/adapter_model.safetensors" ]]; then
        echo "  [WARN] OSM-KG adapter not found — running train_runner_osm_kg.py"
        $PYTHON train_runner_osm_kg.py
    else
        echo "  [OK] finetuned_gptoss_osm_kg/final_adapter  ✅"
    fi
else
    echo ""
    header "PHASE 1 — Skipping fine-tuning (--skip-train)"
    echo "  Using : finetuned_gptoss_topo_v2/final_adapter"
    echo "  Using : finetuned_gptoss_osm_kg/final_adapter"
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 2 — Evaluating 5 configurations × 3 strategies on 105 v2 examples"

echo "  v2-01 · Base GPT-OSS-20B · CoT / ToT / GoT (512 tok)"
line
$PYTHON exp_v2_01_base.py

echo ""
echo "  v2-02 · Topo-LoRA v2 · CoT / ToT / GoT (512 tok)"
line
$PYTHON exp_v2_02_topo.py

echo ""
echo "  v2-03 · OSM-KG LoRA + KG evidence at inference · CoT / ToT / GoT (1024 tok)"
line
$PYTHON exp_v2_03_osm_kg.py

echo ""
echo "  v2-04 · OSM-KG LoRA (KG in fine-tuning only, no KG at inference) · CoT / ToT / GoT (512 tok)"
line
$PYTHON exp_v2_04_wikidata_kg.py

echo ""
echo "  v2-05 · Topo-LoRA v2 + enriched KG inference · CoT / ToT / GoT (1024 tok)"
line
$PYTHON exp_v2_05_extended.py

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Analysis
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 3 — Analysing v2 results"
$PYTHON analyze_experiments.py

echo ""
line
echo "  v2 pipeline complete."
echo ""
echo "  Results  : results/v2_exp*"
echo "  Report   : open report.html in a browser"
line
echo ""
