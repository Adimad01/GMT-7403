#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# Master runner — full pipeline: fine-tuning + 5 experiments × 3 strategies
#
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 0  Build balanced training datasets (39 examples/predicate, 234 total)
#             from triplet_update_v3_70.csv and osm_kg_train.jsonl
#
#  PHASE 1  Fine-tune adapters from scratch on balanced data
#             FT-1  Topo-LoRA       → finetuned_gptoss_topological/final_adapter
#             FT-2  OSM-KG LoRA     → finetuned_gptoss_osm_kg/final_adapter
#
#  PHASE 2  Evaluate 5 configurations × 3 strategies (CoT / ToT / GoT)
#             Config 1  Base model (no adapter)                        512 tok
#             Config 2  Topo-LoRA                                      512 tok
#             Config 3  OSM-KG LoRA + KG evidence at inference        1024 tok
#             Config 4  OSM-KG LoRA, NO KG at inference (FT only)      512 tok
#             Config 5  Topo-LoRA + extended reasoning budget          1024 tok
#
#  PHASE 3  Analyse and summarise results
# ─────────────────────────────────────────────────────────────────────────────
# Usage
#   cd /path/to/Topological-Reasoning/code
#   bash run_experiments.sh                         # full run (fresh start)
#   PYTHON=/path/to/python bash run_experiments.sh  # custom interpreter
#   bash run_experiments.sh --skip-train            # reuse existing adapters
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
# PHASE 0 — Build balanced training datasets
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 0 — Building balanced training datasets (39/predicate · 234 total)"

$PYTHON build_balanced_training_data.py

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Fine-tuning (from scratch unless --skip-train)
# ─────────────────────────────────────────────────────────────────────────────
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo ""
    header "PHASE 1 — Fine-tuning adapters from scratch"

    # Remove old adapters and stale result checkpoints
    echo "  Cleaning old adapter artefacts and result checkpoints..."
    rm -rf finetuned_gptoss_topological/
    rm -rf finetuned_gptoss_osm_kg/
    rm -rf finetuned_gptoss_wikidata_kg/
    rm -f  results/*_ckpt.json
    echo "  Done. Starting fine-tuning."
    echo ""

    echo "  FT-1 · Topo-LoRA  (triplet_balanced_train.csv — 234 rows, no KG in input)"
    line
    $PYTHON train_runner_topo.py

    echo ""
    echo "  FT-2 · OSM-KG LoRA  (osm_kg_balanced_train.jsonl — 234 rows)"
    line
    $PYTHON train_runner_osm_kg.py

    echo ""
else
    echo ""
    header "PHASE 1 — Skipping fine-tuning (--skip-train flag)"
    echo "  Checking Topo-LoRA adapter ..."
    if [[ ! -f "finetuned_gptoss_topological/final_adapter/adapter_model.safetensors" ]]; then
        echo "  [TRAIN] Topo-LoRA not found — running train_runner_topo.py"
        line
        $PYTHON train_runner_topo.py
    else
        echo "  [OK] finetuned_gptoss_topological/final_adapter  ✅"
    fi

    echo "  Checking OSM-KG LoRA adapter (used by Config 3 AND Config 4) ..."
    if [[ ! -f "finetuned_gptoss_osm_kg/final_adapter/adapter_model.safetensors" ]]; then
        echo "  [TRAIN] OSM-KG LoRA not found — running train_runner_osm_kg.py"
        line
        $PYTHON train_runner_osm_kg.py
    else
        echo "  [OK] finetuned_gptoss_osm_kg/final_adapter  ✅"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 2 — Evaluating 5 configurations × 3 strategies on 96 balanced examples"

echo "  Config 1 · Base GPT-OSS-20B · CoT / ToT / GoT (512 tok)"
line
$PYTHON exp01_base_model.py

echo ""
echo "  Config 2 · Topo-LoRA · CoT / ToT / GoT (512 tok)"
line
$PYTHON exp02_finetuned_topo.py

echo ""
echo "  Config 3 · OSM-KG LoRA + KG evidence at inference · CoT / ToT / GoT (1024 tok)"
line
$PYTHON exp03_finetuned_osm_kg.py

echo ""
echo "  Config 4 · OSM-KG LoRA, NO KG at inference (ablation: FT only) · CoT / ToT / GoT (512 tok)"
line
$PYTHON exp04_finetuned_wikidata_kg.py

echo ""
echo "  Config 5 · Topo-LoRA + extended reasoning budget · CoT / ToT / GoT (1024 tok)"
line
$PYTHON exp05_finetuned_extended.py

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Analysis
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 3 — Analysing vol-1 results"
$PYTHON analyze_experiments_v1.py

echo ""
line
echo "  Pipeline complete."
echo ""
echo "  Results  : results/"
echo "  Report   : open report.html in a browser"
line
echo ""
