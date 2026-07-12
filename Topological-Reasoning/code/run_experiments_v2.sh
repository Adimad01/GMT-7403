#!/usr/bin/env bash
# =============================================================================
# run_experiments_v2.sh
# Full pipeline: fine-tuning + 6 experiments × 3 strategies (CoT / ToT / GoT)
#
# Experiments
# ─────────────────────────────────────────────────────────────────────────────
#  Exp 1  Base GPT-OSS-20B — no adapter, no KG
#  Exp 2  Fine-tuned LoRA — no KG at training or inference
#  Exp 3  OSM-KG LoRA — KG in training only, no KG at inference
#  Exp 4  Base + KG as input — no adapter, OSM KG injected at inference
#  Exp 5  Fine-tuned LoRA + KG as input — LoRA + OSM KG at inference
#  Exp 6  Base + KG RAG — retrieval-augmented generation
#
# Usage
#   cd ~/Topological-Reasoning/code
#   bash run_experiments_v2.sh              # full run (trains + evaluates)
#   bash run_experiments_v2.sh --skip-train # skip fine-tuning
#   bash run_experiments_v2.sh --fresh      # wipe adapters + results, restart
# =============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
SKIP_TRAIN=0
FRESH=0

for arg in "$@"; do
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=1
    [[ "$arg" == "--fresh" ]]      && FRESH=1
done

# Default behavior is RESUME: existing adapters are reused, existing result
# checkpoints are kept, and the eval engine skips rows already done. Pass
# --fresh to wipe adapters + results and start completely from scratch
# (use this when the dataset / eval split changed).

line()   { echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
header() { line; printf "  %s\n" "$1"; line; echo ""; }

export CUDA_VISIBLE_DEVICES=MIG-deebd0a8-233a-5ef8-b31a-f6f99bbe27a4
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — Dataset check
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 0 — Dataset check"

for f in "../dataset/topological_relations.csv" \
         "../dataset/topo_v2_eval.csv" \
         "../dataset/topo_v2_eval_indices.json"; do
    if [[ ! -f "$f" ]]; then
        echo "  [ERROR] $f not found"
        exit 1
    fi
    echo "  [OK] $f"
done

if [[ ! -f "../dataset/topo_v2_train.csv" ]]; then
    echo "  Building topological training data..."
    $PYTHON build_dataset_topological_v2.py
fi
echo "  [OK] topo_v2_train.csv"

if [[ ! -f "../dataset/osm_kg_balanced_train.jsonl" ]]; then
    echo "  Building OSM-KG training data..."
    $PYTHON build_osm_kg_from_topo_csv.py \
        --input  ../dataset/topo_v2_train.csv \
        --output ../dataset/osm_kg_balanced_train.jsonl
fi
echo "  [OK] osm_kg_balanced_train.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
if [[ $FRESH -eq 1 ]]; then
    echo ""
    header "FRESH START — wiping adapters and result checkpoints"
    rm -rf finetuned_gptoss_topo_v2/ finetuned_gptoss_osm_kg/
    rm -f results/voletc_*
    echo "  [OK] cleared adapters + results"
fi

if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo ""
    header "PHASE 1 — Fine-tuning adapters (train only if missing)"

    echo "  FT-1 · No-KG LoRA → finetuned_gptoss_topo_v2"
    line
    if [[ -f "finetuned_gptoss_topo_v2/final_adapter/adapter_model.safetensors" ]]; then
        echo "  [SKIP] adapter already exists — reusing (pass --fresh to retrain)"
    else
        $PYTHON train_runner_topo_v2.py
    fi

    echo ""
    echo "  FT-2 · OSM-KG LoRA → finetuned_gptoss_osm_kg"
    line
    if [[ -f "finetuned_gptoss_osm_kg/final_adapter/adapter_model.safetensors" ]]; then
        echo "  [SKIP] adapter already exists — reusing (pass --fresh to retrain)"
    else
        $PYTHON train_runner_osm_kg.py
    fi
else
    echo ""
    header "PHASE 1 — Skipping fine-tuning (--skip-train)"

    for adapter in finetuned_gptoss_topo_v2 finetuned_gptoss_osm_kg; do
        if [[ -f "$adapter/final_adapter/adapter_model.safetensors" ]]; then
            echo "  [OK] $adapter/final_adapter  ✅"
        else
            echo "  [WARN] $adapter not found — will train on demand"
        fi
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2a — Zero-shot evaluation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 2a — Zero-shot: 6 experiments × 3 strategies"

echo "  Exp 1 · Base GPT-OSS-20B (no adapter, no KG) · CoT / ToT / GoT"
line
$PYTHON exp1_base.py

echo ""
echo "  Exp 2 · Fine-tuned LoRA (no KG) · CoT / ToT / GoT"
line
$PYTHON exp2_ft_nokg.py

echo ""
echo "  Exp 3 · OSM-KG LoRA (KG in training only) · CoT / ToT / GoT"
line
$PYTHON exp3_ft_osmkg.py

echo ""
echo "  Exp 4 · Base + OSM KG as input · CoT / ToT / GoT"
line
$PYTHON exp4_base_kg_input.py

echo ""
echo "  Exp 5 · Fine-tuned LoRA + OSM KG as input · CoT / ToT / GoT"
line
$PYTHON exp5_ft_kg_input.py

echo ""
echo "  Exp 6 · Base + KG RAG · CoT / ToT / GoT"
line
$PYTHON exp6_base_kg_rag.py

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2b — Few-shot evaluation (5 same-label demos)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 2b — Few-shot (5 shots): 6 experiments × 3 strategies"

echo "  Exp 1 · Base GPT-OSS-20B (no adapter, no KG) · CoT / ToT / GoT · 5-shot"
line
$PYTHON exp1_base.py --shots 5

echo ""
echo "  Exp 2 · Fine-tuned LoRA (no KG) · CoT / ToT / GoT · 5-shot"
line
$PYTHON exp2_ft_nokg.py --shots 5

echo ""
echo "  Exp 3 · OSM-KG LoRA (KG in training only) · CoT / ToT / GoT · 5-shot"
line
$PYTHON exp3_ft_osmkg.py --shots 5

echo ""
echo "  Exp 4 · Base + OSM KG as input · CoT / ToT / GoT · 5-shot"
line
$PYTHON exp4_base_kg_input.py --shots 5

echo ""
echo "  Exp 5 · Fine-tuned LoRA + OSM KG as input · CoT / ToT / GoT · 5-shot"
line
$PYTHON exp5_ft_kg_input.py --shots 5

echo ""
echo "  Exp 6 · Base + KG RAG · CoT / ToT / GoT · 5-shot"
line
$PYTHON exp6_base_kg_rag.py --shots 5

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Analysis
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 3 — Analysing topological results"
$PYTHON analyze_results.py

echo ""
line
echo "  Topological pipeline complete."
echo "  Results : results/"
line
echo ""
