#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
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
#   cd ~/Relative-Reasoning/code
#   bash run_experiments.sh              # full run (trains + evaluates)
#   bash run_experiments.sh --skip-train # skip fine-tuning
# =============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
SKIP_TRAIN=0

for arg in "$@"; do
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=1
done

line()   { echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
header() { line; printf "  %s\n" "$1"; line; echo ""; }

export CUDA_VISIBLE_DEVICES=MIG-deebd0a8-233a-5ef8-b31a-f6f99bbe27a4
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — Dataset check
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 0 — Dataset check"

for f in "../dataset/relative_direction_relations.csv" \
         "../dataset/eval_20_balanced_indices.json"; do
    if [[ ! -f "$f" ]]; then
        echo "  [ERROR] $f not found"
        exit 1
    fi
    echo "  [OK] $f"
done

if [[ ! -f "../dataset/relative_balanced_train.csv" ]] || \
   [[ ! -f "../dataset/relative_osm_kg_train.jsonl" ]]; then
    echo "  Building relative training data..."
    $PYTHON build_relative_train_data.py
fi
echo "  [OK] Training data ready"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo ""
    header "PHASE 1 — Fine-tuning adapters from scratch"
    rm -rf finetuned_gptoss_relative/ finetuned_gptoss_relative_osm_kg/
    rm -f results/*_ckpt.json
    echo ""

    echo "  FT-1 · No-KG LoRA → finetuned_gptoss_relative"
    line
    $PYTHON train_runner_relative.py

    echo ""
    echo "  FT-2 · OSM-KG LoRA → finetuned_gptoss_relative_osm_kg"
    line
    $PYTHON train_runner_relative_osm_kg.py
else
    echo ""
    header "PHASE 1 — Skipping fine-tuning (--skip-train)"

    for adapter in finetuned_gptoss_relative finetuned_gptoss_relative_osm_kg; do
        if [[ -f "$adapter/final_adapter/adapter_model.safetensors" ]]; then
            echo "  [OK] $adapter/final_adapter  ✅"
        else
            echo "  [WARN] $adapter not found — will train on demand"
        fi
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 2 — Evaluating 6 experiments × 3 strategies on 20 examples"

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
# PHASE 3 — Analysis
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 3 — Analysing relative direction results"
$PYTHON analyze_results.py

echo ""
line
echo "  Relative Direction pipeline complete."
echo "  Results : results/"
line
echo ""
