#!/usr/bin/env bash
# =============================================================================
# run_all_96_experiments.sh
# Master runner — 6 experiments on the 96 balanced test examples (OSM KG)
#
# Experiments:
#   1. GPTOSS Base
#   2. GPTOSS Fine-tuné
#   3. GPTOSS Fine-tuné + KG en entrée
#   4. GPTOSS Fine-tuné + KG comme entrée de fine-tuning
#   5. GPTOSS Fine-tuné + Inférence LLM enrichie par KG  (CoT/ToT/GoT, GPU)
#   6. GPTOSS + Inférence LLM enrichie par KG            (CoT/ToT/GoT, Ollama) ← ALREADY DONE
#
# Usage:
#   cd /path/to/Topological-Reasoning/code
#   bash run_all_96_experiments.sh
# =============================================================================

set -euo pipefail

# Python interpreter with all dependencies (torch, transformers, peft, pandas)
# Adjust to match your GPU environment (conda activate, virtualenv, etc.)
PYTHON="${PYTHON:-python}"

DATASET="../dataset/triplet_update_v3_30.csv"
INDICES="../dataset/eval_96_balanced_indices.json"
OUTPUT_DIR="results"
MODEL_ID="openai/gpt-oss-20b"
OSM_CACHE="results/osm_cache.json"

ADAPTER_TOPO="finetuned_gptoss_topological/final_adapter"
ADAPTER_OSM_KG="finetuned_gptoss_osm_kg/final_adapter"

echo ""
echo "============================================================"
echo "  Experiment 1 — GPTOSS Base (no adapter, no KG)"
echo "============================================================"
$PYTHON eval_kg_instruction_finetuned.py \
    --dataset        "$DATASET" \
    --model-id       "$MODEL_ID" \
    --kg-source      none \
    --model-tag      gptoss_base_96 \
    --filter-indices "$INDICES" \
    --output-dir     "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  Experiment 2 — GPTOSS Fine-tuné (adapter, no KG)"
echo "============================================================"
$PYTHON eval_kg_instruction_finetuned.py \
    --dataset        "$DATASET" \
    --model-id       "$MODEL_ID" \
    --adapter-path   "$ADAPTER_TOPO" \
    --kg-source      none \
    --model-tag      gptoss_finetuned_96 \
    --filter-indices "$INDICES" \
    --output-dir     "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  Experiment 3 — GPTOSS Fine-tuné + KG en entrée"
echo "  (topological adapter + OSM KG evidence at inference)"
echo "============================================================"
$PYTHON eval_kg_instruction_finetuned.py \
    --dataset        "$DATASET" \
    --model-id       "$MODEL_ID" \
    --adapter-path   "$ADAPTER_TOPO" \
    --kg-source      osm \
    --osm-cache      "$OSM_CACHE" \
    --model-tag      gptoss_finetuned_kg_input_96 \
    --filter-indices "$INDICES" \
    --output-dir     "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  Experiment 4 — GPTOSS Fine-tuné + KG comme entrée de fine-tuning"
echo "  (OSM-KG instruction-tuned adapter + OSM KG evidence)"
echo "============================================================"
$PYTHON eval_kg_instruction_finetuned.py \
    --dataset        "$DATASET" \
    --model-id       "$MODEL_ID" \
    --adapter-path   "$ADAPTER_OSM_KG" \
    --kg-source      osm \
    --osm-cache      "$OSM_CACHE" \
    --model-tag      gptoss_osm_kg_ft_96 \
    --filter-indices "$INDICES" \
    --output-dir     "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  Experiment 5 — GPTOSS Fine-tuné + Inférence LLM enrichie par KG"
echo "  (topological adapter + CoT/ToT/GoT via GPU)"
echo "============================================================"
$PYTHON run_eval_osm_gpu.py \
    --dataset        "$DATASET" \
    --model-id       "$MODEL_ID" \
    --adapter-path   "$ADAPTER_TOPO" \
    --filter-indices "$INDICES" \
    --strategy       all \
    --output-dir     "$OUTPUT_DIR" \
    --model-tag      dynamic_osm_finetuned_gpu

echo ""
echo "============================================================"
echo "  Experiment 6 — GPTOSS + Inférence LLM enrichie par KG"
echo "  (base model via Ollama, CoT/ToT/GoT) — ALREADY DONE"
echo "  Results: results/voletc_dynamic_osm_improved_version_*_ckpt.json"
echo "============================================================"
echo "  Skipping — checkpoint files already contain all 96 results."

echo ""
echo "All experiments complete. Run analyze_96_experiments.py for the comparison table."
