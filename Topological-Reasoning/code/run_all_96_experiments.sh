#!/usr/bin/env bash
# =============================================================================
# run_all_96_experiments.sh
# Master runner — 6 experiments × 3 strategies (CoT/ToT/GoT) on 96 examples
#
# All experiments use OSM KG evidence + CoT, ToT, GoT reasoning strategies.
# The only difference between experiments is the model / adapter configuration.
#
#  Exp 1 — Base (no adapter)                   → exp1_base_gpu
#  Exp 2 — Fine-tuné (raw data adapter)        → exp2_finetuned_topo_gpu
#  Exp 3 — Fine-tuné + KG en entrée            → exp3_finetuned_kg_in_gpu
#  Exp 4 — Fine-tuné + KG comme ft             → exp4_finetuned_osm_kg_gpu
#  Exp 5 — Fine-tuné + Enriched GPU            → exp5_finetuned_enriched_gpu
#  Exp 6 — Base + Enriched Ollama (DONE ✅)    → dynamic_osm_improved_version
#
# Usage:
#   cd /path/to/Topological-Reasoning/code
#   bash run_all_96_experiments.sh
#   PYTHON=/path/to/python bash run_all_96_experiments.sh   # custom interpreter
# =============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"

echo ""
echo "============================================================"
echo "  Experiment 1 — GPTOSS Base + CoT/ToT/GoT (GPU)"
echo "============================================================"
$PYTHON exp1_gptoss_base.py

echo ""
echo "============================================================"
echo "  Experiment 2 — GPTOSS Fine-tuné + CoT/ToT/GoT (GPU)"
echo "============================================================"
$PYTHON exp2_gptoss_finetuned.py

echo ""
echo "============================================================"
echo "  Experiment 3 — Fine-tuné + KG en entrée + CoT/ToT/GoT (GPU)"
echo "============================================================"
$PYTHON exp3_gptoss_finetuned_kg_input.py

echo ""
echo "============================================================"
echo "  Experiment 4 — Fine-tuné KG ft + CoT/ToT/GoT (GPU)"
echo "============================================================"
$PYTHON exp4_gptoss_finetuned_kg_ft.py

echo ""
echo "============================================================"
echo "  Experiment 5 — Fine-tuné + Enriched inference (GPU)"
echo "============================================================"
$PYTHON exp5_gptoss_finetuned_enriched_gpu.py

echo ""
echo "============================================================"
echo "  Experiment 6 — Base + Enriched inference (Ollama)"
echo "  [ALREADY COMPLETE — will print results and exit]"
echo "============================================================"
$PYTHON exp6_gptoss_enriched_ollama.py

echo ""
echo "All experiments complete."
echo "Run: $PYTHON analyze_96_experiments.py"
