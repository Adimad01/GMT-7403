#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# Master runner — 4 experiments × 3 strategies (CoT/ToT/GoT) on 96 examples
# All experiments run on GPU A100 80GB with OSM KG evidence at inference.
#
#  Exp 1 — Base (no adapter),          512 tok   → exp1_base_gpu
#  Exp 2 — FT topo (raw data),         512 tok   → exp2_finetuned_topo_gpu
#  Exp 3 — FT OSM-KG (KG in training), 1024 tok  → exp3_finetuned_kg_in_gpu
#  Exp 4 — FT topo, extended budget,   1024 tok  → exp5_finetuned_enriched_gpu
#
# Usage:
#   cd /path/to/Topological-Reasoning/code
#   bash run_experiments.sh
#   PYTHON=/path/to/python bash run_experiments.sh   # custom interpreter
# =============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"

echo ""
echo "============================================================"
echo "  Experiment 1 — Base GPT-OSS + CoT/ToT/GoT (GPU, 512 tok)"
echo "============================================================"
$PYTHON exp01_base_model.py

echo ""
echo "============================================================"
echo "  Experiment 2 — FT topo + CoT/ToT/GoT (GPU, 512 tok)"
echo "============================================================"
$PYTHON exp02_finetuned_topo.py

echo ""
echo "============================================================"
echo "  Experiment 3 — FT OSM-KG + CoT/ToT/GoT (GPU, 1024 tok)"
echo "  [OSM-KG adapter: trained WITH KG evidence]"
echo "============================================================"
$PYTHON exp03_finetuned_osm_kg.py

echo ""
echo "============================================================"
echo "  Experiment 4 — FT topo extended + CoT/ToT/GoT (GPU, 1024 tok)"
echo "  [Same topo adapter as Exp 2, longer reasoning budget]"
echo "============================================================"
$PYTHON exp05_finetuned_extended.py

echo ""
echo "All experiments complete."
echo "Run: $PYTHON analyze_experiments.py"
