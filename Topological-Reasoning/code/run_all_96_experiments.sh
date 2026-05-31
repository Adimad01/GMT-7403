#!/usr/bin/env bash
# =============================================================================
# run_all_96_experiments.sh
# Master runner — 5 experiments × 3 strategies (CoT/ToT/GoT) on 96 examples
#
# All experiments use OSM KG evidence at inference via CoT, ToT, GoT strategies.
#
#  Exp 1 — Base (no adapter),          512 tok   → exp1_base_gpu
#  Exp 2 — FT topo (raw data),         512 tok   → exp2_finetuned_topo_gpu
#  Exp 3 — FT OSM-KG (KG in training), 1024 tok  → exp3_finetuned_kg_in_gpu
#  Exp 4 — FT topo, extended budget,   1024 tok  → exp5_finetuned_enriched_gpu
#  Exp 5 — Base + Ollama (DONE ✅)     N/A       → dynamic_osm_improved_version
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
echo "  Experiment 1 — Base GPT-OSS + CoT/ToT/GoT (GPU, 512 tok)"
echo "============================================================"
$PYTHON exp1_gptoss_base.py

echo ""
echo "============================================================"
echo "  Experiment 2 — FT topo + CoT/ToT/GoT (GPU, 512 tok)"
echo "============================================================"
$PYTHON exp2_gptoss_finetuned.py

echo ""
echo "============================================================"
echo "  Experiment 3 — FT OSM-KG + CoT/ToT/GoT (GPU, 1024 tok)"
echo "  [OSM-KG adapter: trained WITH KG evidence]"
echo "============================================================"
$PYTHON exp3_gptoss_finetuned_kg_input.py

echo ""
echo "============================================================"
echo "  Experiment 4 — FT topo extended + CoT/ToT/GoT (GPU, 1024 tok)"
echo "  [Same topo adapter as Exp 2, longer reasoning budget]"
echo "============================================================"
$PYTHON exp5_gptoss_finetuned_enriched_gpu.py

echo ""
echo "============================================================"
echo "  Experiment 5 — Base + Ollama (ALREADY COMPLETE)"
echo "  [Will print existing results and exit]"
echo "============================================================"
$PYTHON exp6_gptoss_enriched_ollama.py

echo ""
echo "All experiments complete."
echo "Run: $PYTHON analyze_96_experiments.py"
