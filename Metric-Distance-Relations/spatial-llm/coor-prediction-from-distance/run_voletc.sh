#!/usr/bin/env bash
# ============================================================================
# run_voletc.sh — Run Volet C: KG + Reasoning Strategies for Metric Distance
# ============================================================================
#
# Usage:
#   bash run_voletc.sh              # Run all strategies (CoT, ToT, GoT)
#   bash run_voletc.sh cot          # Run only CoT
#   bash run_voletc.sh tot          # Run only ToT
#   bash run_voletc.sh got          # Run only GoT
#   bash run_voletc.sh all 5        # Run all strategies on first 5 pairs (debug)
#
# After running, evaluate with:
#   python evaluate_voletc.py
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the project virtual environment
VENV_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)/.venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    echo "  Using venv: $VENV_DIR"
fi

STRATEGY="${1:-all}"
MAX_ROWS="${2:-}"
MODEL_TAG="base"
OUTPUT_DIR="./results"
TOP_K_ANCHORS=15
TEMPERATURE=0.1
MAX_NEW_TOKENS=2048

echo "=============================================="
echo "  VOLET C — Metric Distance Estimation"
echo "  Strategy:    ${STRATEGY}"
echo "  Model tag:   ${MODEL_TAG}"
echo "  Anchors:     top-${TOP_K_ANCHORS}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Output:      ${OUTPUT_DIR}"
if [ -n "${MAX_ROWS}" ]; then
    echo "  Max rows:    ${MAX_ROWS} (debug mode)"
fi
echo "=============================================="
echo ""

# Build the command
CMD="python3 gptoss_voletc_eval.py \
    --strategy ${STRATEGY} \
    --output-dir ${OUTPUT_DIR} \
    --model-tag ${MODEL_TAG} \
    --top-k-anchors ${TOP_K_ANCHORS} \
    --temperature ${TEMPERATURE} \
    --max-new-tokens ${MAX_NEW_TOKENS}"

if [ -n "${MAX_ROWS}" ]; then
    CMD="${CMD} --max-rows ${MAX_ROWS}"
fi

echo "Running: ${CMD}"
echo ""

eval ${CMD}

echo ""
echo "=============================================="
echo "  Done! Now run evaluation:"
echo "  python evaluate_voletc.py"
echo "=============================================="
