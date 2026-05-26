#!/usr/bin/env bash
# =============================================================================
# run_improved_version.sh
# Runs all 6 experiments (OSM + Wikidata) x (CoT + ToT + GoT) with the
# improved strategies. Outputs files contain "_improved_version" in the name.
# =============================================================================
set -euo pipefail

PYTHON="$(dirname "$0")/venv/bin/python3"
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET="$CODE_DIR/../dataset/triplet_update_v3_30.csv"
OUTPUT_DIR="$CODE_DIR/results"

OSM_TAG="dynamic_osm_improved_version"
WD_TAG="dynamic_wikidata_improved_version"

echo "============================================================"
echo "  IMPROVED VERSION EXPERIMENTS"
echo "  Dataset : $DATASET"
echo "  Output  : $OUTPUT_DIR"
echo "============================================================"

cd "$CODE_DIR"

# ── OSM ──────────────────────────────────────────────────────────────────────
for STRAT in cot tot got; do
    echo ""
    echo ">>> OSM / $(echo "$STRAT" | tr '[:lower:]' '[:upper:]')"
    "$PYTHON" run_eval_osm.py \
        --dataset "$DATASET" \
        --strategy "$STRAT" \
        --output-dir "$OUTPUT_DIR" \
        --model-tag "$OSM_TAG"
done

# ── Wikidata ──────────────────────────────────────────────────────────────────
for STRAT in cot tot got; do
    echo ""
    echo ">>> Wikidata / $(echo "$STRAT" | tr '[:lower:]' '[:upper:]')"
    "$PYTHON" run_eval_wikidata.py \
        --dataset "$DATASET" \
        --strategy "$STRAT" \
        --output-dir "$OUTPUT_DIR" \
        --model-tag "$WD_TAG"
done

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS DONE"
echo "  Results in: $OUTPUT_DIR"
echo "  Pattern   : voletc_*_improved_version_*_ckpt.json"
echo "============================================================"
