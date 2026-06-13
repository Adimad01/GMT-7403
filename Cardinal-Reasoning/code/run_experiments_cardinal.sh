#!/usr/bin/env bash
# =============================================================================
# run_experiments_cardinal.sh
# Master runner — full pipeline: fine-tuning + 5 experiments × 3 strategies
#
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 0  Build cardinal training datasets (eval-excluded, 130/direction)
#             cardinal_balanced_train.csv  (1040 rows, balanced)
#             cardinal_train.jsonl         (1040 rows, plain)
#             cardinal_kg_train.jsonl      (1040 rows, KG-enriched)
#
#  PHASE 1  Fine-tune adapters from scratch (unless --skip-train)
#             FT-1  Cardinal-LoRA    → finetuned_gptoss_cardinal/final_adapter
#             FT-2  Cardinal-KG LoRA → finetuned_gptoss_cardinal_kg/final_adapter
#             FT-3  Wikidata-KG LoRA → (from Topological-Reasoning, cross-task)
#
#  PHASE 2  Evaluate 5 configurations × 3 strategies (CoT / ToT / GoT)
#             Config 1  Base model (no adapter)                      512 tok
#             Config 2  Cardinal-LoRA                                512 tok
#             Config 3  Cardinal-KG LoRA + compass KG at inference  1024 tok
#             Config 4  Wikidata-KG LoRA + compass KG (cross-task)  1024 tok
#             Config 5  Cardinal-LoRA + extended budget + KG        1024 tok
#
#  PHASE 3  Analyse and summarise results
# ─────────────────────────────────────────────────────────────────────────────
# Usage
#   cd /path/to/Cardinal-Reasoning/code
#   bash run_experiments_cardinal.sh                         # full run (fresh)
#   PYTHON=/path/to/python bash run_experiments_cardinal.sh  # custom interpreter
#   bash run_experiments_cardinal.sh --skip-train            # reuse adapters
# =============================================================================

set -euo pipefail

# Disable PyTorch CUDA CachingAllocator NVML queries — prevents NVML_SUCCESS assert
# on MIG-partitioned A100 GPUs where NVML cannot enumerate the virtual GPU correctly.
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

PYTHON="${PYTHON:-python}"
SKIP_TRAIN=0

for arg in "$@"; do
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=1
done

line()   { echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
header() { line; printf "  %s\n" "$1"; line; echo ""; }

# ─────────────────────────────────────────────────────────────────────────────
# PREREQUISITE — Install Triton (required for MXFP4 model loading)
# Without Triton, transformers tries to dequantize MXFP4 MoE expert layers on
# GPU which triggers a CUDA CachingAllocator NVML assertion on this server.
# With Triton, the model runs natively in MXFP4 format — no dequantization.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PREREQUISITE — Triton for MXFP4 support"
if $PYTHON -c "import triton; print('  [OK] Triton', triton.__version__, 'already installed')" 2>/dev/null; then
    :
else
    echo "  Triton not found — installing..."
    pip install "triton>=3.4.0" && echo "  [OK] Triton installed successfully" \
        || { echo "  [ERROR] Triton install failed."; echo "  Run manually: pip install triton>=3.4.0"; exit 1; }
fi

# Wikidata-KG adapter path (cross-task, lives in Topological-Reasoning/)
WIKIDATA_ADAPTER="../../Topological-Reasoning/code/finetuned_gptoss_wikidata_kg/final_adapter/adapter_model.safetensors"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — Build training datasets (eval-excluded, balanced 130/direction)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 0 — Dataset check (source: Topological-Reasoning/dataset/)"

SRC_DATA="../../Topological-Reasoning/dataset/cardinal_direction_relations.csv"
TRAIN_FILE="../dataset/cardinal_balanced_train.csv"
EVAL_IDX="../dataset/eval_32_balanced_indices.json"

if [[ ! -f "$SRC_DATA" ]]; then
    echo "  [ERROR] Source dataset not found: $SRC_DATA"
    exit 1
fi
echo "  [OK] Source: $SRC_DATA  (120 rows)"

if [[ ! -f "$EVAL_IDX" ]]; then
    echo "  [ERROR] Eval indices not found: $EVAL_IDX"
    exit 1
fi
echo "  [OK] Eval indices: $EVAL_IDX  (32 rows, 4/direction × 8)"

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "  [WARN] Training CSV not found — rebuilding..."
    $PYTHON build_cardinal_training_data.py --exclude-eval
else
    TRAIN_ROWS=$($PYTHON -c "import csv; print(sum(1 for _ in csv.reader(open('$TRAIN_FILE'))) - 1)")
    echo "  [OK] $TRAIN_FILE  ($TRAIN_ROWS rows)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo ""
    header "PHASE 1 — Fine-tuning adapters from scratch"

    echo "  Cleaning old adapter artefacts and result checkpoints..."
    rm -rf finetuned_gptoss_cardinal/
    rm -rf finetuned_gptoss_cardinal_kg/
    rm -f  results/*_ckpt.json
    echo "  Done. Starting fine-tuning."
    echo ""

    echo "  FT-1 · Cardinal-LoRA  (cardinal_train.jsonl — 1040 rows, plain, eval-excluded)"
    line
    $PYTHON train_runner_cardinal.py

    echo ""
    echo "  FT-2 · Cardinal-KG LoRA  (cardinal_kg_train.jsonl — 1040 rows, KG-enriched, eval-excluded)"
    line
    $PYTHON train_runner_cardinal_kg.py

    echo ""
    echo "  FT-3 · Wikidata-KG LoRA  — cross-task adapter from Topological-Reasoning/"
    if [[ ! -f "$WIKIDATA_ADAPTER" ]]; then
        echo "  [WARN] Wikidata-KG adapter not found at:"
        echo "         $WIKIDATA_ADAPTER"
        echo "         Config 4 (cross-task) will be SKIPPED."
    else
        echo "  [OK] $WIKIDATA_ADAPTER  ✅"
    fi
else
    echo ""
    header "PHASE 1 — Skipping fine-tuning (--skip-train flag)"
    echo "  Checking Cardinal-LoRA adapter ..."
    if [[ ! -f "finetuned_gptoss_cardinal/final_adapter/adapter_model.safetensors" ]]; then
        echo "  [TRAIN] Cardinal-LoRA not found — running train_runner_cardinal.py"
        line
        $PYTHON train_runner_cardinal.py
    else
        echo "  [OK] finetuned_gptoss_cardinal/final_adapter  ✅"
    fi

    echo "  Checking Cardinal-KG LoRA adapter ..."
    if [[ ! -f "finetuned_gptoss_cardinal_kg/final_adapter/adapter_model.safetensors" ]]; then
        echo "  [TRAIN] Cardinal-KG LoRA not found — running train_runner_cardinal_kg.py"
        line
        $PYTHON train_runner_cardinal_kg.py
    else
        echo "  [OK] finetuned_gptoss_cardinal_kg/final_adapter  ✅"
    fi

    echo "  Checking Wikidata-KG adapter (cross-task, required for Config 4) ..."
    if [[ ! -f "$WIKIDATA_ADAPTER" ]]; then
        echo "  [WARN] $WIKIDATA_ADAPTER not found"
        echo "         Config 4 will be SKIPPED"
    else
        echo "  [OK] $WIKIDATA_ADAPTER  ✅"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 2 — Evaluating 5 configurations × 3 strategies on 32 examples"

echo "  Config 1 · Base GPT-OSS-20B · CoT / ToT / GoT (512 tok)"
line
$PYTHON exp01_base_cardinal.py

echo ""
echo "  Config 2 · Cardinal-LoRA · CoT / ToT / GoT (512 tok)"
line
$PYTHON exp02_finetuned_cardinal.py

echo ""
echo "  Config 3 · Cardinal-KG LoRA + compass KG · CoT / ToT / GoT (1024 tok)"
line
$PYTHON exp03_cardinal_kg.py

echo ""
echo "  Config 4 · Wikidata-KG LoRA (cross-task) + compass KG · CoT / ToT / GoT (1024 tok)"
if [[ -f "$WIKIDATA_ADAPTER" ]]; then
    line
    $PYTHON exp04_wikidata_cardinal.py
else
    echo "  [SKIP] Wikidata-KG adapter not available."
fi

echo ""
echo "  Config 5 · Cardinal-LoRA + extended budget + KG · CoT / ToT / GoT (1024 tok)"
line
$PYTHON exp05_extended_cardinal.py

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Analysis
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 3 — Analysing cardinal results"
$PYTHON analyze_experiments_cardinal.py

echo ""
line
echo "  Pipeline complete."
echo ""
echo "  Results  : results/"
echo "  Chart    : results/cardinal_experiment_results.png"
line
echo ""
