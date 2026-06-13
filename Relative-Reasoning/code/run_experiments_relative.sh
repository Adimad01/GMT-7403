#!/usr/bin/env bash
# =============================================================================
# run_experiments_relative.sh
# Master runner — full pipeline: fine-tuning + 5 experiments × 3 strategies
#
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 0  No dataset building needed
#             Splits already exist:
#               ../dataset/relative_train.jsonl  (630 rows)
#               ../dataset/relative_eval.jsonl   (270 rows)
#
#  PHASE 1  Fine-tune Relative-LoRA adapter (unless --skip-train)
#             FT-1  Relative-LoRA → finetuned_gptoss_relative/final_adapter
#
#  PHASE 2  Evaluate 5 configurations × 3 strategies on 270 eval examples
#             Config 1  Base model (no adapter)                512 tok
#             Config 2  Relative-LoRA                          512 tok
#             Config 3  Relative-LoRA + extended budget        1024 tok
#             Config 4  Relative-LoRA ablation (512 tok)       512 tok
#             Config 5  Base model + extended budget (no FT)   1024 tok
#
#  PHASE 3  Analysis (analyze_experiments_relative.py if exists, else skip)
# ─────────────────────────────────────────────────────────────────────────────
# Usage
#   cd /path/to/Relative-Reasoning/code
#   bash run_experiments_relative.sh                         # full run (fresh)
#   PYTHON=/path/to/python bash run_experiments_relative.sh  # custom interpreter
#   bash run_experiments_relative.sh --skip-train            # reuse adapters
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

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — Dataset check (splits already exist)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 0 — Dataset check (source: Topological-Reasoning/dataset/)"

SRC_DATA="../../Topological-Reasoning/dataset/relative_direction_relations.csv"
TRAIN_FILE="../dataset/relative_balanced_train.csv"
EVAL_IDX="../dataset/eval_20_balanced_indices.json"

if [[ ! -f "$SRC_DATA" ]]; then
    echo "  [ERROR] Source dataset not found: $SRC_DATA"
    exit 1
fi
echo "  [OK] Source: $SRC_DATA  (75 rows)"

if [[ ! -f "$EVAL_IDX" ]]; then
    echo "  [ERROR] Eval indices not found: $EVAL_IDX"
    exit 1
fi
echo "  [OK] Eval indices: $EVAL_IDX  (20 rows, 4/class × 5)"

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "  [ERROR] Training CSV not found: $TRAIN_FILE"
    echo "         Expected relative_balanced_train.csv (55 rows) — regenerate splits."
    exit 1
fi
TRAIN_ROWS=$($PYTHON -c "import csv; print(sum(1 for _ in csv.reader(open('$TRAIN_FILE'))) - 1)")
echo "  [OK] $TRAIN_FILE  ($TRAIN_ROWS rows)"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo ""
    header "PHASE 1 — Fine-tuning Relative-LoRA adapter from scratch"

    echo "  Cleaning old adapter artefacts and result checkpoints..."
    rm -rf finetuned_gptoss_relative/
    rm -f  results/*_ckpt.json
    echo "  Done. Starting fine-tuning."
    echo ""

    echo "  FT-1 · Relative-LoRA  (relative_train.jsonl — 630 rows)"
    line
    $PYTHON train_runner_relative.py

else
    echo ""
    header "PHASE 1 — Skipping fine-tuning (--skip-train flag)"
    echo "  Checking Relative-LoRA adapter ..."
    if [[ ! -f "finetuned_gptoss_relative/final_adapter/adapter_model.safetensors" ]]; then
        echo "  [TRAIN] Relative-LoRA not found — running train_runner_relative.py"
        line
        $PYTHON train_runner_relative.py
    else
        echo "  [OK] finetuned_gptoss_relative/final_adapter  ✅"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 2 — Evaluating 5 configurations × 3 strategies on 20 examples"

echo "  Config 1 · Base GPT-OSS-20B · CoT / ToT / GoT (512 tok)"
line
$PYTHON exp01_base_relative.py

echo ""
echo "  Config 2 · Relative-LoRA · CoT / ToT / GoT (512 tok)"
line
$PYTHON exp02_finetuned_relative.py

echo ""
echo "  Config 3 · Relative-LoRA + extended budget · CoT / ToT / GoT (1024 tok)"
line
$PYTHON exp03_extended_relative.py

echo ""
echo "  Config 4 · Relative-LoRA ablation (512 tok) · CoT / ToT / GoT"
line
$PYTHON exp04_finetuned_512_relative.py

echo ""
echo "  Config 5 · Base model + extended budget (no FT) · CoT / ToT / GoT (1024 tok)"
line
$PYTHON exp05_base_1024_relative.py

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Analysis
# ─────────────────────────────────────────────────────────────────────────────
echo ""
header "PHASE 3 — Analysing relative navigation results"
if [[ -f "analyze_experiments_relative.py" ]]; then
    $PYTHON analyze_experiments_relative.py
else
    echo "  [SKIP] analyze_experiments_relative.py not found — skipping analysis."
fi

echo ""
line
echo "  Pipeline complete."
echo ""
echo "  Results  : results/"
line
echo ""
