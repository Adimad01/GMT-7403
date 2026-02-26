#!/usr/bin/env bash
set -euo pipefail

# run_all_inference.sh
# Converted from run_all_inference.ps1
# Usage: ./run_all_inference.sh

# --- CONFIG (override via environment if desired) ---
HF_API_TOKEN="${HF_API_TOKEN:-}"
BaseModel="unsloth/gpt-oss-20b-bnb-4bit"
AdapterDir="Topological_Reasoning_GPTOSS_Standard/final_adapter"
DataFile="../dataset/triplet_update_v3_30.csv"
ShotFile="../dataset/triplet_update_v3_70.csv"
SystemPromptFile="results/system_prompt.txt"
UserPromptFile="results/user_prompt_template.txt"
CheckpointInterval=5
BatchSize=1
Device="cuda"

# Outputs (will be placed in subdir below)
OutZero="../results/gptoss_preds_30_zero.jsonl"
OutFew="../results/gptoss_preds_30_few.jsonl"

# Move to script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# HF token
if [ -n "$HF_API_TOKEN" ]; then
  echo "Setting HF_API_TOKEN from environment/script variable."
  export HF_API_TOKEN
else
  if [ -z "${HF_API_TOKEN:-}" ] && [ -z "${HF_API_TOKEN+x}" ]; then
    if [ -z "${HF_API_TOKEN}" ]; then
      echo "Warning: HF_API_TOKEN not set. Will rely on existing environment or fail when needed."
    fi
  fi
fi

# Create results subfolder and adjust output paths
ResultsSubdir="../results/Results Gptoss20b FT"
mkdir -p "$ResultsSubdir"
OutZero="$ResultsSubdir/gptoss_preds_30_zero.jsonl"
OutFew="$ResultsSubdir/gptoss_preds_30_few.jsonl"

# Detect compute device
gpuName="CPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpuName=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | tr -d '\r') || true
else
  if command -v python >/dev/null 2>&1; then
    isCuda=$(python - <<'PY'
import sys
try:
    import torch
    print(torch.cuda.is_available())
except Exception:
    print('False')
PY
)
    if [[ "$isCuda" == "True" ]]; then
      gpuName=$(python - <<'PY'
import torch
try:
    print(torch.cuda.get_device_name(0))
except Exception:
    print('Unknown GPU')
PY
)
    fi
  fi
fi

echo "Compute device detected: $gpuName"

# Save run info
runInfo="$ResultsSubdir/run_info.txt"
printf 'Run started: %s\n' "$(date --iso-8601=seconds)" > "$runInfo"
printf 'Compute device: %s\n' "$gpuName" >> "$runInfo"

# Helper to run commands and exit on failure with message
run_cmd() {
  echo "+ $*"
  if ! "$@"; then
    echo "Command failed: $*" >&2
    exit 1
  fi
}

# Run zero-shot
echo "Running ZERO-SHOT inference..."
run_cmd python run_gptoss_inference.py \
  --base-model "$BaseModel" \
  --adapter-dir "$AdapterDir" \
  --data "$DataFile" \
  --system-prompt-file "$SystemPromptFile" \
  --user-prompt-file "$UserPromptFile" \
  --output "$OutZero" \
  --batch-size "$BatchSize" \
  --mode zero \
  --device "$Device" \
  --llm-offload \
  --checkpoint-interval "$CheckpointInterval" \
  --temperature 0

# Run few-shot
echo "Running FEW-SHOT inference (n_shots=4)..."
run_cmd python run_gptoss_inference.py \
  --base-model "$BaseModel" \
  --adapter-dir "$AdapterDir" \
  --data "$DataFile" \
  --shot-file "$ShotFile" \
  --system-prompt-file "$SystemPromptFile" \
  --user-prompt-file "$UserPromptFile" \
  --output "$OutFew" \
  --batch-size "$BatchSize" \
  --mode few \
  --n-shots 4 \
  --device "$Device" \
  --llm-offload \
  --checkpoint-interval "$CheckpointInterval" \
  --temperature 0

echo "All runs completed. Outputs:\n - Zero-shot: $OutZero\n - Few-shot: $OutFew"

exit 0
