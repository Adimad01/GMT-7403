#!/usr/bin/env bash
# Full experiment grid: every strategy x every relation x every seed.
#
# Resumable — rows that already succeeded are never recomputed, so re-running
# after an interruption picks up where it stopped.
#
#   ./scripts/run_all.sh                  # seeds 1 2 3
#   SEEDS="1" ./scripts/run_all.sh        # one seed
#   BACKEND=mock ./scripts/run_all.sh     # dry run, no GPU
set -uo pipefail

SEEDS="${SEEDS:-1 2 3}"
BACKEND="${BACKEND:-hf}"
MODEL_ID="${MODEL_ID:-openai/gpt-oss-20b}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# transformers pulls in TensorFlow when it looks importable; the cluster's TF
# has protobuf-incompatible generated code that kills the import chain.
export USE_TF=0 USE_JAX=0 TF_CPP_MIN_LOG_LEVEL=3
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Refuse to start alongside another run: two model loads on one MIG partition
# surface as an NVML assert rather than a clean out-of-memory error.
if pgrep -f "spatial_eval.cli run" | grep -vq "^$$\$"; then
  echo "ERROR: another run is already active:"
  pgrep -af "spatial_eval.cli run"
  echo "Stop it first:  pkill -f 'spatial_eval.cli run'"
  exit 2
fi

echo "=== data integrity ==="
python3 -m spatial_eval.cli verify || { echo "data verification FAILED"; exit 3; }

echo
echo "=== running (backend=$BACKEND, seeds=$SEEDS) ==="
python3 -m spatial_eval.cli run --all --backend "$BACKEND" \
        --model-id "$MODEL_ID" --seeds $SEEDS
rc=$?

echo
echo "=== metrics ==="
python3 -m spatial_eval.cli evaluate

echo
echo "=== comparison ==="
python3 -m spatial_eval.cli report --metric accuracy_by_fact \
        --csv results/comparison.csv --json results/comparison.json

exit $rc
