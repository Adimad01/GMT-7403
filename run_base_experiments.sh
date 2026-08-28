#!/usr/bin/env bash
#
# run_base_experiments.sh
# =============================================================================
# Experiment 1 (base GPT-OSS-20B, no KG anywhere) across all three spatial
# relation families, both prompting modes, and several seeds.
#
# This is the floor every other arm is measured against, so it runs first and
# it runs with seeds: generation uses do_sample=True, and a single draw would
# give a baseline with unknown run-to-run variance -- which would make every
# later comparison against it unreliable.
#
# Grid
#   3 domains x 2 shot modes x N seeds        (each call runs CoT, ToT and GoT)
#   = 18 invocations / 54 strategy-runs at the default 3 seeds
#
# Resumable: the engine checkpoints per row and skips completed work, so
# re-running after an interruption picks up where it stopped.
#
# Usage
#   ./run_base_experiments.sh                     # seeds 1 2 3, all domains
#   SEEDS="1 2 3 4 5" ./run_base_experiments.sh   # more seeds
#   DOMAINS="Topological-Reasoning" ./run_base_experiments.sh
#   SHOTS="0" ./run_base_experiments.sh           # zero-shot only
#   STRATEGY=cot ./run_base_experiments.sh        # single strategy
#
# Long runs: launch under tmux (or nohup) so an SSH drop does not kill it.
#   tmux new -s exp1 './run_base_experiments.sh 2>&1 | tee exp1.log'
# =============================================================================

set -uo pipefail

SEEDS="${SEEDS:-1 2 3}"
SHOTS="${SHOTS:-0 5}"
STRATEGY="${STRATEGY:-all}"
DOMAINS="${DOMAINS:-Topological-Reasoning Cardinal-Reasoning Relative-Reasoning}"
PY="${PY:-python3}"

# Set before any python starts, so import order inside the scripts cannot matter.
# transformers imports TensorFlow via image_transforms when TF looks available;
# the cluster's TF has protobuf-incompatible generated code and takes the whole
# import chain down with it. These engines are torch-only.
export USE_TF="${USE_TF:-0}"
export USE_JAX="${USE_JAX:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

# --- single-instance lock ----------------------------------------------------
# Two of these running at once put two 20B model loads on the same GPU, which
# on a MIG A100 surfaces as an NVML assert inside the CUDA caching allocator
# rather than a clean OOM. Refuse to start instead.
LOCK="$ROOT/.exp_running.lock"
if [ -e "$LOCK" ]; then
  other=$(cat "$LOCK" 2>/dev/null || echo "?")
  if kill -0 "$other" 2>/dev/null; then
    echo "ERROR: another experiment run is already active (PID $other)."
    echo "       Wait for it, or stop it with:  kill $other"
    echo "       Two runs share one GPU and will fail with NVML/allocator errors."
    exit 2
  fi
  echo "[LOCK] stale lock from PID $other (not running) - taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

total=0; done_n=0; failed=0
declare -a FAILURES=()

for d in $DOMAINS; do for s in $SHOTS; do for sd in $SEEDS; do
  total=$((total + 1))
done; done; done

echo "==============================================================================="
echo "  EXPERIMENT 1 - base model, no KG"
echo "==============================================================================="
echo "  domains   : $DOMAINS"
echo "  shots     : $SHOTS   (0 = zero-shot, 5 = few-shot)"
echo "  seeds     : $SEEDS"
echo "  strategy  : $STRATEGY"
echo "  invocations: $total   (each runs every selected strategy)"
echo "  started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo

# Pre-flight. Geocoding-coverage failures are known and accepted, so they do not
# block. LEAKAGE does block: if an eval row's answer is also in train, the
# fine-tuned arms can memorise it and every comparison against the base arms
# becomes meaningless. Burning GPU hours to produce invalid numbers is worse
# than stopping. Override with ALLOW_LEAKAGE=1 only to reproduce a known-bad run.
if [ -f audit_data.py ]; then
  echo "--- data pre-flight -----------------------------------------------------------"
  # Strip any ANSI codes so the status words are greppable regardless.
  audit_out=$($PY audit_data.py 2>&1 | sed $'s/\033\[[0-9;]*m//g')
  echo "$audit_out" | grep -E "leakage:|SUMMARY" || true
  if echo "$audit_out" | grep -qE "FAIL[[:space:]]+leakage"; then
    echo
    echo "ERROR: train/eval leakage detected - refusing to run."
    echo "$audit_out" | grep -E "FAIL[[:space:]]+leakage" | sed 's/^/       /'
    echo
    echo "       Most likely the split files on this machine are out of sync."
    echo "       Fix:   git pull --rebase origin main"
    echo "       Check: $PY audit_data.py"
    if [ "${ALLOW_LEAKAGE:-0}" != "1" ]; then
      exit 3
    fi
    echo "       ALLOW_LEAKAGE=1 set - continuing with KNOWN-INVALID data."
  fi
  echo
fi

# Environment pre-flight. A 20B load takes minutes; version skew should surface
# in seconds. Warn rather than block -- only the user can judge their cluster.
echo "--- environment ---------------------------------------------------------------"
$PY - <<'PYCHK' || true
import importlib
def v(m):
    try:
        return importlib.import_module(m).__version__
    except Exception as e:
        return f"UNAVAILABLE ({type(e).__name__})"
tf_v, hub_v = v("transformers"), v("huggingface_hub")
print(f"    torch {v('torch')}   transformers {tf_v}   huggingface_hub {hub_v}")
try:
    maj = int(str(tf_v).split(".")[0])
    if maj >= 5:
        print("    WARNING: transformers 5.x breaks the MXFP4 patches — "
              "pip install 'transformers>=4.55,<5'")
except Exception:
    pass
try:
    a, b = (int(x) for x in str(hub_v).split(".")[:2])
    if (a, b) < (0, 34):
        print(f"    WARNING: huggingface_hub {hub_v} is too old for transformers "
              f"{tf_v} (needs >=0.34) — pip install 'huggingface-hub>=0.34,<1.0'")
except Exception:
    pass
PYCHK
echo

start_all=$(date +%s)

for domain in $DOMAINS; do
  if [ ! -d "$domain/code" ]; then
    echo "[SKIP] $domain - no code/ directory"
    continue
  fi
  for shots in $SHOTS; do
    for seed in $SEEDS; do
      done_n=$((done_n + 1))
      label="$domain shots=$shots seed=$seed"
      echo "-------------------------------------------------------------------------------"
      echo "[$done_n/$total] $label"
      echo "-------------------------------------------------------------------------------"
      t0=$(date +%s)
      (
        cd "$domain/code" || exit 1
        $PY exp1_base.py --strategy "$STRATEGY" --shots "$shots" --seed "$seed"
      )
      rc=$?
      t1=$(date +%s)
      if [ $rc -eq 0 ]; then
        echo "[OK]   $label   ($((t1 - t0))s)"
      else
        # Keep going: one bad cell must not throw away the rest of a long run.
        echo "[FAIL] $label   (exit $rc, $((t1 - t0))s)"
        failed=$((failed + 1))
        FAILURES+=("$label (exit $rc)")
      fi
      echo
    done
  done
done

elapsed=$(( $(date +%s) - start_all ))
echo "==============================================================================="
echo "  DONE  -  $((done_n - failed))/$total succeeded, $failed failed"
printf '  elapsed: %dh %dm %ds\n' $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
echo "==============================================================================="
if [ ${#FAILURES[@]} -gt 0 ]; then
  echo "  failures:"
  for f in "${FAILURES[@]}"; do echo "    - $f"; done
  echo
fi
echo "  Analyse:  $PY stats_analysis.py --by-level"
echo

exit $(( failed > 0 ? 1 : 0 ))
