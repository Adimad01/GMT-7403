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

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

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

# Informational pre-flight. Not a gate: the audit reports known, accepted
# failures (train-split geocoding coverage), and those must not block a run.
if [ -f audit_data.py ]; then
  echo "--- data pre-flight -----------------------------------------------------------"
  $PY audit_data.py 2>&1 | grep -E "leakage:|SUMMARY" || true
  echo
fi

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
