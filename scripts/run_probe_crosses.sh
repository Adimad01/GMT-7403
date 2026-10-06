#!/usr/bin/env bash
# Run the crosses probe (data/topological/probe_crosses.csv) on the five arms.
#
# The probe asks `crosses`, `within` and `disjoint` about highways and long-distance trails,
# never about rivers (see data_generation/build_crosses_probe.py). It is read through
# SPATIAL_EVAL_SET, so its results go to results_probe_crosses/ and never mix with the main grid.
# Few-shot strategies are pinned to the main evaluation set and are not run here.
#
#   setsid nohup bash scripts/run_probe_crosses.sh > logs/probe_crosses.log 2>&1 < /dev/null &
#
# Safe to relaunch after an interruption: finished rows are kept.
set -u
cd "$(dirname "$0")/.."
export SPATIAL_EVAL_SET=probe_crosses
STRATS=${STRATS:-"zero_shot cot"}
A=adapters
for s in $STRATS; do
    echo "=== base / $s";      python3 -m spatial_eval.cli run -r topological -s "$s"
    echo "=== kg / $s";        python3 -m spatial_eval.cli run -r topological -s "$s" --kg-mode input
    echo "=== lora / $s";      python3 -m spatial_eval.cli run -r topological -s "$s" --adapter "$A/topological"
    echo "=== lora_kg / $s";   python3 -m spatial_eval.cli run -r topological -s "$s" --adapter "$A/topological" --kg-mode input
    echo "=== lorakg_kg / $s"; python3 -m spatial_eval.cli run -r topological -s "$s" --adapter "$A/topological_kg" --kg-mode input
done
echo "=== done; read with: python3 scripts/probe_report.py"
