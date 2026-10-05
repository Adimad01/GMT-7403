#!/usr/bin/env bash
# Replay a handful of wrong answers with --save-traces, to read the reasoning behind them.
#
# The grid was run without traces, so predictions.jsonl holds the final label and nothing
# else. These rows are rerun in CoT, with the same seed: generation is seeded from the seed
# and the prompt, so the same prompt gives the same completion, and the trace shows the
# reasoning that produced the recorded error. The replayed answers are saved next to the
# traces, so a row that does not reproduce the recorded one can be seen and reported.
#
# It runs in a separate, detached worktree ($REPLAY, default ~/replay), never in this
# checkout: a subset run writes into the cell's own directory, and doing that here would
# touch the real results. Adapters are read from this checkout, where the
# weights are. The traces are copied back to traces/<relation>__<arm>.jsonl.
#
# The rows were chosen from the CoT cell of each arm: the arm's most frequent confusions,
# preferring rows the arm gets wrong in most of the five strategies.
#
#   bash scripts/replay_traces.sh
#
set -u
cd "$(dirname "$0")/.."
REPO=$PWD
REPLAY=${REPLAY:-$HOME/replay}
if [ ! -d "$REPLAY" ]; then
    git worktree add --detach "$REPLAY" HEAD || exit 1
fi
mkdir -p "$REPO/traces"

# The package must be imported from the replay copy, not from an editable install of this
# checkout: results are written under the package's own repository root, so importing the
# wrong copy would send the replay into the real results. Checked before anything runs.
export PYTHONPATH="$REPLAY/src${PYTHONPATH:+:$PYTHONPATH}"
root=$(cd "$REPLAY" && python3 -c "import spatial_eval.config as c; print(c.REPO_ROOT.resolve())")
if [ "$root" != "$(cd "$REPLAY" && pwd -P)" ]; then
    echo "spatial_eval est importé depuis $root, pas depuis $REPLAY : arrêt." >&2
    exit 1
fi

replay() {  # relation arm "rows" [extra cli args]
    rel=$1; arm=$2; rows=$3; shift 3
    case "$arm" in
        base)      var="" ;;
        kg)        var="_kg" ;;
        lora)      var="_lora" ;;
        lora_kg)   var="_lora_kg" ;;
        lorakg_kg) var="_lorakg_kg" ;;
    esac
    cell="$REPLAY/results/$rel/cot/seed1$var"
    # The replay copy's cell is emptied first: the runner refuses to write a subset over a
    # finished cell (rightly, in the real checkout), and here the cell is disposable.
    rm -rf "$cell"
    echo "=== $rel / $arm / rows $rows"
    (cd "$REPLAY" && python3 -m spatial_eval.cli run -r "$rel" -s cot --seeds 1 \
        --rows $rows --no-resume --save-traces "$@") || { echo "    ÉCHEC $rel $arm"; return; }
    cp "$cell/traces.jsonl" "$REPO/traces/${rel}__${arm}.jsonl"
    cp "$cell/predictions.jsonl" "$REPO/traces/${rel}__${arm}.predictions.jsonl"
}

A="$REPO/adapters"
replay topological base      "54 130"
replay topological kg        "56 127"   --kg-mode input
replay topological lora      "189 56"   --adapter "$A/topological"
replay topological lora_kg   "56 180"   --adapter "$A/topological" --kg-mode input
replay topological lorakg_kg "180 183"  --adapter "$A/topological_kg" --kg-mode input

replay cardinal base         "270 100"
replay cardinal kg           "273 60"   --kg-mode input
replay cardinal lora         "135 172"  --adapter "$A/cardinal"
replay cardinal lora_kg      "208"      --adapter "$A/cardinal" --kg-mode input
replay cardinal lorakg_kg    "133 187"  --adapter "$A/cardinal_kg" --kg-mode input

replay relative base         "66 145"
replay relative kg           "69 240"   --kg-mode input
replay relative lora         "129 245"  --adapter "$A/relative"
replay relative lora_kg      "129 245"  --adapter "$A/relative" --kg-mode input
replay relative lorakg_kg    "121 245"  --adapter "$A/relative_kg" --kg-mode input

echo
echo "Traces dans $REPO/traces :"
ls -1 "$REPO/traces"
