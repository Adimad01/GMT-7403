#!/usr/bin/env python3
"""Draw few-shot demonstrations from the evaluation split instead of the train split.

Why a second manifest rather than a change to the first.

The demonstrations in `fewshot_manifest.json` are training rows, and the train
split is also what the LoRA adapters were fitted on. So a fine-tuned model's
few-shot prompt is built out of text it has already been trained on, and its
score is optimistic for a reason that has nothing to do with few-shot
prompting. That is why the adapter arms had no few-shot cell at all: the number
would not have meant anything.

Demonstrations taken from the evaluation split are unseen by every adapter, so
the four arms become comparable on this strategy. The cost is that a
demonstration is now a scored item, which brings one condition that does not
arise with train-sourced demos: a row must never appear among its own
demonstrations. That is enforced below, and asserted afterwards.

Everything else is deliberately identical to the train-sourced rule, so that
the only difference between the two manifests is where the rows come from:
three shots, a draw seeded from the relation and the row index so every arm
sees the same demonstrations, at most one demonstration carrying the row's own
label, and at least two distinct labels among the three.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from spatial_eval.config import COLUMNS, RELATIONS          # noqa: E402

SHOTS = 3
# Distinct from the train-sourced DEMO_SEED (42): the two manifests must not
# draw the same row positions, or a reader comparing them would be looking at
# one coincidence and calling it a control.
DEMO_SEED = 4242


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def build(relation: str) -> int:
    data = REPO / "data" / relation
    evalr = _rows(data / "eval.csv")
    lc = COLUMNS[relation]["label"]

    eval_manifest = json.loads((data / "eval_manifest.json").read_text(encoding="utf-8"))
    man_sha = eval_manifest["manifest_sha256"]

    if len(evalr) < SHOTS + 1:
        print(f"  {relation}: only {len(evalr)} eval rows, cannot draw {SHOTS}")
        return 1

    labels_present = {r[lc].strip().lower() for r in evalr}
    if len(labels_present) < 2:
        print(f"  {relation}: eval split carries a single label, cannot vary demos")
        return 1

    demos: dict[str, list[int]] = {}
    for i, r in enumerate(evalr):
        gold = r[lc].strip().lower()
        # The row itself is out of its own pool. Without this a row would be
        # shown its own description and its own gold label, and would be scored
        # on reproducing it.
        pool = [j for j in range(len(evalr)) if j != i]
        rng = random.Random(f"{DEMO_SEED}:{relation}:{i}")

        def acceptable(sel: list[int]) -> bool:
            labs = [evalr[j][lc].strip().lower() for j in sel]
            return labs.count(gold) <= 1 and len(set(labs)) >= 2

        picked = rng.sample(pool, SHOTS)
        tries = 0
        while not acceptable(picked) and tries < 200:
            picked = rng.sample(pool, SHOTS)
            tries += 1
        if not acceptable(picked):
            print(f"  {relation}: could not draw varied demos for eval row {i}")
            return 1
        demos[str(i)] = sorted(picked)

    # Assert the contract rather than trust the loop above. A silent violation
    # here is a leak that would show up only as an implausibly high score.
    for key, idxs in demos.items():
        i = int(key)
        assert i not in idxs, f"{relation}: row {i} is its own demonstration"
        assert len(set(idxs)) == SHOTS, f"{relation}: repeated demo for row {i}"
        labs = [evalr[j][lc].strip().lower() for j in idxs]
        gold = evalr[i][lc].strip().lower()
        assert labs.count(gold) <= 1, f"{relation}: row {i} sees its label twice"
        assert len(set(labs)) >= 2, f"{relation}: row {i} sees one label only"

    demo_sha = hashlib.sha256(
        json.dumps(demos, sort_keys=True).encode()).hexdigest()

    out = data / "fewshot_manifest_eval.json"
    out.write_text(json.dumps({
        "domain": relation,
        "shots": SHOTS,
        "demo_csv": f"data/{relation}/eval.csv",
        "demo_rows": len(evalr),
        "demo_split": "eval",
        "selection_rule": (
            f"{SHOTS} evaluation rows drawn by an RNG seeded from base seed "
            f"{DEMO_SEED}, the relation and the row index, so every arm sees "
            f"the same demonstrations. A row is excluded from its own draw. At "
            f"most one demonstration may carry the row's own label, and the "
            f"three must span at least two labels."),
        "eval_manifest_sha256": man_sha,
        "demo_map_sha256": demo_sha,
        "warning": (
            "the demo pool is the evaluation split, so no adapter has been "
            "trained on these rows and the four arms are comparable on this "
            "strategy. In exchange a demonstration is itself a scored item: "
            "a row never demonstrates itself, but it does appear in other "
            "rows' prompts. Report this strategy as few_shot_eval, never "
            "merged with the train-sourced few_shot numbers."),
        "contract": ["every few_shot_eval arm must use exactly these demo indices",
                     "no row appears among its own demonstrations"],
        "demos": demos,
    }, indent=2) + "\n", encoding="utf-8")

    spread = sum(1 for k, v in demos.items()
                 if evalr[int(k)][lc].strip().lower()
                 in [evalr[j][lc].strip().lower() for j in v])
    print(f"  {relation:12} eval {len(evalr)} rows, {SHOTS} shots, "
          f"demo sha {demo_sha[:12]}, {spread} rows see their own label once")
    return 0


def main() -> int:
    print("Few-shot demonstrations drawn from the evaluation split")
    rc = 0
    for relation in RELATIONS:
        rc |= build(relation)
    if rc == 0:
        print("\n  written: data/<relation>/fewshot_manifest_eval.json")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
