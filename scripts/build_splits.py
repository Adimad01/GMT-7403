"""Rebuild a relation's train/eval split and both manifests together.

The train pool is sized for fine-tuning and doubles as the few-shot demo
source: each evaluation row draws three demonstrations carrying its own label,
chosen by an RNG seeded from the row so the choice is fixed across arms.

One consequence to keep in mind when reporting. For the BASE model those demos
are unseen text, so few-shot is a clean comparison. For a model fine-tuned on
this same pool they are training data, and its few-shot numbers will be
optimistic. The two should be reported as separate arms, not pooled.

Splits and manifests must be regenerated as one operation: an eval manifest
pins row content by hash, and the few-shot manifest pins itself to that hash,
so producing them separately is how they drift apart.

The grid is regular -- a fixed number of rows in every (label, level) cell --
so the split takes one row per cell for training and the rest for evaluation.
That keeps both sides balanced and makes demo selection deterministic: for any
eval row there is exactly one training row per level carrying the same label,
so there is nothing to sample and no seed to get wrong.

    python3 scripts/build_splits.py --relation relative
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# eval_per_cell: rows held out of each (label, level) cell for evaluation. The
# rest become the training pool, so eval stays balanced and eval size is fixed
# by design rather than by whatever is left over.
SPEC = {
    "cardinal": dict(domain="Cardinal-Reasoning", eval_per_cell=6,
                     ground_truth="labels are cone-based cardinal sectors "
                                  "computed from city centroids, each at least "
                                  "8 degrees inside its 45-degree sector, "
                                  "agreeing with the projection-based reading, "
                                  "and reciprocal"),
    "relative": dict(domain="Relative-Reasoning", eval_per_cell=10,
                     ground_truth="labels are derived from the stated observer "
                                  "frame: the angle of the subject off the "
                                  "observer's sight line to the target, plus "
                                  "relative depth"),
    "topological": dict(domain="Topological-Reasoning", eval_per_cell=7,
                        ground_truth="labels are DE-9IM relations computed from "
                                     "OpenStreetMap polygons with a tolerance "
                                     "sized to the simplification error"),
}
SHOTS = 3
DEMO_SEED = 42


def read(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(p: Path, rows: list[dict], fields: list[str]) -> None:
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def row_hash(r: dict) -> str:
    return hashlib.sha256("|".join(str(r[k]) for k in sorted(r)).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", required=True, choices=list(SPEC))
    ap.add_argument("--dry-run", action="store_true",
                    help="report the hashes without writing anything")
    args = ap.parse_args()

    spec = SPEC[args.relation]
    data = REPO / "data" / args.relation
    corpus = read(data / "corpus.csv")
    fields = list(corpus[0])

    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in corpus:
        cells[(r["relation_label"], r["ambiguity_level"])].append(r)

    sizes = {len(v) for v in cells.values()}
    if len(sizes) != 1:
        # 'equals' cannot be filled to the same depth as the rest: genuine
        # coincident pairs are finite, so its cells are shallower by design.
        print(f"  ragged grid, cell sizes {sorted(sizes)} — taking one training "
              f"row per cell and the remainder for evaluation")

    n_eval = spec["eval_per_cell"]
    train, evalr = [], []
    for k in sorted(cells):
        rows = cells[k]
        if len(rows) <= n_eval:
            print(f"  cell {k} has only {len(rows)} rows; holding out "
                  f"{max(1, len(rows) // 2)} for eval")
            cut = max(1, len(rows) // 2)
        else:
            cut = n_eval
        evalr.extend(rows[:cut])
        train.extend(rows[cut:])

    def pair(r):
        return (r["source_entity"].lower(), r["target_entity"].lower())
    tp = {pair(r) for r in train}
    leak = [r for r in evalr if pair(r) in tp or (pair(r)[1], pair(r)[0]) in tp]
    if leak:
        print(f"  {len(leak)} eval pair(s) also appear in train (or mirrored) — aborting")
        return 1

    facts: dict[tuple, str] = {}
    entries = []
    for i, r in enumerate(evalr):
        fk = (r["source_entity"], r["target_entity"], r["relation_label"])
        fid = facts.setdefault(fk, f"f{len(facts):04d}")
        entries.append({"row_index": i, "fact_id": fid,
                        "subject": r["source_entity"], "target": r["target_entity"],
                        "label": r["relation_label"],
                        "ambiguity_level": r["ambiguity_level"],
                        "row_sha256": row_hash(r)})
    man_sha = hashlib.sha256("".join(e["row_sha256"] for e in entries).encode()).hexdigest()

    # Three demonstrations per eval row, same label, drawn from the training
    # pool. Seeding from the row index keeps the draw identical across every
    # arm, so a difference between strategies cannot come from different demos.
    import random
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(train):
        by_label[r["relation_label"]].append(i)
    demos = {}
    for i, r in enumerate(evalr):
        pool = by_label.get(r["relation_label"], [])
        if len(pool) < SHOTS:
            print(f"  only {len(pool)} training rows for {r['relation_label']} "
                  f"— cannot draw {SHOTS} demos")
            return 1
        rng = random.Random(f"{DEMO_SEED}:{args.relation}:{i}")
        demos[str(i)] = sorted(rng.sample(pool, SHOTS))
    demo_sha = hashlib.sha256(json.dumps(demos, sort_keys=True).encode()).hexdigest()

    print(f"  {args.relation}: train {len(train)}, eval {len(evalr)}, "
          f"{SHOTS} shots")
    print(f"  eval sha {man_sha[:12]}   demo sha {demo_sha[:12]}")
    if args.dry_run:
        return 0

    write(data / "train.csv", train, fields)
    write(data / "eval.csv", evalr, fields)
    (data / "eval_manifest.json").write_text(json.dumps({
        "domain": spec["domain"], "source_csv": f"data/{args.relation}/eval.csv",
        "n_rows": len(entries), "n_unique_facts": len(facts),
        "duplicate_rows": len(entries) - len(facts),
        "manifest_sha256": man_sha,
        "contract": {
            "every_experiment_must": [
                "evaluate exactly these row_index values, in this order",
                "apply NO geocodability filtering: the OSM cache is mutable, so "
                "filtering at run time makes the eval set differ between arms "
                "run before and after a cache re-warm",
                "verify manifest_sha256 before running"],
            "analysis_note": "every row asserts a distinct fact; no fact_id "
                             "clustering is needed for this relation",
            "ground_truth": spec["ground_truth"]},
        "rows": entries}, indent=2) + "\n", encoding="utf-8")
    (data / "fewshot_manifest.json").write_text(json.dumps({
        "domain": spec["domain"], "shots": SHOTS,
        "train_csv": f"data/{args.relation}/train.csv", "train_rows": len(train),
        "selection_rule": f"{SHOTS} training rows carrying the eval row's "
                          f"label, drawn by an RNG seeded from base seed "
                          f"{DEMO_SEED}, the relation and the row index, so "
                          f"every arm sees the same demonstrations.",
        "eval_manifest_sha256": man_sha, "demo_map_sha256": demo_sha,
        "warning": "demos are label-conditioned by design: they reveal the "
                   "answer class. Few-shot numbers are a leakage-aware probe, "
                   "comparable to zero-shot, NOT across labels. The pool is "
                   "also the fine-tuning set, so a fine-tuned model's few-shot "
                   "result is optimistic and must be reported separately from "
                   "the base model's.",
        "contract": ["every few-shot arm must use exactly these demo indices"],
        "demos": demos}, indent=2) + "\n", encoding="utf-8")
    for stale in ("eval_indices.json",):
        if (data / stale).exists():
            (data / stale).unlink()
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
