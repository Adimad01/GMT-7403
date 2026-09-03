"""Rebuild a relation's train/eval split and both manifests together.

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

# demo_levels: which ambiguity levels supply few-shot demonstrations. Cardinal
# carries Level 6 for every label so it can use all six; relative has no Level 6
# for next_to, and a shot count that varied by label would make the arms
# incomparable, so it uses five.
SPEC = {
    "cardinal": dict(domain="Cardinal-Reasoning", demo_levels=range(1, 7),
                     ground_truth="labels are cone-based cardinal sectors "
                                  "computed from city centroids, each at least "
                                  "8 degrees inside its 45-degree sector, "
                                  "agreeing with the projection-based reading, "
                                  "and reciprocal"),
    "relative": dict(domain="Relative-Reasoning", demo_levels=range(1, 6),
                     ground_truth="labels are derived from the stated observer "
                                  "frame: the angle of the subject off the "
                                  "observer's sight line to the target, plus "
                                  "relative depth. The ambiguity level is the "
                                  "rotation of that sight line from north"),
}


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
        print(f"  cells are not a regular grid: sizes {sorted(sizes)}")
        return 1

    train, evalr = [], []
    for k in sorted(cells):
        train.append(cells[k][0])
        evalr.extend(cells[k][1:])

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

    levels = [f"Level {i}" for i in spec["demo_levels"]]
    by_cell = {(r["relation_label"], r["ambiguity_level"]): i
               for i, r in enumerate(train)}
    demos = {}
    for i, r in enumerate(evalr):
        lab = r["relation_label"]
        picked = [by_cell[(lab, lv)] for lv in levels if (lab, lv) in by_cell]
        if len(picked) != len(levels):
            print(f"  {lab} lacks a training row at every demo level — aborting")
            return 1
        demos[str(i)] = picked
    demo_sha = hashlib.sha256(json.dumps(demos, sort_keys=True).encode()).hexdigest()

    print(f"  {args.relation}: train {len(train)}, eval {len(evalr)}, "
          f"{len(levels)} shots")
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
        "domain": spec["domain"], "shots": len(levels),
        "train_csv": f"data/{args.relation}/train.csv", "train_rows": len(train),
        "selection_rule": "the unique training row at each demo level carrying "
                          "the eval row's label. The split leaves exactly one "
                          "candidate per cell, so the choice is deterministic "
                          "and needs no seed.",
        "eval_manifest_sha256": man_sha, "demo_map_sha256": demo_sha,
        "warning": "demos are label-conditioned by design: they reveal the "
                   "answer class. Few-shot numbers are a leakage-aware probe, "
                   "comparable to zero-shot, NOT across labels.",
        "contract": ["every few-shot arm must use exactly these demo indices"],
        "demos": demos}, indent=2) + "\n", encoding="utf-8")
    for stale in ("eval_indices.json",):
        if (data / stale).exists():
            (data / stale).unlink()
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
