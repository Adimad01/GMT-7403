"""Rebuild the cardinal train/eval split and both manifests.

The corpus was regenerated from scratch, which invalidates every pinned
artefact that referenced the old rows. This rebuilds them together so they
cannot drift apart: an eval manifest that names the rows and hashes their
content, and a few-shot manifest keyed to that eval manifest's hash.

The grid is exactly 3 rows per (label, level) cell, so the split takes one row
from each cell for training and the other two for evaluation. That keeps both
sides balanced and makes demo selection deterministic -- for any eval row there
is exactly one training row per level carrying the same label, so there is
nothing to sample and no seed to get wrong.

    python3 scripts/build_cardinal_splits.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "cardinal"
LEVELS = [f"Level {i}" for i in range(1, 7)]


def read(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(p: Path, rows: list[dict], fields: list[str]) -> None:
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def row_hash(r: dict) -> str:
    blob = "|".join(str(r[k]) for k in sorted(r))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def main() -> int:
    corpus = read(DATA / "corpus.csv")
    fields = list(corpus[0])

    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in corpus:
        cells[(r["relation_label"], r["ambiguity_level"])].append(r)

    bad = {k: len(v) for k, v in cells.items() if len(v) != 3}
    if bad:
        print(f"  expected 3 rows per cell; found {bad}")
        return 1

    train, evalr = [], []
    for key in sorted(cells):
        rows = cells[key]
        train.append(rows[0])
        evalr.extend(rows[1:])

    # A pair appearing on both sides would let a demo answer its own eval row.
    def pair(r):
        return (r["source_entity"].lower(), r["target_entity"].lower())
    tp = {pair(r) for r in train}
    overlap = [r for r in evalr if pair(r) in tp or (pair(r)[1], pair(r)[0]) in tp]
    if overlap:
        print(f"  {len(overlap)} eval pair(s) also appear in train — aborting")
        return 1

    write(DATA / "train.csv", train, fields)
    write(DATA / "eval.csv", evalr, fields)

    # --- eval manifest: row_index is an offset into eval.csv ---------------
    facts: dict[tuple, str] = {}
    entries = []
    for i, r in enumerate(evalr):
        fk = (r["source_entity"], r["target_entity"], r["relation_label"])
        fid = facts.setdefault(fk, f"f{len(facts):04d}")
        entries.append({
            "row_index": i, "fact_id": fid,
            "subject": r["source_entity"], "target": r["target_entity"],
            "label": r["relation_label"], "ambiguity_level": r["ambiguity_level"],
            "row_sha256": row_hash(r),
        })
    man_sha = hashlib.sha256(
        "".join(e["row_sha256"] for e in entries).encode()).hexdigest()

    eval_manifest = {
        "domain": "Cardinal-Reasoning",
        "source_csv": "data/cardinal/eval.csv",
        "n_rows": len(entries),
        "n_unique_facts": len(facts),
        "duplicate_rows": len(entries) - len(facts),
        "manifest_sha256": man_sha,
        "contract": {
            "every_experiment_must": [
                "evaluate exactly these row_index values, in this order",
                "apply NO geocodability filtering: the OSM cache is mutable, so "
                "filtering at run time makes the eval set differ between arms "
                "run before and after a cache re-warm",
                "verify manifest_sha256 before running",
            ],
            "analysis_note": "every row asserts a distinct fact; no fact_id "
                             "clustering is needed for this relation",
            "ground_truth": "labels are cone-based cardinal sectors computed "
                            "from city centroids, each at least 8 degrees "
                            "inside its 45-degree sector, agreeing with the "
                            "projection-based reading, and reciprocal",
        },
        "rows": entries,
    }
    (DATA / "eval_manifest.json").write_text(
        json.dumps(eval_manifest, indent=2) + "\n", encoding="utf-8")

    # --- few-shot manifest -------------------------------------------------
    by_label_level = {(r["relation_label"], r["ambiguity_level"]): i
                      for i, r in enumerate(train)}
    demos = {}
    for i, r in enumerate(evalr):
        lab = r["relation_label"]
        demos[str(i)] = [by_label_level[(lab, lv)] for lv in LEVELS]
    demo_sha = hashlib.sha256(
        json.dumps(demos, sort_keys=True).encode()).hexdigest()

    fewshot = {
        "domain": "Cardinal-Reasoning",
        "shots": len(LEVELS),
        "train_csv": "data/cardinal/train.csv",
        "train_rows": len(train),
        "selection_rule": "the unique training row at each ambiguity level "
                          "L1-L6 carrying the eval row's label. The split "
                          "leaves exactly one candidate per cell, so the "
                          "choice is deterministic and needs no seed.",
        "eval_manifest_sha256": man_sha,
        "demo_map_sha256": demo_sha,
        "warning": "demos are label-conditioned by design: they reveal the "
                   "answer class. Few-shot numbers are a leakage-aware probe, "
                   "comparable to zero-shot, NOT across labels.",
        "contract": ["every few-shot arm must use exactly these demo indices"],
        "demos": demos,
    }
    (DATA / "fewshot_manifest.json").write_text(
        json.dumps(fewshot, indent=2) + "\n", encoding="utf-8")

    stale = DATA / "eval_indices.json"
    if stale.exists():
        stale.unlink()

    print(f"  train {len(train)} rows, eval {len(evalr)} rows")
    print(f"  eval manifest sha {man_sha[:12]}   demo map sha {demo_sha[:12]}")
    print(f"  {len(LEVELS)} shots per few-shot item, deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
