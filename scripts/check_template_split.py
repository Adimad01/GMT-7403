"""Fail if a training description shares a core sentence with an evaluation description.

A string-level split is not enough. The topological bank was built by combining an opening phrase,
a core sentence and a filler ("the region of {B}"), so "Visually, {A} completely swallows {B}" in
training and "Geographically, {A} completely swallows the region of {B}" in evaluation were two
different strings carrying the same sentence -- and 87 % of evaluation rows had such a twin in
training. This compares cores, after stripping the opener and the filler, at two places:

- the template pools in data_generation/paraphrases_<relation>.json;
- the rendered rows of train.csv and eval.csv, with each row's places turned back into slots,
  which also catches two templates that only coincide once places are filled in.

    python3 scripts/check_template_split.py            # all three relations
    python3 scripts/check_template_split.py -r topological
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OPENER = re.compile(r"^(looking at the map|in spatial terms|geographically|visually|of the two),\s*")
FILLER = re.compile(r"the (region|limits) of \{b\}")
SLOTS = (("source_entity", "{A}"), ("target_entity", "{B}"),
         ("via_entity", "{C}"), ("observer_entity", "{V}"))
QUALIFIERS = ("City of ", "State of ", "Province of ", "Borough of ")


def core(t: str) -> str:
    t = OPENER.sub("", t.strip().lower())
    t = FILLER.sub("{b}", t)
    return re.sub(r"[^a-z0-9{}]", "", t)


def shape(r: dict) -> str:
    t = r["corpus"]
    for col, slot in SLOTS:
        v = (r.get(col) or "").strip()
        if v:
            t = t.replace(v, slot)
            for q in QUALIFIERS:
                if v.startswith(q):
                    t = t.replace(v[len(q):], slot)
    return t


def check(rel: str) -> int:
    pools = json.loads((REPO / "data_generation" / f"paraphrases_{rel}.json").read_text())
    ev = {core(t) for ts in pools["eval"].values() for t in ts}
    pool_hits = sorted({k for k, ts in pools["train"].items() for t in ts if core(t) in ev})

    read = lambda n: list(csv.DictReader((REPO / "data" / rel / n).open(encoding="utf-8")))
    ev_rows = read("eval.csv")
    ev_shapes = {core(shape(r)) for r in ev_rows}
    tr_rows = read("train.csv")
    row_hits = [i for i, r in enumerate(tr_rows) if core(shape(r)) in ev_shapes]

    n_ev_shared = sum(core(shape(r)) in {core(shape(t)) for t in tr_rows} for r in ev_rows)
    print(f"{rel:12s} pool cells sharing a core: {len(pool_hits):3d}   "
          f"train rows sharing a core with eval: {len(row_hits):3d}/{len(tr_rows)}   "
          f"eval rows with a twin in train: {n_ev_shared}/{len(ev_rows)}")
    for k in pool_hits[:8]:
        print(f"    pool  {k}")
    for i in row_hits[:8]:
        print(f"    row {i:4d}  {tr_rows[i]['corpus'][:90]}")
    return len(pool_hits) + len(row_hits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--relation", choices=["topological", "cardinal", "relative"])
    a = ap.parse_args()
    rels = [a.relation] if a.relation else ["topological", "cardinal", "relative"]
    bad = sum(check(r) for r in rels)
    print("OK" if not bad else f"FAIL: {bad} overlap(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
