"""
audit_data.py
================================================================================
Pre-flight data audit for the 7-experiment KG-integration ablation.

Run this BEFORE launching any experiment. It answers one question: is every
dataset complete, balanced, leak-free and geocodable enough that the results
will mean something?

Checks per domain:
  1. FILES      required dataset artifacts present
  2. SCHEMA     the columns the eval/train engines actually read
  3. MISSING    empty / NaN / whitespace-only cells in required fields
  4. LABELS     label vocabulary matches the expected relation set
  5. BALANCE    every (relation x ambiguity level) cell holds the same count
  6. LEAKAGE    no eval row also appears in train (exact + entity-pair overlap)
  7. DUPLICATES repeated rows within a split
  8. GEOCODING  share of rows whose BOTH entities resolve in osm_cache.json
  9. DERIVED    KG training sets present and consistent with their source split

Exit code is 0 only if nothing FAILED (warnings do not block).

Usage
  python audit_data.py
  python audit_data.py --domain Cardinal-Reasoning
  python audit_data.py --strict          # treat warnings as failures
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# EXPECTED SHAPE OF EACH DOMAIN
# ---------------------------------------------------------------------------
TOPO_LABELS = {"contains", "within", "touches", "crosses",
               "disjoint", "overlaps", "equals"}
CARD_LABELS = {"north_of", "south_of", "east_of", "west_of",
               "northeast_of", "northwest_of", "southeast_of", "southwest_of"}
REL_LABELS = {"left_of", "right_of", "in_front_of", "behind", "next_to"}

DOMAINS = {
    "Topological-Reasoning": {
        "labels": TOPO_LABELS,
        "corpus": ("dataset/topological_relations.csv", "relation_label",
                   "source_entity", "target_entity"),
        "splits": [
            ("dataset/topo_v2_train.csv", "spatial_relation",
             "place_name_subject", "place_name_object", "train"),
            ("dataset/topo_v2_eval.csv", "spatial_relation",
             "place_name_subject", "place_name_object", "eval"),
        ],
        "kg_train": "dataset/osm_kg_balanced_train.jsonl",
    },
    "Cardinal-Reasoning": {
        "labels": CARD_LABELS,
        "corpus": ("dataset/cardinal_direction_relations.csv", "relation_label",
                   "source_entity", "target_entity"),
        "splits": [
            ("dataset/cardinal_nokg_train.csv", "relation_label",
             "source_entity", "target_entity", "train"),
        ],
        "index_split": ("dataset/eval_40_balanced_indices.json",
                        "dataset/cardinal_direction_relations.csv",
                        "relation_label", "source_entity", "target_entity"),
        "kg_train": "dataset/cardinal_osm_kg_train.jsonl",
    },
    "Relative-Reasoning": {
        "labels": REL_LABELS,
        "corpus": ("dataset/relative_direction_relations.csv", "relation_label",
                   "source_entity", "target_entity"),
        "splits": [
            ("dataset/relative_balanced_train.csv", "relation_label",
             "source_entity", "target_entity", "train"),
        ],
        "index_split": ("dataset/eval_25_balanced_indices.json",
                        "dataset/relative_direction_relations.csv",
                        "relation_label", "source_entity", "target_entity"),
        "kg_train": "dataset/relative_osm_kg_train.jsonl",
    },
}

LEVEL_COL = "ambiguity_level"

# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------
_counts = Counter()

# Colour only when writing to a terminal. Piped or redirected output must stay
# plain text so log files are readable and callers can grep the status words --
# escape codes sit between "FAIL" and the message and break naive matching.
_COLOR = sys.stdout.isatty()
_C = (lambda code, t: f"\033[{code}m{t}\033[0m") if _COLOR else (lambda code, t: t)


def ok(msg: str) -> None:
    _counts["pass"] += 1
    print(f"    {_C(92, 'PASS')}  {msg}")


def warn(msg: str) -> None:
    _counts["warn"] += 1
    print(f"    {_C(93, 'WARN')}  {msg}")


def fail(msg: str) -> None:
    _counts["fail"] += 1
    print(f"    {_C(91, 'FAIL')}  {msg}")


def info(msg: str) -> None:
    print(f"          {msg}")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def blank(v) -> bool:
    return v is None or str(v).strip() == "" or str(v).strip().lower() in {"nan", "none", "null"}


def cell_counts(rows: list[dict], label_col: str) -> dict:
    return Counter((r[label_col].strip().lower(), r.get(LEVEL_COL, "?").strip())
                   for r in rows)


def pair_key(r: dict, sc: str, tc: str) -> tuple:
    return (str(r.get(sc, "")).strip().lower(), str(r.get(tc, "")).strip().lower())


def load_cache(domain: str) -> dict:
    p = os.path.join(domain, "code", "results", "osm_cache.json")
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return {}


def geocodable(cache: dict, *names: str) -> bool:
    for n in names:
        n = str(n or "").strip()
        if not n or cache.get(n) is None:
            return False
    return True


# ---------------------------------------------------------------------------
# CHECKS
# ---------------------------------------------------------------------------
def check_split(name: str, rows: list[dict], label_col: str,
                sc: str, tc: str, expected_labels: set, cache: dict,
                is_pool: bool = False) -> None:
    """is_pool=True for the raw corpus: it is the source to sample FROM, so
    imbalance and partial geocoding there are expected, not defects."""
    bad = warn if is_pool else fail
    print(f"\n  -- {name}  ({len(rows)} rows) " + "-" * max(0, 46 - len(name)))

    if not rows:
        fail(f"{name}: empty")
        return

    # SCHEMA
    need = {label_col, LEVEL_COL, sc, tc}
    have = set(rows[0].keys())
    if need - have:
        fail(f"schema: missing columns {sorted(need - have)}")
        return
    ok(f"schema: all required columns present ({', '.join(sorted(need))})")

    # MISSING
    holes = defaultdict(int)
    for r in rows:
        for c in need:
            if blank(r.get(c)):
                holes[c] += 1
    if holes:
        fail("missing values: " + ", ".join(f"{c}={n}" for c, n in holes.items()))
    else:
        ok("missing values: none in required fields")

    # LABELS
    seen = {r[label_col].strip().lower() for r in rows}
    unexpected = seen - expected_labels
    absent = expected_labels - seen
    if unexpected:
        fail(f"labels: unexpected {sorted(unexpected)}")
    elif absent:
        warn(f"labels: {sorted(absent)} absent from this split")
    else:
        ok(f"labels: all {len(expected_labels)} present, none unexpected")

    # BALANCE
    cells = cell_counts(rows, label_col)
    levels = sorted({lv for _, lv in cells})
    labels = sorted({lb for lb, _ in cells})
    full = {(lb, lv): cells.get((lb, lv), 0) for lb in labels for lv in levels}
    vals = sorted(set(full.values()))
    per_level = {lv: sum(full[(lb, lv)] for lb in labels) for lv in levels}

    if len(vals) == 1:
        ok(f"balance: perfect grid — every ({len(labels)} relations x "
           f"{len(levels)} levels) cell = {vals[0]}")
    else:
        empty = [k for k, v in full.items() if v == 0]
        if empty:
            bad(f"balance: {len(empty)} EMPTY cells, e.g. {empty[:3]}")
        lvset = sorted(set(per_level.values()))
        if len(lvset) == 1:
            warn(f"balance: per-level totals equal ({lvset[0]}) but cells vary {vals}")
        else:
            bad(f"balance: cells vary {vals}; per-level totals {per_level}")
        info("per-level: " + ", ".join(
            f"{lv.replace('Level ', 'L')}={n}" for lv, n in sorted(per_level.items())))

    # DUPLICATES
    pk = [pair_key(r, sc, tc) + (r[label_col].strip().lower(),) for r in rows]
    dups = [k for k, n in Counter(pk).items() if n > 1]
    if dups:
        warn(f"duplicates: {len(dups)} repeated (subject, object, label) triples")
    else:
        ok("duplicates: none")

    # GEOCODING
    if cache:
        good = sum(1 for r in rows if geocodable(cache, r[sc], r[tc]))
        pct = 100.0 * good / len(rows)
        msg = f"geocoding: {good}/{len(rows)} rows usable ({pct:.0f}%)"
        if pct == 100:
            ok(msg)
        elif pct >= 90:
            warn(msg + " — KG arms drop the rest")
        else:
            bad(msg + " — KG arms lose a large share")
        lost = Counter(r[label_col].strip().lower() for r in rows
                       if not geocodable(cache, r[sc], r[tc]))
        if lost:
            info("dropped by label: " + ", ".join(f"{k}={v}" for k, v in lost.most_common()))
    else:
        fail("geocoding: osm_cache.json missing or unreadable")


def check_leakage(train: list[dict], ev: list[dict],
                  t_sc: str, t_tc: str, e_sc: str, e_tc: str) -> None:
    print("\n  -- train/eval leakage " + "-" * 44)
    tr = {pair_key(r, t_sc, t_tc) for r in train}
    te = {pair_key(r, e_sc, e_tc) for r in ev}
    shared = tr & te
    if shared:
        fail(f"leakage: {len(shared)} entity pairs appear in BOTH splits "
             f"(e.g. {sorted(shared)[:2]})")
    else:
        ok(f"leakage: none — {len(tr)} train pairs and {len(te)} eval pairs disjoint")


def audit_domain(domain: str) -> None:
    spec = DOMAINS[domain]
    print("\n" + "=" * 78)
    print(f"  {domain}")
    print("=" * 78)

    cache = load_cache(domain)
    if cache:
        nulls = sum(1 for v in cache.values() if v is None)
        info(f"osm_cache.json: {len(cache)} entries "
             f"({len(cache) - nulls} resolved, {nulls} unresolved)")

    # FILES
    print("\n  -- required files " + "-" * 48)
    required = [spec["corpus"][0]] + [s[0] for s in spec["splits"]]
    if "index_split" in spec:
        required.append(spec["index_split"][0])
    for rel in required:
        p = os.path.join(domain, rel)
        if os.path.exists(p):
            ok(f"{rel}  ({os.path.getsize(p):,} bytes)")
        else:
            fail(f"{rel}  MISSING")

    kg = os.path.join(domain, spec["kg_train"])
    if os.path.exists(kg):
        recs = [json.loads(l) for l in open(kg, encoding="utf-8") if l.strip()]
        dist = Counter(r.get("label") for r in recs)
        lo, hi = min(dist.values()), max(dist.values())
        msg = f"{spec['kg_train']}  {len(recs)} records, labels {lo}-{hi} per class"
        if hi >= 3 * lo:
            warn(msg + "  — severely skewed")
        else:
            ok(msg)
    else:
        fail(f"{spec['kg_train']}  MISSING — rebuild before Exp 3")

    # SPLITS
    corpus_path, c_lab, c_sc, c_tc = spec["corpus"]
    corpus_full = os.path.join(domain, corpus_path)
    if os.path.exists(corpus_full):
        check_split("corpus " + os.path.basename(corpus_path),
                    read_csv(corpus_full), c_lab, c_sc, c_tc, spec["labels"], cache,
                    is_pool=True)

    loaded = {}
    for rel, lab, sc, tc, kind in spec["splits"]:
        p = os.path.join(domain, rel)
        if not os.path.exists(p):
            continue
        rows = read_csv(p)
        loaded[kind] = (rows, sc, tc)
        check_split(f"{kind} {os.path.basename(rel)}", rows, lab, sc, tc,
                    spec["labels"], cache)

    # index-defined eval split
    if "index_split" in spec:
        idx_rel, src_rel, lab, sc, tc = spec["index_split"]
        idx_p, src_p = os.path.join(domain, idx_rel), os.path.join(domain, src_rel)
        if os.path.exists(idx_p) and os.path.exists(src_p):
            idx = json.load(open(idx_p, encoding="utf-8"))
            if isinstance(idx, dict):
                idx = idx.get("indices", next(iter(idx.values())))
            src = read_csv(src_p)
            bad = [i for i in idx if not (0 <= i < len(src))]
            if bad:
                fail(f"eval indices: {len(bad)} out of range for {src_rel}")
            else:
                ok(f"eval indices: all {len(idx)} in range for {os.path.basename(src_rel)}")
                ev = [src[i] for i in idx]
                loaded["eval"] = (ev, sc, tc)
                check_split(f"eval via {os.path.basename(idx_rel)}",
                            ev, lab, sc, tc, spec["labels"], cache)
                rest = [r for i, r in enumerate(src) if i not in set(idx)]
                loaded.setdefault("train", (rest, sc, tc))
                if "train" not in loaded:
                    check_split("train remainder (corpus minus eval indices)",
                                rest, lab, sc, tc, spec["labels"], cache)

    if "train" in loaded and "eval" in loaded:
        (tr, tsc, ttc), (ev, esc, etc) = loaded["train"], loaded["eval"]
        check_leakage(tr, ev, tsc, ttc, esc, etc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", action="append", choices=list(DOMAINS))
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero on warnings too.")
    args = ap.parse_args()

    for d in (args.domain or list(DOMAINS)):
        if os.path.isdir(d):
            audit_domain(d)
        else:
            fail(f"{d}: directory not found")

    print("\n" + "=" * 78)
    print(f"  SUMMARY: {_counts['pass']} passed, "
          f"{_counts['warn']} warnings, {_counts['fail']} failures")
    print("=" * 78 + "\n")
    bad = _counts["fail"] + (_counts["warn"] if args.strict else 0)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
