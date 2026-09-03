"""Choose a balanced set of topological items from the computed relations.

Two constraints do most of the work here. Entity pairs must not repeat, and a
pair must never appear alongside its mirror: contains(A,B) and within(B,A) are
the same geographic fact stated twice, so putting one in train and the other
in eval hands the model its answer. The old corpus had 88 such mirrors
internally and 7 across the split.

Level 6 chains are composed, not searched for by name:

    contains   A > B > C  =>  A > C
    within     A < B < C  =>  A < C
    disjoint   A < B, B disjoint C  =>  A disjoint C

Each is forced. Plain disjoint is not transitive -- two countries both
separate from a third tell you nothing about each other -- which is why the
disjoint chain routes through a containment step instead.

    python3 data_generation/select_topo_items.py --per-cell 5
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RELS = REPO / "data" / "topological" / "osm" / "relations.json"
OUT = REPO / "data_generation" / "topo_picks.json"

LABELS = ["contains", "within", "touches", "crosses", "overlaps", "disjoint", "equals"]
HOP_LABELS = ["contains", "within", "disjoint"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=5)
    args = ap.parse_args()

    rels = json.loads(RELS.read_text())
    by_cell: dict[tuple[str, int], list[dict]] = defaultdict(list)
    lookup: dict[tuple[str, str], str] = {}
    for r in rels:
        by_cell[(r["label"], r["level"])].append(r)
        lookup[(r["subject"], r["object"])] = r["label"]

    used: set[frozenset] = set()
    load = Counter()
    picks = []

    def cost(names):
        return sum(load[n] for n in names)

    def take(label, level, subj, obj, via=None):
        picks.append({"label": label, "level": level, "subject": subj,
                      "object": obj, "via": via})
        used.add(frozenset((subj, obj)))
        for n in (subj, obj) + ((via,) if via else ()):
            load[n] += 1

    short = []
    for label in LABELS:
        for level in range(1, 6):
            cands = [r for r in by_cell.get((label, level), [])
                     if frozenset((r["subject"], r["object"])) not in used]
            got = 0
            while got < args.per_cell and cands:
                cands.sort(key=lambda r: cost((r["subject"], r["object"])))
                r = cands[0]
                if frozenset((r["subject"], r["object"])) in used:
                    cands.pop(0)
                    continue
                take(label, level, r["subject"], r["object"])
                got += 1
                cands = [c for c in cands
                         if frozenset((c["subject"], c["object"])) not in used]
            if got < args.per_cell:
                short.append((label, f"L{level}", got))

    # --- Level 6 -----------------------------------------------------------
    contains = defaultdict(list)     # A -> [B] where A contains B
    for r in rels:
        if r["label"] == "contains":
            contains[r["subject"]].append(r["object"])

    for label in HOP_LABELS:
        cands = []
        if label in ("contains", "within"):
            for a, mids in contains.items():
                for c in mids:
                    for b in contains.get(c, []):
                        if b == a or lookup.get((a, b)) != "contains":
                            continue
                        cands.append((a, c, b) if label == "contains" else (b, c, a))
        else:                        # A within B, B disjoint C  =>  A disjoint C
            for b, mids in contains.items():
                for a in mids:
                    for c in [x["object"] for x in rels
                              if x["subject"] == b and x["label"] == "disjoint"]:
                        if lookup.get((a, c)) != "disjoint":
                            continue
                        cands.append((a, b, c))
        got = 0
        seen = set()
        while got < args.per_cell and cands:
            cands.sort(key=lambda t: cost(t))
            for subj, via, obj in cands:
                key = frozenset((subj, obj))
                if key in used or key in seen:
                    continue
                seen.add(key)
                take(label, 6, subj, obj, via=via)
                got += 1
                break
            else:
                break
            cands = [t for t in cands if frozenset((t[0], t[2])) not in used]
        if got < args.per_cell:
            short.append((label, "L6", got))

    OUT.write_text(json.dumps(picks, indent=1) + "\n")
    print(f"  selected {len(picks)} items -> {OUT.name}")
    print(f"  distinct entities {len(load)}, busiest {load.most_common(3)}")
    if short:
        print(f"\n  cells that could not be filled to {args.per_cell}:")
        for lab, lv, n in short:
            print(f"    {lab:<10} {lv}  got {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
