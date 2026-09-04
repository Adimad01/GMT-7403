"""Choose a balanced set of cardinal items from verified pairs.

Difficulty for cardinal cannot be read off the geometry the way it can for the
other two relations. Once the direction is removed from the wording -- and it
must be, or the item answers itself -- what is left to vary is whether the
reader knows where the two places are. A surprising north/south pair is not
available either: a pair that contradicts a mental map almost always carries a
large longitude gap, which makes it a diagonal rather than an axis case.

So the level is set by how obscure the harder of the two places is, measured by
Nominatim's published importance score rather than by my own sense of which
cities are famous. Cut points are the quintiles of the observed distribution,
so every cell has candidates by construction.

    python3 data_generation/select_cardinal_items.py --per-cell 7
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_cardinal_truth import COORDS, separation                   # noqa: E402
from build_cardinal_corpus import check                               # noqa: E402

REPO = Path(__file__).resolve().parents[1]
IMP = REPO / "data" / "cardinal" / "osm" / "importance.json"
OUT = REPO / "data_generation" / "cardinal_picks.json"

LABELS = ["north_of", "south_of", "east_of", "west_of",
          "northeast_of", "northwest_of", "southeast_of", "southwest_of"]


def load_importance() -> tuple[dict, list[float]]:
    raw = json.loads(IMP.read_text())
    imp = {k: v["importance"] for k, v in raw.items() if v}
    vals = sorted(imp.values())
    cuts = [vals[int(len(vals) * q)] for q in (0.8, 0.6, 0.4, 0.2)]
    return imp, cuts


def level_of(a: str, b: str, imp: dict, cuts: list[float]) -> int | None:
    """Level 1 when both places are prominent, 5 when the harder one is obscure."""
    if a not in imp or b not in imp:
        return None
    m = min(imp[a], imp[b])
    for lvl, c in enumerate(cuts, start=1):
        if m >= c:
            return lvl
    return 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=7)
    args = ap.parse_args()

    imp, cuts = load_importance()
    print(f"  importance quintile cuts: {[round(c, 3) for c in cuts]}")
    names = [c for c in sorted(COORDS) if c in imp]
    print(f"  {len(names)} cities with a prominence score")

    used: set[frozenset] = set()
    load = Counter()
    picks = []

    def take(label, level, a, b, via=None):
        picks.append({"label": label, "level": level, "subject": a,
                      "object": b, "via": via})
        used.add(frozenset((a, b)))
        for n in (a, b) + ((via,) if via else ()):
            load[n] += 1

    short = []
    for label in LABELS:
        pool = []
        for a in names:
            for b in names:
                if a == b or frozenset((a, b)) in used:
                    continue
                lv = level_of(a, b, imp, cuts)
                if lv is None or check(a, b, label):
                    continue
                pool.append((lv, a, b))
        by_level: dict[int, list] = {}
        for lv, a, b in pool:
            by_level.setdefault(lv, []).append((a, b))
        for level in range(1, 6):
            cands = by_level.get(level, [])
            got = 0
            while got < args.per_cell and cands:
                cands.sort(key=lambda t: load[t[0]] + load[t[1]])
                a, b = cands[0]
                if frozenset((a, b)) in used:
                    cands.pop(0)
                    continue
                take(label, level, a, b)
                got += 1
                cands = [t for t in cands if frozenset(t) not in used]
            if got < args.per_cell:
                short.append((label, f"L{level}", got))

    # --- Level 6: A -> C -> B, both steps carrying the same label ----------
    for label in LABELS:
        good: dict[str, list[str]] = {}
        for a in names:
            for b in names:
                if a != b and not check(a, b, label):
                    good.setdefault(a, []).append(b)
        cands = []
        for a, mids in good.items():
            for c in mids:
                for b in good.get(c, []):
                    if b != a and not check(a, b, label):
                        cands.append((separation(COORDS[a], COORDS[b]), a, c, b))
        got = 0
        while got < args.per_cell and cands:
            cands.sort(key=lambda t: load[t[1]] + load[t[2]] + load[t[3]])
            for _, a, c, b in cands:
                if frozenset((a, b)) in used:
                    continue
                take(label, 6, a, b, via=c)
                got += 1
                break
            else:
                break
            cands = [t for t in cands if frozenset((t[1], t[3])) not in used]
        if got < args.per_cell:
            short.append((label, "L6", got))

    OUT.write_text(json.dumps(picks, indent=1) + "\n")
    print(f"  selected {len(picks)} items -> {OUT.name}")
    print(f"  distinct cities {len(load)}, busiest {load.most_common(3)}")
    if short:
        print(f"\n  cells short of {args.per_cell}:")
        for lab, lv, n in short:
            print(f"    {lab:<13} {lv}  got {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
