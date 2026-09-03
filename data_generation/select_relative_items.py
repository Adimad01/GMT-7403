"""Pick a balanced, non-overlapping set of relative-direction triples.

Supply is not the problem -- there are hundreds of thousands of sound triples.
The problem is choosing a set that is balanced across every cell, never reuses
a subject/target pair or its mirror, and spreads the observers and cities
around so the corpus does not lean on a handful of places.

    python3 data_generation/select_relative_items.py --per-cell 5
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_cardinal_truth import COORDS, separation                  # noqa: E402
from check_relative_truth import classify, rotation_level            # noqa: E402
from find_relative_items import famous                               # noqa: E402

LABELS = ["left_of", "right_of", "in_front_of", "behind", "next_to"]
HOP_LABELS = ["left_of", "right_of", "in_front_of", "behind"]
MARGIN = 12.0


def build_index(names):
    """observer -> {(subject, target): (label, margin)} for sound triples."""
    idx = {}
    for v in names:
        pv = COORDS[v]
        near = [n for n in names if n != v and 3.0 <= separation(pv, COORDS[n]) <= 70.0]
        cell = {}
        for b in near:
            for a in near:
                if a == b:
                    continue
                g, m = classify(v, a, b)
                if g and m >= MARGIN:
                    cell[(a, b)] = (g, m)
        idx[v] = cell
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=5)
    ap.add_argument("--out", default="data_generation/relative_picks.json")
    args = ap.parse_args()

    # 'next_to' needs two cities close together as seen from far off, and the
    # famous-city pool is too sparse in such pairs to fill every level. It is
    # also the one label whose difficulty does not come from frame rotation, so
    # widening its pool costs nothing and buys the variety the others get for
    # free.
    names = famous()
    idx = build_index(names)
    wide = build_index(sorted(COORDS))

    used_pairs: set[tuple[str, str]] = set()
    obs_load, city_load = Counter(), Counter()
    picks = []

    def take(cand, level, label, via=None):
        v, a, b = cand
        picks.append({"level": level, "label": label, "observer": v,
                      "subject": a, "target": b, "via": via})
        used_pairs.add((a, b))
        used_pairs.add((b, a))
        obs_load[v] += 1
        for c in (v, a, b) + ((via,) if via else ()):
            city_load[c] += 1

    def cost(v, a, b, via=None):
        parts = [v, a, b] + ([via] if via else [])
        return obs_load[v] * 3 + sum(city_load[c] for c in parts)

    # --- Levels 1-5 --------------------------------------------------------
    for label in LABELS:
        for level in range(1, 6):
            cands = []
            source = wide if label == "next_to" else idx
            for v, cell in source.items():
                for (a, b), (g, m) in cell.items():
                    if g != label or rotation_level(v, b) != level:
                        continue
                    if (a, b) in used_pairs:
                        continue
                    cands.append((v, a, b))
            chosen = 0
            while chosen < args.per_cell and cands:
                cands.sort(key=lambda t: cost(*t))
                for cand in cands:
                    if (cand[1], cand[2]) in used_pairs:
                        continue
                    take(cand, level, label)
                    chosen += 1
                    break
                else:
                    break
                cands = [c for c in cands if (c[1], c[2]) not in used_pairs]
            if chosen < args.per_cell:
                print(f"  SHORT: {label} L{level} got {chosen}/{args.per_cell}")

    # --- Level 6: A | B | C under one observer -----------------------------
    for label in HOP_LABELS:
        cands = []
        for v, cell in idx.items():
            same = [(a, b) for (a, b), (g, _) in cell.items() if g == label]
            bymid = {}
            for a, b in same:
                bymid.setdefault(b, []).append(a)
            for (b, c) in same:
                for a in bymid.get(b, []):
                    if a == c or cell.get((a, c), (None,))[0] != label:
                        continue
                    if (a, c) in used_pairs:
                        continue
                    cands.append((v, a, c, b))
            if len(cands) > 4000:
                break
        chosen = 0
        while chosen < args.per_cell and cands:
            cands.sort(key=lambda t: cost(t[0], t[1], t[2], t[3]))
            for v, a, c, b in cands:
                if (a, c) in used_pairs:
                    continue
                take((v, a, c), 6, label, via=b)
                chosen += 1
                break
            else:
                break
            cands = [t for t in cands if (t[1], t[2]) not in used_pairs]
        if chosen < args.per_cell:
            print(f"  SHORT: {label} L6 got {chosen}/{args.per_cell}")

    Path(args.out).write_text(json.dumps(picks, indent=1) + "\n")
    print(f"  selected {len(picks)} items -> {args.out}")
    print(f"  distinct observers {len(obs_load)}, distinct cities {len(city_load)}")
    print(f"  busiest observer {obs_load.most_common(1)}, busiest city {city_load.most_common(1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
