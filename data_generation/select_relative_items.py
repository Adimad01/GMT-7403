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


def build_index(names, observers=None):
    """observer -> {(subject, target): (label, margin)} for sound triples.

    Every city can be a subject or a target, but the observer pool is capped.
    The triple count grows with the cube of the catalogue, and at 500 cities the
    full enumeration is 125 million combinations -- more memory than the result
    is worth, when a few hundred viewpoints already give far more candidates
    than any cell can use.
    """
    idx = {}
    for v in (observers if observers is not None else names):
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
    ap.add_argument("--max-observers", type=int, default=260,
                    help="cap the observer pool; subjects and targets still "
                         "range over every city")
    ap.add_argument("--out", default="data_generation/relative_picks.json")
    args = ap.parse_args()

    # 'next_to' needs two cities close together as seen from far off, and the
    # famous-city pool is too sparse in such pairs to fill every level. It is
    # also the one label whose difficulty does not come from frame rotation, so
    # widening its pool costs nothing and buys the variety the others get for
    # free.
    import random as _random
    # Relative grades difficulty by how far the sight line is turned from
    # north, not by how well known the places are, so there is no reason to
    # restrict it to the recognisable subset the way cardinal does. Every city
    # in the table is eligible.
    names = sorted(COORDS)
    allc = names
    rng = _random.Random(20260903)
    obs = names if len(names) <= args.max_observers else rng.sample(
        sorted(names), args.max_observers)
    obs_wide = allc if len(allc) <= args.max_observers else rng.sample(
        allc, args.max_observers)
    idx = build_index(names, obs)
    wide = build_index(allc, obs_wide)

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
    # A pair can qualify at several levels depending on where the observer
    # stands, and each pair may be used once. Filling cells in label order lets
    # an abundant cell take pairs that a scarce one had no alternative to, which
    # is how next_to Level 5 ended up with nothing. Scarcest cells choose first.
    supply: dict[tuple[str, int], int] = {}
    for label in LABELS:
        source = wide if label == "next_to" else idx
        for level in range(1, 6):
            n = sum(1 for v, cell in source.items()
                    for (a, b), (g, _) in cell.items()
                    if g == label and rotation_level(v, b) == level)
            supply[(label, level)] = n
    order = sorted(supply, key=lambda k: supply[k])

    for label, level in order:
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
    # Chains are gathered from every viewpoint, with a cap on how many any one
    # can contribute. Stopping once a global total was reached meant the first
    # few observers supplied everything: one city ended up as the viewpoint for
    # all 140 Level 6 left/right rows, which a model could exploit without
    # doing any reasoning.
    PER_OBSERVER = 12
    for label in HOP_LABELS:
        cands = []
        for v, cell in idx.items():
            same = [(a, b) for (a, b), (g, _) in cell.items() if g == label]
            bymid = {}
            for a, b in same:
                bymid.setdefault(b, []).append(a)
            taken = 0
            for (b, c) in same:
                if taken >= PER_OBSERVER:
                    break
                for a in bymid.get(b, []):
                    if a == c or cell.get((a, c), (None,))[0] != label:
                        continue
                    if (a, c) in used_pairs:
                        continue
                    cands.append((v, a, c, b))
                    taken += 1
                    if taken >= PER_OBSERVER:
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
