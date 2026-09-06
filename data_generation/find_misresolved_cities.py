"""Flag cities whose resolved position contradicts the company they arrived in.

A name check confirms that a result carries the name asked for. It cannot tell
which of several places bearing that name was meant, and importance picks the
wrong one whenever an obscure homonym happens to outrank the intended city:
Tarawa resolved to a village in Nigeria rather than the Kiribati capital, and
Mary to a hamlet in Burgundy rather than the Turkmen city.

Nothing in the record exposes that. But the source list arrived grouped by
region -- a run of Pacific capitals, a run of Siberian cities -- so a place
that lands thousands of kilometres from its neighbours in the file is almost
certainly the wrong homonym. This measures that: for each city, the distance
from the median position of the entries around it.

    python3 data_generation/find_misresolved_cities.py accepted_cities.txt
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_cardinal_truth import COORDS, key, separation   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?",
                    default="data_generation/accepted_cities.txt")
    ap.add_argument("--window", type=int, default=9,
                    help="how many neighbouring entries define the region")
    ap.add_argument("--threshold", type=float, default=45.0,
                    help="degrees from the neighbourhood median to flag")
    args = ap.parse_args()

    names = [l.split("|")[0].strip()
             for l in Path(args.file).read_text(encoding="utf-8").splitlines()
             if "|" in l]
    pos = [(n, COORDS.get(key(n))) for n in names]
    known = [(i, n, p) for i, (n, p) in enumerate(pos) if p]
    print(f"  {len(known)} of {len(names)} cities have coordinates")

    half = args.window // 2
    flagged = []
    for idx, (i, name, p) in enumerate(known):
        lo, hi = max(0, idx - half), min(len(known), idx + half + 1)
        neigh = [q for j, (_, _, q) in enumerate(known[lo:hi], lo) if j != idx]
        if len(neigh) < 4:
            continue
        med = (statistics.median(q[0] for q in neigh),
               statistics.median(q[1] for q in neigh))
        d = separation(p, med)
        if d > args.threshold:
            flagged.append((d, name, p, med))

    flagged.sort(reverse=True)
    print(f"  {len(flagged)} city(ies) sit more than {args.threshold:.0f} deg "
          f"from their neighbours in the list\n")
    for d, name, p, med in flagged:
        print(f"    {d:5.0f} deg  {name}")
        print(f"               resolved {p[0]:+.2f},{p[1]:+.2f}   "
              f"neighbourhood {med[0]:+.1f},{med[1]:+.1f}")
    if flagged:
        print("\n  Each of these is probably a different place of the same name.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
