"""Find A -> C -> B triples where both hops and the composition all hold.

A multi-hop item is only sound if each stated link is true on its own AND the
composition is true, all three under the same constraints used for ordinary
rows. Searching for that by hand does not work -- 22 of my 24 first attempts
failed on at least one leg.

    python3 data_generation/find_cardinal_chains.py --label southwest_of --top 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cardinal_corpus import check                       # noqa: E402
from check_cardinal_truth import COORDS, separation           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--only", default="",
                    help="comma-separated substrings; endpoints must match one")
    args = ap.parse_args()

    names = sorted(COORDS)
    want = [s.strip() for s in args.only.split(",") if s.strip()]

    # pre-compute which ordered pairs are individually sound for this label
    good = {}
    for a in names:
        for b in names:
            if a != b and not check(a, b, args.label):
                good.setdefault(a, []).append(b)

    chains = []
    for a, mids in good.items():
        if want and not any(w in a for w in want):
            continue
        for c in mids:
            for b in good.get(c, []):
                if b == a or b in mids and False:
                    continue
                if check(a, b, args.label):
                    continue
                if want and not any(w in b for w in want):
                    continue
                span = separation(COORDS[a], COORDS[b])
                chains.append((span, a, c, b))

    chains.sort()
    print(f"{len(chains)} sound chains for {args.label}")
    for span, a, c, b in chains[:args.top]:
        print(f"  {span:5.0f} deg   {a}  ->  {c}  ->  {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
