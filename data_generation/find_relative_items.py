"""Surface (observer, subject, target) triples that are usable for a label.

A relative-direction item needs three places, not two, and the observer's
facing decides both the answer and the difficulty. Searching for those by hand
is hopeless -- the frame rotation is not something anyone estimates reliably --
so this enumerates them and keeps only the triples that sit well inside their
region.

    python3 data_generation/find_relative_items.py --label left_of --level 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_cardinal_truth import COORDS, bearing, separation      # noqa: E402
from check_relative_truth import (classify, check, norm180,       # noqa: E402
                                  rotation_level)

FAMOUS_RAW = """tokyo seoul beijing shanghai hong kong manila bangkok hanoi singapore jakarta
delhi mumbai kolkata chennai karachi dhaka colombo kathmandu tashkent almaty london paris
berlin madrid lisbon rome milan naples venice vienna prague warsaw budapest athens istanbul
moscow kyiv stockholm oslo copenhagen helsinki dublin edinburgh amsterdam brussels zurich
munich lyon marseille valencia riga vilnius minsk belgrade bucharest sofia zagreb new york
boston chicago washington miami atlanta dallas houston denver seattle san francisco
los angeles san diego las vegas phoenix toronto montreal vancouver calgary winnipeg halifax
mexico city havana panama city bogota lima quito caracas santiago buenos aires montevideo
asuncion la paz sao paulo rio de janeiro cairo lagos accra nairobi casablanca algiers tunis
tripoli khartoum addis ababa johannesburg cape town dakar harare maputo luanda windhoek
kampala tehran baghdad riyadh dubai abu dhabi beirut damascus jerusalem amman baku yerevan
tbilisi ankara sydney melbourne brisbane perth auckland wellington"""


def famous() -> list[str]:
    out = []
    for line in FAMOUS_RAW.split("\n"):
        toks = line.split()
        i = 0
        while i < len(toks):
            for span in (3, 2, 1):
                c = " ".join(toks[i:i + span])
                if c in COORDS:
                    out.append(c)
                    i += span
                    break
            else:
                i += 1
    return sorted(set(out))


def search(label: str, level: int, margin: float, limit: int):
    names = famous()
    hits = []
    for v in names:
        pv = COORDS[v]
        near = [(n, bearing(pv, COORDS[n]), separation(pv, COORDS[n]))
                for n in names if n != v]
        near = [(n, br, sp) for n, br, sp in near if 3.0 <= sp <= 70.0]
        for b, tb, _ in near:
            if abs(norm180(tb)) > 180:
                continue
            if rotation_level(v, b) != level:
                continue
            for a, ab, _ in near:
                if a == b:
                    continue
                got, m = classify(v, a, b)
                if got != label or m < margin:
                    continue
                if check(v, a, b, label, margin=margin):
                    continue
                hits.append((m, v, a, b))
                if len(hits) >= limit * 40:
                    break
    hits.sort(key=lambda t: -t[0])
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--margin", type=float, default=12.0)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    hits = search(args.label, args.level, args.margin, args.top)
    print(f"{len(hits)} triples for {args.label} at Level {args.level}")
    print(f"{'margin':>7}   observer -> looking at | subject")
    for m, v, a, b in hits[:args.top]:
        print(f"{m:7.1f}   from {v}, gaze on {b}: {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
