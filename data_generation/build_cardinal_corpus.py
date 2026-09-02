"""Build the cardinal corpus from cardinal_rows.py, verifying every claim.

Nothing here is taken on trust. Each row's label is recomputed from real
coordinates, and a row that does not survive every geometric test is reported
rather than written. For multi-hop rows both individual links are checked as
well, because a chain whose steps are wrong composes to nothing.

    python3 data_generation/build_cardinal_corpus.py --out data/cardinal/corpus.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cardinal_rows import ROWS                                    # noqa: E402
from check_cardinal_truth import (COORDS, bearing, components_agree,  # noqa: E402
                                  key, reciprocal, sector, separation)

MARGIN, MAX_SEP, MIN_SEP = 8.0, 110.0, 8.0
HEADER = ["source_entity", "source_geometry", "target_entity", "target_geometry",
          "corpus", "via_entity", "relation_type", "relation_label",
          "explanation", "ambiguity_level"]


def coord(name: str):
    k = key(name)
    if k not in COORDS:
        raise KeyError(name)
    return COORDS[k]


def check(a: str, b: str, label: str) -> list[str]:
    """Every reason the pair (a <label> b) is unusable."""
    try:
        pa, pb = coord(a), coord(b)
    except KeyError as e:
        return [f"no coordinates for {e.args[0]}"]
    problems = []
    deg = bearing(pb, pa)
    got, m = sector(deg)
    sep = separation(pa, pb)
    if got != label:
        problems.append(f"bearing {deg:.1f} is {got}, not {label}")
    elif m < MARGIN:
        problems.append(f"only {m:.1f} deg from a sector boundary")
    if sep > MAX_SEP:
        problems.append(f"{sep:.0f} deg apart, direction not well defined")
    if sep < MIN_SEP:
        problems.append(f"only {sep:.0f} deg apart, too close to be a clear item")
    if not reciprocal(pa, pb):
        problems.append("reverse bearing is not the opposite sector")
    if not components_agree(pa, pb, label):
        problems.append("cone label disagrees with the projection-based reading")
    return problems


def fmt(name: str) -> str:
    return f"City of {name}"


def latlon(p) -> str:
    ns = "N" if p[0] >= 0 else "S"
    ew = "E" if p[1] >= 0 else "W"
    return f"{abs(p[0]):.1f}{ns} {abs(p[1]):.1f}{ew}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()

    out, failures = [], []
    for lvl, label, src, tgt, via, desc in ROWS:
        probs = check(src, tgt, label)
        if lvl == 6:
            # both links must independently hold, or there is nothing to compose
            for a, b, which in ((src, via, "first"), (via, tgt, "second")):
                for pr in check(a, b, label):
                    probs.append(f"{which} hop ({a} -> {b}): {pr}")
        if probs:
            failures.append((lvl, label, src, tgt, probs))
            continue

        pa, pb = coord(src), coord(tgt)
        deg = bearing(pb, pa)
        if lvl == 6:
            expl = (f"Two {label.replace('_of', '')} steps compose through "
                    f"{via}. {src} {latlon(pa)}, {via} {latlon(coord(via))}, "
                    f"{tgt} {latlon(pb)}.")
        else:
            expl = (f"{src} {latlon(pa)}, {tgt} {latlon(pb)}; the bearing from "
                    f"{tgt} to {src} is {deg:.0f} degrees, "
                    f"{sector(deg)[1]:.0f} degrees inside the {label} sector.")
        out.append({
            "source_entity": fmt(src), "source_geometry": "Polygon",
            "target_entity": fmt(tgt), "target_geometry": "Polygon",
            "corpus": desc, "via_entity": fmt(via) if via else "",
            "relation_type": "cardinal_direction", "relation_label": label,
            "explanation": expl, "ambiguity_level": f"Level {lvl}",
        })

    if failures:
        print(f"  {len(failures)} row(s) failed verification and were NOT written:\n")
        for lvl, label, src, tgt, probs in failures:
            print(f"    L{lvl} {label:<13} {src} -> {tgt}")
            for pr in probs:
                print(f"         - {pr}")
        print()

    print(f"  {len(out)}/{len(ROWS)} rows verified")
    if args.out and not failures:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(out)
        print(f"  wrote {dest}")
    elif args.out:
        print("  nothing written — fix the failures first")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
