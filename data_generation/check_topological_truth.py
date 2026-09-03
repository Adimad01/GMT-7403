"""Refute impossible topological labels using cached bounding boxes.

Exact DE-9IM needs real polygons and a geometry library. Bounding boxes are
weaker, but they are already in the OSM cache and they are enough to prove that
certain labels CANNOT hold:

  contains / within   the inner box must sit inside the outer one
  equals              the two boxes must very nearly coincide
  touches / overlaps  the boxes must at least intersect -- two shapes whose
  crosses             boxes are apart cannot meet
  disjoint            boxes apart CONFIRMS it; boxes overlapping proves
                      nothing, since disjoint shapes can share a box

So the verdicts are: REFUTED (the label cannot hold), CONFIRMED (only for
disjoint), or INCONCLUSIVE. A refutation means the row is wrong or its
geocoding is wrong -- and either way the row cannot be used by a
knowledge-graph arm.

    python3 data_generation/check_topological_truth.py data/topological/eval.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "topological" / "osm" / "cache.json"


def expected_classes(name: str):
    if re.search(r"Lake|Loch|Sea$|Ocean|River|Bay|Gulf|Strait", name):
        return {"waterway", "natural", "place", "water"}
    if re.search(r"Mountain|Range|Desert|Forest|Island|Peninsula|Glacier", name):
        return {"natural", "place", "boundary", "landuse", "leisure"}
    if re.match(r"^(City|Town|State|Province) of ", name) or re.search(r"County|Borough", name):
        return {"boundary", "place"}
    return None


def box(entry):
    """(south, north, west, east) or None."""
    if not entry or "boundingbox" not in entry:
        return None
    s, n, w, e = (float(x) for x in entry["boundingbox"])
    return s, n, w, e


def inside(inner, outer, pad=0.02):
    """inner box lies within outer, allowing a small relative slack."""
    hs = (outer[1] - outer[0]) * pad
    hw = (outer[3] - outer[2]) * pad
    return (inner[0] >= outer[0] - hs and inner[1] <= outer[1] + hs
            and inner[2] >= outer[2] - hw and inner[3] <= outer[3] + hw)


def intersects(a, b):
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def near_equal(a, b, tol=0.1):
    span = max(a[1] - a[0], b[1] - b[0], a[3] - a[2], b[3] - b[2], 1e-6)
    return all(abs(x - y) <= tol * span for x, y in zip(a, b))


def verdict(label, ba, bb):
    """(verdict, reason) for 'A <label> B'."""
    if label == "contains":
        return ("REFUTED", "the object's box is not inside the subject's") \
            if not inside(bb, ba) else ("INCONCLUSIVE", "boxes are consistent")
    if label == "within":
        return ("REFUTED", "the subject's box is not inside the object's") \
            if not inside(ba, bb) else ("INCONCLUSIVE", "boxes are consistent")
    if label == "equals":
        return ("INCONCLUSIVE", "boxes coincide") if near_equal(ba, bb) \
            else ("REFUTED", "the two boxes are not the same extent")
    if label in ("touches", "overlaps", "crosses"):
        return ("INCONCLUSIVE", "boxes intersect") if intersects(ba, bb) \
            else ("REFUTED", f"boxes do not intersect, so they cannot {label}")
    if label == "disjoint":
        return ("CONFIRMED", "boxes do not intersect") if not intersects(ba, bb) \
            else ("INCONCLUSIVE", "boxes intersect, which does not settle it")
    return ("INCONCLUSIVE", "no rule for this label")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file")
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    cache = json.loads(CACHE.read_text())
    rows = list(Path(args.csv_file).open(newline="", encoding="utf-8")
                and csv.DictReader(Path(args.csv_file).open(newline="", encoding="utf-8")))
    fields = rows[0].keys()
    S = "source_entity" if "source_entity" in fields else "place_name_subject"
    T = "target_entity" if "target_entity" in fields else "place_name_object"
    L = "relation_label" if "relation_label" in fields else "spatial_relation"

    tally = {"REFUTED": [], "CONFIRMED": 0, "INCONCLUSIVE": 0,
             "NO GEOMETRY": 0, "BAD GEOCODE": 0}
    for i, r in enumerate(rows):
        a, b, lab = r[S].strip(), r[T].strip(), r[L].strip().lower()
        ea, eb = cache.get(a), cache.get(b)
        if not ea or not eb:
            tally["NO GEOMETRY"] += 1
            continue
        for nm, en in ((a, ea), (b, eb)):
            exp = expected_classes(nm)
            if exp and en.get("class") not in exp:
                tally["BAD GEOCODE"] += 1
                break
        else:
            ba, bb = box(ea), box(eb)
            if not ba or not bb:
                tally["NO GEOMETRY"] += 1
                continue
            v, why = verdict(lab, ba, bb)
            if v == "REFUTED":
                tally["REFUTED"].append((i + 2, a, b, lab, why))
            else:
                tally[v] += 1

    n = len(rows)
    print("=" * 78)
    print(f"  BOUNDING-BOX CHECK  {Path(args.csv_file).name}   {n} rows")
    print("=" * 78)
    print(f"  refuted (label cannot hold)   {len(tally['REFUTED']):>5}")
    print(f"  confirmed (disjoint proven)   {tally['CONFIRMED']:>5}")
    print(f"  inconclusive                  {tally['INCONCLUSIVE']:>5}")
    print(f"  not checkable, bad geocode    {tally['BAD GEOCODE']:>5}")
    print(f"  not checkable, no geometry    {tally['NO GEOMETRY']:>5}")
    if tally["REFUTED"]:
        print(f"\n  refutations (first {args.show}):")
        for ln, a, b, lab, why in tally["REFUTED"][:args.show]:
            print(f"    line {ln:>4}  {a} {lab} {b}")
            print(f"              {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
