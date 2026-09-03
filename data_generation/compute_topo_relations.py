"""Compute the true topological relation for every pair in the catalogue.

Run with the geometry virtualenv, which has shapely:

    <venv>/bin/python data_generation/compute_topo_relations.py

Real OSM outlines are simplified before download and are not perfectly
coincident where two units share a border, so exact DE-9IM predicates give
brittle answers -- two countries that plainly touch may come out disjoint by a
few hundred metres of vertex noise. Every predicate here therefore carries a
tolerance, and the tests are ordered from most specific to least so that the
first one that fits wins.

Difficulty is derived from the geometry too, on the same principle used for
the other two relations: an item is easy when the configuration is extreme and
hard when it is marginal. A city inside a country is obvious; two units of
almost the same size, one just inside the other, is not.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from shapely.geometry import shape
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parents[1]
GEOM = REPO / "data" / "topological" / "osm" / "geometry.json"
OUT = REPO / "data" / "topological" / "osm" / "relations.json"

EPS = 0.01          # degrees, ~1 km: vertex noise and simplification slack
SLIVER = 0.02       # intersection below this fraction of the smaller area is noise


def load():
    raw = json.loads(GEOM.read_text())
    geoms = {}
    for name, rec in raw.items():
        if not rec:
            continue
        try:
            g = shape(rec["geojson"])
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty:
                continue
            geoms[name] = (g, rec)
        except Exception:
            continue
    return geoms


def area_km2(g) -> float:
    """Crude equal-area correction: scale longitude by cos(mean latitude)."""
    if g.geom_type in ("LineString", "MultiLineString"):
        return 0.0
    lat = g.centroid.y
    return g.area * (111.32 ** 2) * max(math.cos(math.radians(lat)), 0.01)


def relate(a, b) -> tuple[str | None, dict]:
    """Relation of a with respect to b, plus the measurements behind it."""
    la = a.geom_type in ("LineString", "MultiLineString")
    lb = b.geom_type in ("LineString", "MultiLineString")
    info: dict = {}

    if la and lb:
        return (None, info)                     # line/line: not used here

    if la or lb:                                # one line, one polygon
        line, poly, reversed_ = (a, b, False) if la else (b, a, True)
        inside = line.intersection(poly)
        frac = (inside.length / line.length) if line.length else 0.0
        info["inside_fraction"] = frac
        if frac >= 0.995:
            return ("within" if not reversed_ else "contains", info)
        if frac > 0.02:
            return ("crosses", info)
        if line.distance(poly) <= EPS:
            return ("touches", info)
        info["gap_deg"] = line.distance(poly)
        return ("disjoint", info)

    # polygon / polygon
    aa, ab = area_km2(a), area_km2(b)
    info["area_a_km2"], info["area_b_km2"] = aa, ab
    if min(aa, ab) <= 0:
        return (None, info)
    inter = a.intersection(b)
    ia = area_km2(inter) if not inter.is_empty else 0.0
    info["intersection_km2"] = ia
    union = area_km2(unary_union([a, b]))
    info["jaccard"] = ia / union if union else 0.0

    if info["jaccard"] >= 0.97:
        return ("equals", info)
    if ia / ab >= 0.97 and aa > ab:
        info["ratio"] = aa / ab
        return ("contains", info)
    if ia / aa >= 0.97 and ab > aa:
        info["ratio"] = ab / aa
        return ("within", info)
    if ia / min(aa, ab) > SLIVER:
        info["overlap_fraction"] = ia / min(aa, ab)
        return ("overlaps", info)
    if a.distance(b) <= EPS:
        shared = a.boundary.intersection(b.boundary)
        info["shared_len_deg"] = shared.length if hasattr(shared, "length") else 0.0
        info["perimeter_deg"] = min(a.boundary.length, b.boundary.length)
        return ("touches", info)
    info["gap_deg"] = a.distance(b)
    return ("disjoint", info)


def difficulty(label: str, info: dict) -> int:
    """Ambiguity level 1-5. Extreme configurations are easy, marginal ones hard."""
    def band(x, cuts):                       # cuts high -> low, level 1 -> 5
        for i, c in enumerate(cuts, start=1):
            if x >= c:
                return i
        return 5

    if label in ("contains", "within"):
        return band(info.get("ratio", 1.0), [10000, 1000, 100, 10])
    if label == "disjoint":
        return band(info.get("gap_deg", 0.0), [40, 15, 5, 1])
    if label == "touches":
        p = info.get("perimeter_deg") or 1.0
        return band(info.get("shared_len_deg", 0.0) / p, [0.30, 0.15, 0.06, 0.02])
    if label == "overlaps":
        return band(info.get("overlap_fraction", 0.0), [0.7, 0.4, 0.2, 0.08])
    if label == "crosses":
        return band(info.get("inside_fraction", 0.0), [0.6, 0.3, 0.12, 0.04])
    if label == "equals":
        return band(info.get("jaccard", 0.0), [0.999, 0.995, 0.99, 0.98])
    return 3


def main() -> int:
    geoms = load()
    names = sorted(geoms)
    print(f"  {len(names)} geometries loaded", flush=True)

    out = []
    for i, a in enumerate(names):
        ga, _ = geoms[a]
        for b in names:
            if a == b:
                continue
            gb, _ = geoms[b]
            if ga.distance(gb) > 60:            # far apart: certainly disjoint, skip
                continue
            lab, info = relate(ga, gb)
            if not lab:
                continue
            out.append({"subject": a, "object": b, "label": lab,
                        "level": difficulty(lab, info),
                        "info": {k: round(v, 6) for k, v in info.items()
                                 if isinstance(v, (int, float))}})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(names)} subjects, {len(out)} relations", flush=True)

    OUT.write_text(json.dumps(out))
    from collections import Counter
    c = Counter((r["label"], r["level"]) for r in out)
    labs = ["contains", "within", "touches", "crosses", "overlaps", "disjoint", "equals"]
    print("\n  available pairs per (label, level)")
    print("  " + "label".ljust(11) + "".join(f"L{i}".rjust(8) for i in range(1, 6)))
    for l in labs:
        print("  " + l.ljust(11) + "".join(str(c.get((l, i), 0)).rjust(8) for i in range(1, 6)))
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
