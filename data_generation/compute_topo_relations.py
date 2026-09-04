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
import re
import sys
from pathlib import Path

from shapely.geometry import shape
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parents[1]
GEOM = REPO / "data" / "topological" / "osm" / "geometry.json"
OUT = REPO / "data" / "topological" / "osm" / "relations.json"

EPS = 0.01          # degrees, ~1 km: vertex noise and simplification slack
SLIVER = 0.02       # intersection below this fraction of the smaller area is noise
SLIVER_BAND = 3.0   # multiples of the expected noise band along a shared border


USABLE = {"Polygon", "MultiPolygon", "LineString", "MultiLineString"}


def load():
    """Geometries that can carry a topological relation, and what was dropped.

    A point has no interior or boundary to speak of, so it cannot stand in a
    DE-9IM relation worth testing. Several catalogue names resolve to one --
    OSM maps some mountain ranges as a single node, and 'Andes' matches a town
    in Colombia before it matches anything in the cordillera.
    """
    raw = json.loads(GEOM.read_text())
    geoms, dropped = {}, []
    for name, rec in raw.items():
        if not rec:
            dropped.append((name, "unresolved"))
            continue
        gt = rec["geojson"].get("type")
        if gt not in USABLE:
            dropped.append((name, f"geometry is a {gt}"))
            continue
        try:
            g = shape(rec["geojson"])
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty:
                dropped.append((name, "empty geometry"))
                continue
            geoms[name] = (g, rec)
        except Exception as exc:
            dropped.append((name, f"unreadable: {exc}"))
    return geoms, dropped


def area_km2(g) -> float:
    """Crude equal-area correction: scale longitude by cos(mean latitude)."""
    if g.geom_type in ("LineString", "MultiLineString"):
        return 0.0
    lat = g.centroid.y
    return g.area * (111.32 ** 2) * max(math.cos(math.radians(lat)), 0.01)


def convention_dependent(a, b, label: str | None = None) -> bool:
    """True when the answer hinges on how OSM models the pair, not on geography.

    A lake is mapped as a polygon with a hole cut for each of its islands, so
    the island is not inside the lake: they touch. In plain language the island
    is plainly in the lake. Grading such an item would test knowledge of the
    data model rather than spatial reasoning, and either answer can be
    defended, so the pair is not usable either way.

    The signature is that one shape sits inside the other's outer ring while
    the computed relation is not a containment. Comparing against strict
    shapely containment instead would fire on every simplified pair, because a
    simplified city pokes a few metres outside its simplified state.
    """
    from shapely.geometry import Polygon as _P
    def outer(g):
        if g.geom_type == "Polygon":
            return _P(g.exterior)
        if g.geom_type == "MultiPolygon":
            return unary_union([_P(p.exterior) for p in g.geoms])
        return None
    for x, y in ((a, b), (b, a)):
        ox = outer(x)
        if ox is None or y.geom_type in ("LineString", "MultiLineString"):
            continue
        if ox.contains(y.representative_point()) and label in (
                "touches", "disjoint", "overlaps"):
            return True
    return False


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
    # Two units that share a border overlap by a sliver once their outlines are
    # simplified: the error band runs the length of the shared boundary, so its
    # area grows with that length, not with the size of either shape. Comparing
    # the intersection against a fixed fraction of the smaller area therefore
    # calls adjacent cities 'overlaps' and adjacent countries 'touches'. The
    # comparison has to be against the expected noise instead.
    shared = a.boundary.intersection(b.boundary)
    shared_len = shared.length if hasattr(shared, "length") else 0.0
    lat = a.centroid.y
    km_per_deg = 111.32 * max(math.cos(math.radians(lat)), 0.01)
    noise_km2 = shared_len * km_per_deg * EPS * 111.32 * SLIVER_BAND
    info["shared_len_deg"] = shared_len
    info["noise_km2"] = noise_km2
    if ia > noise_km2 and ia / min(aa, ab) > SLIVER:
        info["overlap_fraction"] = ia / min(aa, ab)
        return ("overlaps", info)
    if a.distance(b) <= EPS or ia > 0:
        info["perimeter_deg"] = min(a.boundary.length, b.boundary.length)
        return ("touches", info)
    info["gap_deg"] = a.distance(b)
    return ("disjoint", info)


def name_overlap(a: str, b: str) -> float:
    """Share of distinctive words the two names have in common.

    'equals' needs a difficulty measure that geometry cannot give. Two objects
    either coincide or they do not, so every genuine pair scores identically on
    any geometric scale and lands in one level. What actually varies is whether
    a reader can see that the two names denote the same thing: 'City of
    Philadelphia' and 'Philadelphia County' announce it, 'Borough of Brooklyn'
    and 'Kings County' do not. That is the difficulty, and it is measurable.
    """
    stop = {"city", "county", "and", "of", "the", "borough", "parish",
            "district", "consolidated", "state"}
    wa = {w for w in re.findall(r"[a-z]+", a.lower()) if w not in stop}
    wb = {w for w in re.findall(r"[a-z]+", b.lower()) if w not in stop}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


# The quantity that makes each relation easy when large and hard when small.
METRIC = {"contains": "ratio", "within": "ratio", "disjoint": "gap_deg",
          "touches": "touch_share", "overlaps": "overlap_fraction",
          "crosses": "inside_fraction", "equals": "name_share"}


def metric_of(label: str, info: dict, subject: str = "", obj: str = "") -> float:
    if label == "equals":
        return name_overlap(subject, obj)
    if label == "touches":
        p = info.get("perimeter_deg") or 1.0
        return info.get("shared_len_deg", 0.0) / p
    return info.get(METRIC.get(label, ""), 0.0)


def assign_levels(records: list[dict]) -> None:
    """Give each record a level 1-5 by its rank within its own label.

    Fixed thresholds do not transfer between labels: requiring 70% shared area
    for an easy 'overlaps' left two pairs in the whole catalogue. Cutting on
    metric VALUES does not work either, because the metrics tie heavily -- many
    adjacent pairs share an identical boundary fraction, and name overlap for
    'equals' is almost always exactly 0 or 1. Value cuts then collapse whole
    bands, which is how 'touches' ended up with 408 rows at Level 4 and none at
    Level 5.

    Ranking by position splits ties evenly and keeps the levels comparable in
    meaning: the most extreme fifth of the configurations down to the least.
    """
    records.sort(key=lambda r: -r["metric"])
    n = len(records)
    for i, r in enumerate(records):
        r["level"] = min(5, int(i * 5 / n) + 1) if n else 3


def difficulty(label: str, info: dict, subject: str = "", obj: str = "") -> int:
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
        # shared name -> obvious; nothing shared -> needs real knowledge
        return band(name_overlap(subject, obj), [0.99, 0.6, 0.3, 0.01])
    return 3


def main() -> int:
    geoms, dropped = load()
    names = sorted(geoms)
    print(f"  {len(names)} usable geometries, {len(dropped)} dropped", flush=True)
    for n, why in dropped[:8]:
        print(f"    dropped {n}: {why}", flush=True)

    # Two passes: measure everything, then rank within each label so the five
    # levels are equally populated whatever the metric's natural scale.
    raw = []
    for i, a in enumerate(names):
        ga, _ = geoms[a]
        for b in names:
            if a == b:
                continue
            gb, _ = geoms[b]
            if ga.distance(gb) > 60:            # far apart: certainly disjoint, skip
                continue
            lab, info = relate(ga, gb)
            if not lab or convention_dependent(ga, gb, lab):
                continue
            raw.append({"subject": a, "object": b, "label": lab,
                        "metric": metric_of(lab, info, a, b)})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(names)} subjects, {len(raw)} relations", flush=True)

    from collections import defaultdict as _dd
    by_label = _dd(list)
    for r in raw:
        by_label[r["label"]].append(r)
    for recs in by_label.values():
        assign_levels(recs)
    out = [{"subject": r["subject"], "object": r["object"],
            "label": r["label"], "level": r["level"]} for r in raw]

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
