"""Build the knowledge base for the evaluation entities.

Facts about places, never the relation being asked. The point of the arm this
feeds is to separate two failures that an accuracy cannot tell apart: not
knowing where a place is, and not being able to reason from where it is. So
the store carries coordinates, extents and administrative context, and the
model is left to compute the relation itself.

Sources are the two that were verified: the OSM geometry fetched with the
fixed resolver, and the city coordinate table the cardinal ground truth was
computed from. The osm/cache.json files are NOT used -- they predate the
resolver fixes and hold answers like Loch Ness as a cycleway in Florida.

No geometry library is needed. Centroid, bounding box and extent come from the
coordinates themselves, so this runs anywhere the rest of the analysis does.

    python3 data_generation/build_kg.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GEOMETRY = REPO / "data" / "topological" / "osm" / "geometry.json"
COORD_SRC = REPO / "data_generation" / "check_cardinal_truth.py"
RELATIONS = ("topological", "cardinal", "relative")

# Which source each family's ground truth was computed from. The store has to
# describe the same entity the label was derived from, or it contradicts the
# answer it is meant to support: OSM resolves Athens to Athens, Georgia, where
# "equals Clarke County" is exactly right, while the city table has Athens,
# Greece, which is what the cardinal bearings were computed from.
TRUTH_SOURCE = {"topological": "osm_geometry",
                "cardinal": "city_table",
                "relative": "city_table"}
QUALIFIERS = ("City of ", "State of ", "Province of ", "Borough of ")


def short(name: str) -> str:
    for q in QUALIFIERS:
        if name.startswith(q):
            return name[len(q):]
    return name


def load_coords() -> dict[str, tuple[float, float]]:
    spec = importlib.util.spec_from_file_location("cct", COORD_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.COORDS


def walk(geom) -> list[tuple[float, float]]:
    """Every (lon, lat) pair in a GeoJSON geometry, at any nesting depth."""
    if not geom:
        return []
    coords = geom.get("coordinates") if isinstance(geom, dict) else geom
    out: list[tuple[float, float]] = []
    stack = [coords]
    while stack:
        item = stack.pop()
        if (isinstance(item, (list, tuple)) and len(item) == 2
                and all(isinstance(x, (int, float)) for x in item)):
            out.append((float(item[0]), float(item[1])))
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return out


def rings(geom) -> list[list[tuple[float, float]]]:
    """Outer rings of a GeoJSON geometry, one list of (lon, lat) each."""
    if not geom:
        return []
    kind, coords = geom.get("type"), geom.get("coordinates")
    if kind == "Polygon":
        return [[(float(x), float(y)) for x, y in coords[0]]] if coords else []
    if kind == "MultiPolygon":
        return [[(float(x), float(y)) for x, y in poly[0]]
                for poly in coords if poly]
    return []


def ring_area_centroid(ring) -> tuple[float, float, float]:
    """Signed area and centroid of a ring, by the shoelace formula.

    The mean of the vertices is not the centroid: coastlines carry far more
    points than smooth borders, which drags the average toward whichever edge
    was mapped in most detail, and a country with overseas parts is pulled out
    to sea entirely. France came out at 1.9N 46.0W -- the mid-Atlantic --
    because its outermost territories weigh as much as the mainland.
    """
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-12:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return 0.0, sum(xs) / len(xs), sum(ys) / len(ys)
    a *= 0.5
    return abs(a), cx / (6 * a), cy / (6 * a)


def km_per_degree(lat: float) -> tuple[float, float]:
    """Kilometres per degree of latitude and of longitude at this latitude."""
    return 111.32, 111.32 * max(math.cos(math.radians(lat)), 1e-6)


def describe(name: str, entry: dict | None,
             coords: dict[str, tuple[float, float]],
             keep_context: bool = True,
             prefer: str = "osm_geometry") -> dict | None:
    """The facts recorded for one place, from the source the truth used."""
    key = short(name).lower()
    if prefer == "city_table" and key in coords:
        lat, lon = coords[key]
        return {"name": name, "kind": "place/city",
                "lat": lat, "lon": lon, "source": "city_table"}
    if entry and entry.get("geojson"):
        pts = walk(entry["geojson"])
        if not pts:
            return None
        parts = rings(entry["geojson"])
        if parts:
            weighted = sorted(
                ((ring_area_centroid(r), r) for r in parts if len(r) >= 3),
                key=lambda w: w[0][0], reverse=True)
            total = sum(a for (a, _, _), _ in weighted) or 0.0
            if weighted and total > 0:
                # The largest part, not the average of all of them. France is
                # 57 percent mainland and 43 percent territories scattered
                # from Guiana to Kerguelen, so its area-weighted centre falls
                # in the ocean off Mauritania and locates nothing. The United
                # States is 70 percent contiguous and 27 percent Alaska, which
                # pulls the mean into Montana. The dominant part is what the
                # name denotes, and share says how much of the whole it is.
                (a_main, lon_c, lat_c), main = weighted[0]
                share = a_main / total
            else:
                lon_c = sum(p[0] for p in pts) / len(pts)
                lat_c = sum(p[1] for p in pts) / len(pts)
                main, share = pts, 1.0
        else:
            lon_c = sum(p[0] for p in pts) / len(pts)
            lat_c = sum(p[1] for p in pts) / len(pts)
            main, share = pts, 1.0
        lons = [p[0] for p in main]
        lats = [p[1] for p in main]
        dlat, dlon = km_per_degree(lat_c)
        fact = {
            "name": name,
            "kind": f"{entry.get('class', '')}/{entry.get('type', '')}".strip("/"),
            "lat": round(lat_c, 4),
            "lon": round(lon_c, 4),
            "bbox": [round(min(lats), 4), round(min(lons), 4),
                     round(max(lats), 4), round(max(lons), 4)],
            "extent_km": [round((max(lats) - min(lats)) * dlat, 1),
                          round((max(lons) - min(lons)) * dlon, 1)],
            "n_parts": len(parts),
            "main_part_share": round(share, 3),
            "source": "osm_geometry",
        }
        if keep_context:
            # Administrative context, as OSM writes it: the units containing
            # the place, largest last. Withheld for the topological family,
            # where "Allegheny County, Pennsylvania" states the very relation
            # under test -- 63 of 236 rows had their answer sitting here.
            hierarchy = [p.strip()
                         for p in entry.get("display_name", "").split(",")]
            fact["context"] = hierarchy[1:][-3:]
        return fact
    if key in coords:
        lat, lon = coords[key]
        return {"name": name, "kind": "place/city",
                "lat": lat, "lon": lon, "source": "city_table"}
    return None


def main() -> int:
    geometry = {k: v for k, v in json.loads(
        GEOMETRY.read_text(encoding="utf-8")).items() if v}
    coords = load_coords()
    print(f"  sources : {len(geometry)} géométries, {len(coords)} villes\n")

    for rel in RELATIONS:
        rows = list(csv.DictReader(
            (REPO / "data" / rel / "eval.csv").open(encoding="utf-8")))
        cols = ["source_entity", "target_entity"]
        if rel == "relative":
            cols.append("observer_entity")
        names = sorted({r[c] for r in rows for c in cols if r.get(c)})

        nodes, missing = {}, []
        for n in names:
            entry = geometry.get(n) or geometry.get(short(n))
            # Containment is the topological label set, so its administrative
            # hierarchy cannot be shown to a model being asked for it.
            fact = describe(n, entry, coords,
                            keep_context=(rel != "topological"),
                            prefer=TRUTH_SOURCE[rel])
            if fact:
                nodes[n] = fact
            else:
                missing.append(n)

        out = REPO / "data" / rel / "kg_eval.json"
        out.write_text(json.dumps({
            "relation": rel,
            "n_nodes": len(nodes),
            "note": "facts only; the relation under test is never stated here",
            "nodes": nodes,
        }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

        by_source = {}
        for f in nodes.values():
            by_source[f["source"]] = by_source.get(f["source"], 0) + 1
        print(f"  {rel:<13}{len(nodes)}/{len(names)} entités "
              f"({len(nodes)/len(names)*100:.0f} %)   {by_source}")
        if missing:
            print(f"    manquantes ({len(missing)}) : {', '.join(missing[:5])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
