"""Fetch long lines that are not rivers, to test `crosses` without the river cue.

Every `crosses` item in the frozen evaluation involves a river, so a model can find them from
the place type alone ("one of the two is a river") without reading the sentence. The catalogue
has almost no other lines: two mountain ranges already used in training, and three highways that
Nominatim resolved to a single short segment. This fetches whole route relations from OSM through
the Overpass API instead -- highways, European routes, long-distance trails -- so that a probe set
can ask `crosses` about a road or a trail.

    <venv with shapely>/bin/python data_generation/fetch_probe_lines.py

Writes data/topological/osm/probe_lines.json: name -> {class, type, osm_ids, geojson}. Geometries
are merged and simplified to the same tolerance as the catalogue (0.01 degree), which is what the
relation rules in compute_topo_relations.py are calibrated for.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from shapely.geometry import LineString, MultiLineString, mapping
from shapely.ops import linemerge

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "topological" / "osm" / "probe_lines.json"
URL = "https://overpass-api.de/api/interpreter"
UA = "spatial-eval/1.0 (academic research, Universite Laval)"
SIMPLIFY = 0.01

# name shown to the model -> (OSM class, OSM type, Overpass filter selecting the route relations)
LINES = {
    "Interstate 10": ("route", "road", '["route"="road"]["network"="US:I"]["ref"="10"]'),
    "Interstate 90": ("route", "road", '["route"="road"]["network"="US:I"]["ref"="90"]'),
    "Interstate 95": ("route", "road", '["route"="road"]["network"="US:I"]["ref"="95"]'),
    "Interstate 80": ("route", "road", '["route"="road"]["network"="US:I"]["ref"="80"]'),
    "Interstate 40": ("route", "road", '["route"="road"]["network"="US:I"]["ref"="40"]'),
    "European route E40": ("route", "road", '["route"="road"]["network"="e-road"]["ref"="E 40"]'),
    "European route E45": ("route", "road", '["route"="road"]["network"="e-road"]["ref"="E 45"]'),
    "European route E55": ("route", "road", '["route"="road"]["network"="e-road"]["ref"="E 55"]'),
    "Appalachian Trail": ("route", "hiking", '["route"="hiking"]["name"="Appalachian Trail"]'),
    "Pacific Crest Trail": ("route", "hiking", '["route"="hiking"]["name"="Pacific Crest National Scenic Trail"]'),
    "E1 European long distance path": ("route", "hiking", '["route"="hiking"]["network"="iwn"]["ref"="E1"]'),
}


def overpass(query: str, tries: int = 4) -> dict:
    data = urllib.parse.urlencode({"data": query}).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(URL, data=data, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=400) as r:
                return json.loads(r.read())
        except Exception as exc:                       # rate limit or timeout: wait and retry
            wait = 30 * (k + 1)
            print(f"    {type(exc).__name__}: {exc}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Overpass unreachable")


def fetch(filt: str):
    q = f"[out:json][timeout:360];relation{filt};way(r);out geom;"
    ways = [e for e in overpass(q)["elements"] if e["type"] == "way" and e.get("geometry")]
    segs = [LineString([(p["lon"], p["lat"]) for p in w["geometry"]]) for w in ways if len(w["geometry"]) > 1]
    rels = overpass(f"[out:json][timeout:120];relation{filt};out ids;")["elements"]
    if not segs:
        return None, []
    merged = linemerge(MultiLineString(segs))
    return merged.simplify(SIMPLIFY, preserve_topology=False), [e["id"] for e in rels]


def main() -> int:
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, (cls, typ, filt) in LINES.items():
        if name in out:
            print(f"  {name}: already fetched")
            continue
        print(f"  {name} ...")
        g, ids = fetch(filt)
        if g is None or g.is_empty:
            print("    nothing returned")
            continue
        km = g.length * 111.32 * 0.75
        print(f"    {len(ids)} relations, about {km:,.0f} km after simplification")
        out[name] = {"class": cls, "type": typ, "osm_ids": ids, "geojson": mapping(g)}
        OUT.write_text(json.dumps(out))
        time.sleep(20)
    print(f"  {len(out)} lines in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
