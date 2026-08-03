"""
build_osm_graph.py — offline builder for the Exp 7 GraphRAG artifact.
================================================================================
Reads the domain's existing results/osm_cache.json and derives the spatial graph
(`within`/`contains` from the administrative hierarchy, `near` from haversine),
writing results/osm_graph.json.

Deterministic and network-free: every fact already lives in the cache, so this
can be re-run anywhere and produces byte-identical output for a given cache.
Still run it LOCALLY and commit the artifact — the GPU server has no internet
and inference only ever reads the static JSON.

    python build_osm_graph.py                     # defaults: 50 km, 5 neighbours
    python build_osm_graph.py --proximity-km 25   # tighter proximity edges
"""
import argparse
import json
import os
import sys

from graph_kg import build_graph, DEFAULT_PROXIMITY_KM, DEFAULT_MAX_NEIGHBORS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/osm_cache.json")
    ap.add_argument("--out",   default="results/osm_graph.json")
    ap.add_argument("--proximity-km", type=float, default=DEFAULT_PROXIMITY_KM)
    ap.add_argument("--max-neighbors", type=int, default=DEFAULT_MAX_NEIGHBORS)
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        print(f"[ERROR] cache not found: {args.cache}")
        print("        Run warm_osm_cache.py first (needs internet).")
        sys.exit(1)

    with open(args.cache, "r", encoding="utf-8") as f:
        cache = json.load(f)

    geocoded = sum(1 for v in cache.values() if v)
    print(f"[BUILD] {args.cache}: {len(cache)} entries, {geocoded} geocoded")

    graph = build_graph(cache,
                        proximity_km=args.proximity_km,
                        max_neighbors=args.max_neighbors)

    n_within = sum(1 for e in graph["edges"] if e["rel"] == "within")
    n_near = sum(1 for e in graph["edges"] if e["rel"] == "near")
    n_admin = sum(1 for a in graph["nodes"].values() if a["kind"] == "admin")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=1, sort_keys=True)

    size_kb = os.path.getsize(args.out) / 1024
    print(f"[OK] wrote {args.out}  ({size_kb:.0f} KB)")
    print(f"     nodes {len(graph['nodes'])}  (place {len(graph['nodes']) - n_admin} · admin {n_admin})")
    print(f"     edges {len(graph['edges'])}  (within {n_within} · contains {n_within} · near {n_near})")


if __name__ == "__main__":
    main()
