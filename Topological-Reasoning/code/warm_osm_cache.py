"""
warm_osm_cache.py
================================================================================
Pre-warm results/osm_cache.json by geocoding every entity in a dataset, so the
OFFLINE GPU server can serve OSM evidence from cache (kg-mode input / rag).

Run this LOCALLY where Nominatim is reachable, then commit the cache JSON and
push it to the server alongside the code.

Usage:
  python warm_osm_cache.py --dataset ../dataset/topo_v2_eval.csv
  python warm_osm_cache.py --dataset ../dataset/cardinal_direction_relations.csv
  python warm_osm_cache.py --dataset ../dataset/relative_direction_relations.csv \\
                           --cache results/osm_cache.json
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from osm_client import OSMClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="CSV with source_entity/target_entity columns")
    ap.add_argument("--cache", default="results/osm_cache.json")
    ap.add_argument("--force", action="store_true",
                    help="Re-query entities already present in the cache "
                         "(use after fixing a geocoding bug)")
    ap.add_argument("--cols", nargs="+", default=["source_entity", "target_entity",
                                                  "place_name_subject", "place_name_object"],
                    help="Candidate columns holding place names")
    args = ap.parse_args()

    if not os.path.exists(args.dataset):
        print(f"[ERROR] Not found: {args.dataset}")
        sys.exit(1)

    names = set()
    with open(args.dataset, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for c in args.cols:
                v = (row.get(c) or "").strip()
                if v and v.lower() != "nan":
                    names.add(v)

    print(f"[DATA] {len(names)} unique entities to geocode from {args.dataset}")
    client = OSMClient(args.cache)

    if args.force:
        stale = [n for n in names if n in client.cache]
        for n in stale:
            del client.cache[n]
        print(f"[FORCE] dropped {len(stale)} cached entries; re-querying")

    hit = miss = 0
    for i, name in enumerate(sorted(names), 1):
        data = client.fetch(name, allow_network=True)
        if data:
            hit += 1
        else:
            miss += 1
        if i % 10 == 0:
            print(f"  ... {i}/{len(names)}  (resolved={hit}, missing={miss})")

    print(f"[OK] Cache warmed → {args.cache}  (resolved={hit}, missing={miss})")


if __name__ == "__main__":
    main()
