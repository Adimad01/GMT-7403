"""
build_cardinal_train_data.py
================================================================================
Builds the two cardinal training sets used by the unified 6-experiment design,
from cardinal_direction_relations.csv (schema: source_entity, target_entity,
corpus, relation_label, ...), excluding the 32 balanced eval indices.

Outputs:
  ../dataset/cardinal_nokg_train.csv      raw rows (no KG)   → Exp 2/5 adapter
  ../dataset/cardinal_osm_kg_train.jsonl  {text,label} with  → Exp 3 adapter
                                          OSM evidence embedded in each prompt

The OSM evidence is pulled from osm_client.OSMEvidenceKG.  Run this LOCALLY
(where Nominatim is reachable) so the cache (results/osm_cache.json) is warmed
and real coordinates are embedded; on a cache miss the evidence degrades to
"No OSM data available".  Use --offline to never hit the network.

Usage:
  python build_cardinal_train_data.py                 # warms cache via Nominatim
  python build_cardinal_train_data.py --offline       # cache-only
"""

import os
import sys
import csv
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from strategies_cardinal import VALID_DIRECTIONS, VALID_LIST
from osm_client import OSMEvidenceKG

DATASET   = "../dataset/cardinal_direction_relations.csv"
EVAL_IDX  = "../dataset/eval_32_balanced_indices.json"
OUT_CSV   = "../dataset/cardinal_nokg_train.csv"
OUT_JSONL = "../dataset/cardinal_osm_kg_train.jsonl"


def build_osm_kg_prompt(src, tgt, corpus, label, evidence) -> str:
    ev = f"\n{evidence}\n" if evidence.strip() else "\n"
    return (
        "You are an expert in spatial geography and cardinal directions.\n\n"
        "Given the description and the OpenStreetMap evidence, determine the "
        f"cardinal direction of '{src}' relative to '{tgt}'.\n\n"
        f"Corpus: \"{corpus}\"\n"
        f"{ev}\n"
        f"Possible directions: {VALID_LIST}\n\n"
        f"Answer: [{label}]"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="Cache-only — never query Nominatim")
    args = ap.parse_args()

    if not os.path.exists(DATASET):
        print(f"[ERROR] Not found: {DATASET}")
        sys.exit(1)

    with open(EVAL_IDX) as f:
        eval_idx = set(json.load(f))

    with open(DATASET, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    train_rows = [r for i, r in enumerate(rows)
                  if i not in eval_idx and r.get("relation_label", "").strip().lower() in VALID_DIRECTIONS]
    print(f"[DATA] {len(rows)} total, {len(eval_idx)} eval excluded → {len(train_rows)} training rows")

    # --- no-KG CSV ---------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(OUT_CSV)), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(train_rows)
    print(f"[OK] no-KG CSV → {OUT_CSV} ({len(train_rows)} rows)")

    # --- OSM-KG jsonl ------------------------------------------------------
    # Skip rows whose entities fail OSM retrieval so the OSM-KG adapter never
    # trains on empty-evidence examples.
    kg = OSMEvidenceKG("results/osm_cache.json", allow_network=not args.offline)
    records, skipped = [], 0
    for i, r in enumerate(train_rows, 1):
        src = r["source_entity"].strip()
        tgt = r["target_entity"].strip()
        corpus = r["corpus"].strip()
        label = r["relation_label"].strip().lower()
        if kg.fetch(src) is None or kg.fetch(tgt) is None:
            skipped += 1
            continue
        evidence = kg.gather_evidence(src, tgt, sentence=corpus)
        records.append({"text": build_osm_kg_prompt(src, tgt, corpus, label, evidence),
                        "label": label})
        if i % 10 == 0:
            print(f"  ... {i}/{len(train_rows)} processed (kept {len(records)}, skipped {skipped})")

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] OSM-KG jsonl → {OUT_JSONL} ({len(records)} records, "
          f"{skipped} skipped for OSM-retrieval failure)")
    print("[DONE] Cardinal training data ready.")


if __name__ == "__main__":
    main()
