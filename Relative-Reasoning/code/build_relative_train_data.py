"""
build_relative_train_data.py
================================================================================
Builds the two relative-direction training sets for the unified 6-experiment
design, from relative_direction_relations.csv (schema: source_entity,
target_entity, corpus, relation_label, ...), excluding the 20 balanced eval
indices.

Outputs:
  ../dataset/relative_balanced_train.csv   raw rows (no KG)   → Exp 2/5 adapter
  ../dataset/relative_osm_kg_train.jsonl   {text,label} with  → Exp 3 adapter
                                           OSM evidence embedded

(The no-KG CSV name matches what train_runner_relative.py already expects.)

OSM evidence is informational only — relative direction depends on the observer
frame in the corpus, which coordinates cannot supply.  Run LOCALLY to warm the
cache; use --offline for cache-only.

Usage:
  python build_relative_train_data.py
  python build_relative_train_data.py --offline
"""
import os
import sys
import csv
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from strategies_relative import VALID_DIRECTIONS, VALID_LIST
from osm_client import OSMEvidenceKG

DATASET   = "../dataset/relative_direction_relations.csv"
EVAL_IDX  = "../dataset/eval_20_balanced_indices.json"
OUT_CSV   = "../dataset/relative_balanced_train.csv"
OUT_JSONL = "../dataset/relative_osm_kg_train.jsonl"


def build_osm_kg_prompt(src, tgt, corpus, label, evidence) -> str:
    ev = f"\n{evidence}\n" if evidence.strip() else "\n"
    return (
        "You are an expert in spatial and relative directions.\n\n"
        "Given the description and the OpenStreetMap evidence, determine the "
        f"relative direction of '{src}' relative to '{tgt}' from an observer's "
        "perspective. The OSM facts are absolute geometry; the observer frame "
        "comes from the corpus.\n\n"
        f"Corpus: \"{corpus}\"\n"
        f"{ev}\n"
        f"Possible directions: {VALID_LIST}\n\n"
        f"Answer: [{label}]"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="Cache-only — never query Nominatim")
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
    kg = OSMEvidenceKG("results/osm_cache.json", allow_network=not args.offline)
    records = []
    for i, r in enumerate(train_rows, 1):
        src = r["source_entity"].strip()
        tgt = r["target_entity"].strip()
        corpus = r["corpus"].strip()
        label = r["relation_label"].strip().lower()
        evidence = kg.gather_evidence(src, tgt, sentence=corpus)
        records.append({"text": build_osm_kg_prompt(src, tgt, corpus, label, evidence),
                        "label": label})
        if i % 10 == 0:
            print(f"  ... {i}/{len(train_rows)} geocoded")

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] OSM-KG jsonl → {OUT_JSONL} ({len(records)} records)")
    print("[DONE] Relative training data ready.")


if __name__ == "__main__":
    main()
