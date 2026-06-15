"""
build_osm_kg_from_topo_csv.py
================================================================================
Builds osm_kg_balanced_train.jsonl from topo_v2_train.csv.

Since we don't have live OSM/Nominatim API access on this server, we format
the corpus text as a structured KG-evidence prompt so the OSM-KG adapter
learns to reason from structured evidence blocks (which WILL be present at
inference time in exp_v2_03 via the eval engine's KG injection).

Output format (per JSONL record):
  {"text": "<full instruction + answer>", "label": "<predicate>"}

Usage:
  python build_osm_kg_from_topo_csv.py
  python build_osm_kg_from_topo_csv.py --input  ../dataset/topo_v2_train.csv
                                        --output ../dataset/osm_kg_balanced_train.jsonl
"""

import argparse
import csv
import json
import os
import random
import collections

VALID_PREDICATES = {
    "contains", "within", "touches", "crosses", "disjoint", "overlaps", "equals"
}

_VALID_LIST = "contains, within, touches, crosses, disjoint, overlaps, equals"

_REASONING = {
    "within": (
        "Entity A is spatially contained within Entity B. "
        "The spatial description confirms A resides inside B's extent."
    ),
    "contains": (
        "Entity A spatially contains Entity B. "
        "The spatial description confirms B lies within A's extent."
    ),
    "touches": (
        "Entities A and B share a boundary point but their interiors do not intersect. "
        "The spatial description indicates they are adjacent or border each other."
    ),
    "crosses": (
        "Entity A crosses Entity B — their interiors intersect but neither contains the other. "
        "The spatial description suggests partial overlap typical of a line-region crossing."
    ),
    "disjoint": (
        "Entities A and B are spatially disjoint — no shared points. "
        "The spatial description indicates they are separate with no overlap."
    ),
    "overlaps": (
        "Entities A and B partially overlap — their interiors share some area but neither "
        "fully contains the other. The spatial description confirms partial shared space."
    ),
    "equals": (
        "Entities A and B are spatially equal — they share the exact same geometry. "
        "The spatial description treats them as co-located or synonymous."
    ),
}


def build_prompt(src: str, tgt: str, corpus: str, label: str) -> str:
    reasoning = _REASONING.get(label, "The spatial relationship can be determined from the evidence.")
    return (
        "You are an expert in geospatial topological reasoning using the DE-9IM model.\n\n"
        "Given a vernacular spatial relation between two geographic entities A and B, "
        "use the spatial evidence provided to determine the correct DE-9IM topological predicate.\n\n"
        f"Entity A : {src}\n"
        f"Entity B : {tgt}\n\n"
        "Spatial Evidence:\n"
        f"  Description : \"{corpus}\"\n\n"
        f"Possible predicates: {_VALID_LIST}\n\n"
        "Rules:\n"
        "1. The relation is DIRECTED from A to B.\n"
        "2. Use the spatial evidence to reason geometrically.\n"
        "3. You must pick EXACTLY ONE predicate from the list above.\n\n"
        f"Reasoning: {reasoning}\n\n"
        f"Answer: [{label}]"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="../dataset/topo_v2_train.csv")
    parser.add_argument("--output", default="../dataset/osm_kg_balanced_train.jsonl")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input not found: {args.input}")
        print("        Run build_dataset_topological_v2.py first.")
        raise SystemExit(1)

    # topo_v2_train.csv uses remapped column names from build_dataset_topological_v2.py:
    #   spatial_relation   (← relation_label)
    #   place_name_subject (← source_entity)
    #   place_name_object  (← target_entity)
    #   relation_predicate (← corpus)
    def _get(row, *keys):
        for k in keys:
            v = row.get(k, "")
            if v:
                return str(v).strip()
        return ""

    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = _get(row, "spatial_relation", "relation_label").lower()
            if label in VALID_PREDICATES:
                rows.append(row)

    print(f"[INFO] Loaded {len(rows)} rows from {args.input}")

    by_pred = collections.defaultdict(list)
    for r in rows:
        by_pred[_get(r, "spatial_relation", "relation_label").lower()].append(r)

    counts = {p: len(by_pred.get(p, [])) for p in VALID_PREDICATES}
    print("[INFO] Per-predicate counts:", counts)

    records = []
    for r in rows:
        src    = _get(r, "place_name_subject", "source_entity")
        tgt    = _get(r, "place_name_object",  "target_entity")
        corpus = _get(r, "relation_predicate",  "corpus")
        label  = _get(r, "spatial_relation",    "relation_label").lower()
        text   = build_prompt(src, tgt, corpus, label)
        records.append({"text": text, "label": label})

    random.seed(args.seed)
    random.shuffle(records)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote {len(records)} records → {args.output}")

    final_counts: dict[str, int] = collections.Counter(r["label"] for r in records)
    print("[INFO] Output label distribution:", dict(final_counts))


if __name__ == "__main__":
    main()
