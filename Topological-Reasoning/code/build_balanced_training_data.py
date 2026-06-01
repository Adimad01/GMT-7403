"""
build_balanced_training_data.py
================================================================================
Balances fine-tuning datasets by downsampling each DE-9IM predicate class
to the same number of examples (minimum count across all classes).

Supports both CSV (triplet_update_v3_70.csv) and JSONL (osm_kg_train.jsonl).

Predicate counts before balancing:
  CSV  (triplet_update_v3_70.csv):  touches 223, within 186, disjoint 183,
                                    overlaps 83, crosses 41, contains 39  → min=39
  JSONL (osm_kg_train.jsonl):       touches 223, within 186, disjoint 183,
                                    overlaps 83, crosses 40, contains 39  → min=39

Both balanced at 39/predicate → 234 examples each.

Usage:
    python build_balanced_training_data.py          # balances both files
    python build_balanced_training_data.py --n 40   # custom per-class count
    python build_balanced_training_data.py --seed 42
"""

import csv
import json
import random
import argparse
import os
import collections

PREDICATES = ["contains", "within", "touches", "crosses", "disjoint", "overlaps"]

JOBS = [
    {
        "input":  "../dataset/triplet_update_v3_70.csv",
        "output": "../dataset/triplet_balanced_train.csv",
        "format": "csv",
        "pred_field": "spatial_relation",
    },
    {
        "input":  "../dataset/osm_kg_train.jsonl",
        "output": "../dataset/osm_kg_balanced_train.jsonl",
        "format": "jsonl",
        "pred_field": "label",
    },
]


def _read(path, fmt):
    if fmt == "csv":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader), reader.fieldnames
    else:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        return rows, None


def _write(rows, path, fmt, fieldnames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fmt == "csv":
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def balance_dataset(job, n_per_class, seed):
    rows, fieldnames = _read(job["input"], job["format"])
    pred_field = job["pred_field"]

    by_pred = collections.defaultdict(list)
    for r in rows:
        by_pred[r[pred_field].strip().lower()].append(r)

    counts = {p: len(by_pred.get(p, [])) for p in PREDICATES}
    actual_n = n_per_class or min(counts.values())

    print(f"\n{'='*60}")
    print(f"  {os.path.basename(job['input'])}  →  {os.path.basename(job['output'])}")
    print(f"{'='*60}")
    print(f"  Per-predicate counts (original → balanced at {actual_n}):")
    for p in PREDICATES:
        n = counts.get(p, 0)
        marker = "← min" if n == min(counts.values()) and not n_per_class else ""
        print(f"    {p:<12}: {n:>4} → {min(n, actual_n):>4}  {marker}")

    if any(counts.get(p, 0) < actual_n for p in PREDICATES):
        short = [p for p in PREDICATES if counts.get(p, 0) < actual_n]
        raise ValueError(f"Not enough examples for: {short}. Lower --n.")

    random.seed(seed)
    balanced = []
    for p in PREDICATES:
        balanced.extend(random.sample(by_pred[p], actual_n))
    random.shuffle(balanced)

    _write(balanced, job["output"], job["format"], fieldnames)

    # Verify
    out_rows, _ = _read(job["output"], job["format"])
    print(f"  Verification — {actual_n} × {len(PREDICATES)} = {len(out_rows)} total  ✅")
    print(f"  Saved → {job['output']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=None,
                        help="Examples per predicate (default: min across classes)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for job in JOBS:
        balance_dataset(job, args.n, args.seed)

    print("\n✅  All datasets balanced.")


if __name__ == "__main__":
    main()
