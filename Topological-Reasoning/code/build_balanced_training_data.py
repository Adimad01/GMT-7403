"""
build_balanced_training_data.py
================================================================================
Balances fine-tuning datasets by downsampling each DE-9IM predicate class
to the same number of examples (minimum count across all classes).

Supports both CSV (topological_relations.csv) and JSONL (osm_kg_train.jsonl).

Predicate counts in topological_relations.csv (7 predicates):
  contains 187, within 188, touches 188, crosses 187,
  disjoint 188, overlaps 185, equals 186

JSONL (osm_kg_train.jsonl) — 6 predicates only (no equals):
  touches 223, within 186, disjoint 183, overlaps 83, crosses 40, contains 39

Usage:
    python build_balanced_training_data.py          # balances both files
    python build_balanced_training_data.py --n 40   # custom per-class count
    python build_balanced_training_data.py --seed 42
    python build_balanced_training_data.py --exclude-eval  # skip eval indices
"""

import csv
import json
import random
import argparse
import os
import collections

PREDICATES_CSV  = ["contains", "within", "touches", "crosses", "disjoint", "overlaps", "equals"]
PREDICATES_JSONL = ["contains", "within", "touches", "crosses", "disjoint", "overlaps"]

EVAL_INDICES_FILE = "../dataset/eval_385_balanced_indices.json"

JOBS = [
    {
        "input":      "../dataset/topological_relations.csv",
        "output":     "../dataset/topological_balanced_train.csv",
        "format":     "csv",
        "pred_field": "relation_label",
        "predicates": PREDICATES_CSV,
    },
    {
        "input":      "../dataset/osm_kg_train.jsonl",
        "output":     "../dataset/osm_kg_balanced_train.jsonl",
        "format":     "jsonl",
        "pred_field": "label",
        "predicates": PREDICATES_JSONL,
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


def balance_dataset(job, n_per_class, seed, exclude_eval_indices=None):
    rows, fieldnames = _read(job["input"], job["format"])
    pred_field = job["pred_field"]
    predicates = job.get("predicates", PREDICATES_CSV)

    # Exclude eval indices from training data (CSV jobs only, by row position)
    if exclude_eval_indices and job["format"] == "csv":
        rows = [r for i, r in enumerate(rows) if i not in exclude_eval_indices]
        print(f"  [INFO] Excluded {len(exclude_eval_indices)} eval rows from training")

    by_pred = collections.defaultdict(list)
    for r in rows:
        label = r[pred_field].strip().lower()
        if label in predicates:
            by_pred[label].append(r)

    counts = {p: len(by_pred.get(p, [])) for p in predicates}
    actual_n = n_per_class or min(counts.values())

    print(f"\n{'='*60}")
    print(f"  {os.path.basename(job['input'])}  →  {os.path.basename(job['output'])}")
    print(f"{'='*60}")
    print(f"  Per-predicate counts (original → balanced at {actual_n}):")
    for p in predicates:
        n = counts.get(p, 0)
        marker = "← min" if n == min(counts.values()) and not n_per_class else ""
        print(f"    {p:<12}: {n:>4} → {min(n, actual_n):>4}  {marker}")

    if any(counts.get(p, 0) < actual_n for p in predicates):
        short = [p for p in predicates if counts.get(p, 0) < actual_n]
        raise ValueError(f"Not enough examples for: {short}. Lower --n.")

    random.seed(seed)
    balanced = []
    for p in predicates:
        balanced.extend(random.sample(by_pred[p], actual_n))
    random.shuffle(balanced)

    _write(balanced, job["output"], job["format"], fieldnames)

    # Verify
    out_rows, _ = _read(job["output"], job["format"])
    print(f"  Verification — {actual_n} × {len(predicates)} = {len(out_rows)} total  ✅")
    print(f"  Saved → {job['output']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",            type=int, default=None,
                        help="Examples per predicate (default: min across classes)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--exclude-eval", action="store_true",
                        help=f"Exclude eval indices ({EVAL_INDICES_FILE}) from training CSV")
    parser.add_argument("--csv-only",     action="store_true",
                        help="Only run the CSV job (skip JSONL)")
    parser.add_argument("--jsonl-only",   action="store_true",
                        help="Only run the JSONL job (skip CSV)")
    args = parser.parse_args()

    exclude_eval = None
    if args.exclude_eval:
        if not os.path.exists(EVAL_INDICES_FILE):
            print(f"[WARN] --exclude-eval: {EVAL_INDICES_FILE} not found — skipping exclusion")
        else:
            with open(EVAL_INDICES_FILE) as f:
                exclude_eval = set(json.load(f))
            print(f"[INFO] Will exclude {len(exclude_eval)} eval indices from CSV training data")

    for job in JOBS:
        fmt = job["format"]
        if args.csv_only  and fmt != "csv":   continue
        if args.jsonl_only and fmt != "jsonl": continue
        balance_dataset(job, args.n, args.seed, exclude_eval_indices=exclude_eval)

    print("\n✅  Done.")


if __name__ == "__main__":
    main()
