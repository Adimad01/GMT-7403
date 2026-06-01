"""
build_balanced_training_data.py
================================================================================
Balances the fine-tuning dataset by downsampling each DE-9IM predicate class
to the same number of examples (the minimum count across all classes).

Input  : ../dataset/triplet_update_v3_70.csv   (755 rows, imbalanced)
Output : ../dataset/triplet_balanced_train.csv  (N × 6 rows, balanced)

Predicate counts before balancing (triplet_update_v3_70.csv):
  touches  : 223
  within   : 186
  disjoint : 183
  overlaps :  83
  crosses  :  41
  contains :  39   ← minimum → all predicates sampled to 39

Result: 39 × 6 = 234 balanced training examples.

Usage:
    python build_balanced_training_data.py
    python build_balanced_training_data.py --n 41   # custom per-class count
    python build_balanced_training_data.py --seed 42
"""

import csv
import random
import argparse
import os
import collections

INPUT  = "../dataset/triplet_update_v3_70.csv"
OUTPUT = "../dataset/triplet_balanced_train.csv"

PREDICATES = ["contains", "within", "touches", "crosses", "disjoint", "overlaps"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=None,
                        help="Examples per predicate (default: min count across classes)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input",  default=INPUT)
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Group by predicate
    by_pred = collections.defaultdict(list)
    for r in rows:
        pred = r["spatial_relation"].strip().lower()
        by_pred[pred].append(r)

    counts = {p: len(by_pred[p]) for p in PREDICATES}
    min_count = args.n or min(counts.values())

    print(f"Input  : {args.input}  ({len(rows)} rows)")
    print(f"Output : {args.output}")
    print(f"\nPer-predicate counts (original → balanced):")
    for p in PREDICATES:
        n = counts.get(p, 0)
        flag = "⚠️ MISSING" if n == 0 else ("← min" if n == min_count and args.n is None else "")
        print(f"  {p:<12}: {n:>4} → {min(n, min_count):>4}  {flag}")
    print(f"\nBalancing at {min_count} per predicate → {min_count * len(PREDICATES)} total examples")

    if any(counts.get(p, 0) < min_count for p in PREDICATES):
        missing = [p for p in PREDICATES if counts.get(p, 0) < min_count]
        print(f"[ERROR] Predicates with fewer than {min_count} examples: {missing}")
        print("        Lower --n or check the input file.")
        raise SystemExit(1)

    balanced = []
    for p in PREDICATES:
        sample = random.sample(by_pred[p], min_count)
        balanced.extend(sample)

    random.shuffle(balanced)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(balanced)

    # Verify
    with open(args.output, newline="", encoding="utf-8") as f:
        out_rows = list(csv.DictReader(f))
    out_counts = collections.Counter(r["spatial_relation"] for r in out_rows)
    print(f"\nVerification — {args.output}:")
    for p in PREDICATES:
        print(f"  {p:<12}: {out_counts.get(p, 0)}")
    print(f"\n✅ Balanced dataset saved → {args.output}")


if __name__ == "__main__":
    main()
