"""
build_cardinal_training_data.py
================================================================================
Builds balanced training datasets from cardinal_direction_relations.csv
(5760 rows, 8 directions × 720), excluding the 440-row balanced eval split.

Outputs:
  ../dataset/cardinal_balanced_train.csv
      Balanced training CSV: 130 examples per direction × 8 = 1040 rows.

  ../dataset/cardinal_train.jsonl
      Plain question → direction pairs (used by Config 2 & 5).

  ../dataset/cardinal_kg_train.jsonl
      Compass-rule-enriched pairs: rules + question → direction (Config 3).

JSONL record format:
  {"text": "<full instruction+answer>", "label": "<direction>"}

Usage:
    python build_cardinal_training_data.py
    python build_cardinal_training_data.py --exclude-eval
    python build_cardinal_training_data.py --seed 42
"""

import csv
import json
import random
import argparse
import os
import sys
from collections import defaultdict

DIRECTIONS = [
    "north", "south", "east", "west",
    "north-east", "north-west", "south-east", "south-west",
]

VALID_LIST = "north, south, east, west, north-east, north-west, south-east, south-west"

COMPASS_RULES = """Compass / Shore-to-Body Rules:
  SHORE-TO-BODY: the water lies on the OPPOSITE side of the shore name.
    north shore  → water to the south
    south shore  → water to the north
    east shore   → water to the west
    west shore   → water to the east
    north-east shore → water to the south-west
    north-west shore → water to the south-east
    south-east shore → water to the north-west
    south-west shore → water to the north-east
  TURN-AROUND: "turns around" / "heads back" reverses travel direction
    but the shore stays the same — water direction is unchanged."""

PLAIN_TEMPLATE = """\
### SYSTEM PROMPT ###
You are an expert in spatial reasoning and cardinal directions.

### TASK ###
Determine the cardinal direction from the person to the water body described in the question.

### VALID DIRECTIONS ###
{valid_list}

### RULES ###
1. Identify which shore of the water body the person is on.
2. The water body lies on the OPPOSITE side of the shore name.
3. If the person turns around, the shore stays the same; only travel direction reverses.

### QUESTION ###
{question}

### ANSWER ###
{direction}"""

KG_TEMPLATE = """\
### SYSTEM PROMPT ###
You are an expert in spatial reasoning and cardinal directions.

### TASK ###
Determine the cardinal direction from the person to the water body described in the question.

### VALID DIRECTIONS ###
{valid_list}

### KNOWLEDGE GRAPH ###
{rules}

### QUESTION ###
{question}

### ANSWER ###
{direction}"""


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  → wrote {len(records)} records to {path}")


def build(args):
    full_path   = "../dataset/cardinal_direction_relations.csv"
    eval_idx_path = "../dataset/eval_440_balanced_indices.json"
    train_csv_path = "../dataset/cardinal_balanced_train.csv"
    plain_path  = "../dataset/cardinal_train.jsonl"
    kg_path     = "../dataset/cardinal_kg_train.jsonl"

    if not os.path.exists(full_path):
        print(f"[ERROR] Input not found: {full_path}")
        sys.exit(1)

    rows = _read_csv(full_path)
    fieldnames = list(rows[0].keys()) if rows else []
    print(f"[DATA] Loaded {len(rows)} rows from {full_path}")

    # Exclude eval indices
    eval_indices: set = set()
    if args.exclude_eval:
        if not os.path.exists(eval_idx_path):
            print(f"[ERROR] Eval indices not found: {eval_idx_path}")
            sys.exit(1)
        with open(eval_idx_path) as f:
            eval_indices = set(json.load(f))
        print(f"[DATA] Excluding {len(eval_indices)} eval indices")

    # Partition by direction, excluding eval rows
    by_dir: defaultdict = defaultdict(list)
    for i, row in enumerate(rows):
        if i in eval_indices:
            continue
        d = row["direction"].strip().lower()
        if d in DIRECTIONS:
            by_dir[d].append(row)

    print("[DATA] Available training rows per direction (after eval exclusion):")
    for d in DIRECTIONS:
        print(f"         {d:15s}: {len(by_dir[d])}")

    # Balance at 130 per direction
    PER_DIR = 130
    random.seed(args.seed)
    train_rows = []
    for d in DIRECTIONS:
        pool = by_dir[d]
        if len(pool) < PER_DIR:
            print(f"[WARN] {d}: only {len(pool)} rows available (need {PER_DIR}), using all")
            selected = pool
        else:
            selected = random.sample(pool, PER_DIR)
        train_rows.extend(selected)

    random.shuffle(train_rows)
    print(f"[DATA] Total training rows: {len(train_rows)} ({PER_DIR}/direction × {len(DIRECTIONS)})")

    # Write balanced train CSV
    os.makedirs(os.path.dirname(os.path.abspath(train_csv_path)), exist_ok=True)
    with open(train_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)
    print(f"  → wrote {len(train_rows)} rows to {train_csv_path}")

    dist = {}
    for row in train_rows:
        d = row["direction"].strip().lower()
        dist[d] = dist.get(d, 0) + 1
    print("[DATA] Training direction distribution:")
    for d in DIRECTIONS:
        print(f"         {d:15s}: {dist.get(d, 0)}")

    plain_records = []
    kg_records    = []

    for row in train_rows:
        question  = row["question"].strip()
        direction = row["direction"].strip().lower()

        if direction not in DIRECTIONS:
            print(f"[WARN] Skipping unknown direction: {direction!r}")
            continue

        plain_text = PLAIN_TEMPLATE.format(
            valid_list=VALID_LIST, question=question, direction=direction
        )
        kg_text = KG_TEMPLATE.format(
            valid_list=VALID_LIST, rules=COMPASS_RULES,
            question=question, direction=direction
        )

        plain_records.append({"text": plain_text, "label": direction})
        kg_records.append({"text": kg_text, "label": direction})

    print(f"\n[BUILD] Writing plain training data ...")
    _write_jsonl(plain_records, plain_path)

    print(f"[BUILD] Writing KG-enriched training data ...")
    _write_jsonl(kg_records, kg_path)

    print("\n[DONE] Cardinal training datasets ready.")
    print(f"  Train CSV : {train_csv_path}  ({len(train_rows)} rows, balanced)")
    print(f"  Plain     : {plain_path}  ({len(plain_records)} examples)")
    print(f"  KG        : {kg_path}     ({len(kg_records)} examples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-eval", action="store_true",
                        help="Exclude the 440 eval indices from training data")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
