"""
build_cardinal_training_data.py
================================================================================
Converts cardinal_train.csv (5680 rows, 8 directions × 710) into two JSONL
instruction-tuning datasets:

  ../dataset/cardinal_train.jsonl
      Plain question → direction pairs (used by Config 2 & 5).

  ../dataset/cardinal_kg_train.jsonl
      Compass-rule-enriched pairs: rules + question → direction (Config 3).

JSONL record format:
  {"text": "<full instruction+answer>", "label": "<direction>"}

Usage:
    python build_cardinal_training_data.py
    python build_cardinal_training_data.py --seed 42
"""

import csv
import json
import random
import argparse
import os
import sys

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
    input_path  = "../dataset/cardinal_train.csv"
    plain_path  = "../dataset/cardinal_train.jsonl"
    kg_path     = "../dataset/cardinal_kg_train.jsonl"

    if not os.path.exists(input_path):
        print(f"[ERROR] Input not found: {input_path}")
        sys.exit(1)

    rows = _read_csv(input_path)
    print(f"[DATA] Loaded {len(rows)} rows from {input_path}")

    dist = {}
    for row in rows:
        d = row["direction"].strip().lower()
        dist[d] = dist.get(d, 0) + 1
    print("[DATA] Direction distribution:")
    for d in DIRECTIONS:
        print(f"         {d:15s}: {dist.get(d, 0)}")

    random.seed(args.seed)
    random.shuffle(rows)

    plain_records = []
    kg_records    = []

    for row in rows:
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
    print(f"  Plain : {plain_path}  ({len(plain_records)} examples)")
    print(f"  KG    : {kg_path}     ({len(kg_records)} examples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
