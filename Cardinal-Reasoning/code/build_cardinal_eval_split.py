"""
build_cardinal_eval_split.py
================================================================================
Builds a LEVEL-STRATIFIED Cardinal eval split. The old eval_32 indices were the
first 32 rows of the CSV (24× Level-1 + 8× Level-2 only) — levels 3/4/5 were
never tested. This picks 1 row per (direction × ambiguity-level) → 40 eval rows:
8 per level (balanced across L1–L5) and 5 per direction.

Output: ../dataset/eval_40_balanced_indices.json   (sorted row indices)

Re-run order after this (the train split changes, so adapters must be retrained):
  python build_cardinal_eval_split.py
  python build_cardinal_train_data.py
  python train_runner_cardinal_nokg.py
  python train_runner_cardinal_osm_kg.py
  for e in exp1_base exp2_ft_nokg exp3_ft_osmkg exp4_base_kg_input exp5_ft_kg_input exp6_base_kg_rag; do
      python $e.py; python $e.py --shots 5
  done
"""
import csv
import json
import random
import collections

DATASET = "../dataset/cardinal_direction_relations.csv"
OUT = "../dataset/eval_40_balanced_indices.json"
SEED = 42


def main():
    rows = list(csv.DictReader(open(DATASET, newline="", encoding="utf-8")))
    cells = collections.defaultdict(list)
    for i, r in enumerate(rows):
        lab = r["relation_label"].strip().lower()
        lvl = r["ambiguity_level"].strip()
        cells[(lab, lvl)].append(i)

    rng = random.Random(SEED)
    chosen = []
    for key in sorted(cells):
        chosen.append(rng.choice(cells[key]))
    chosen = sorted(chosen)

    json.dump(chosen, open(OUT, "w"))
    print(f"[OK] wrote {OUT}  ({len(chosen)} eval rows)")

    by_lvl = collections.Counter(rows[i]["ambiguity_level"].strip() for i in chosen)
    by_dir = collections.Counter(rows[i]["relation_label"].strip().lower() for i in chosen)
    print("[DATA] per level    :", dict(sorted(by_lvl.items())))
    print("[DATA] per direction:", dict(sorted(by_dir.items())))
    print(f"[DATA] train rows remaining: {len(rows) - len(chosen)}")


if __name__ == "__main__":
    main()
