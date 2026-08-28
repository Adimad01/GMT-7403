"""
build_dataset_topological_v2.py
================================================================================
Splits topological_relations.csv into a stratified train / eval pair.

Source : ../dataset/topological_relations.csv
         1309 valid rows · 7 DE-9IM predicates · 5 ambiguity levels (~37/cell)

Eval split  : 3 examples × 5 levels × 7 predicates = 105 rows
Train split : remaining 1204 rows

Columns are renamed to match the eval_engine_gpu.py / train_lora_adapter.py
expected schema:
  relation_predicate   ← corpus
  spatial_relation     ← relation_label
  geometry_type_subject ← source_geometry
  geometry_type_object  ← target_geometry
  place_name_subject   ← source_entity  (also used as placetype_subject)
  place_name_object    ← target_entity  (also used as placetype_object)
  ambiguity_level      kept as-is
  explanation          kept as-is

Outputs
  ../dataset/topo_v2_eval.csv            105 balanced test rows
  ../dataset/topo_v2_eval_indices.json   [0 … 104] — all rows in the eval CSV
  ../dataset/topo_v2_train.csv           1204 training rows

Usage
  python build_dataset_topological_v2.py
  python build_dataset_topological_v2.py --n-per-cell 3 --seed 42
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

PREDICATES = ["contains", "within", "touches", "crosses", "disjoint", "overlaps", "equals"]
LEVELS     = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]

SOURCE = Path("../dataset/topological_relations.csv")
OUT_EVAL   = Path("../dataset/topo_v2_eval.csv")
OUT_TRAIN  = Path("../dataset/topo_v2_train.csv")
OUT_IDX    = Path("../dataset/topo_v2_eval_indices.json")


def remap(row: dict) -> dict:
    """Rename columns from source schema to eval_engine_gpu.py schema."""
    entity_src = row["source_entity"].strip()
    entity_tgt = row["target_entity"].strip()
    return {
        "relation_predicate":    row["corpus"].strip(),
        "spatial_relation":      row["relation_label"].strip().lower(),
        "geometry_type_subject": row["source_geometry"].strip(),
        "geometry_type_object":  row["target_geometry"].strip(),
        "place_name_subject":    entity_src,
        "placetype_subject":     entity_src,
        "place_name_object":     entity_tgt,
        "placetype_object":      entity_tgt,
        "explanation":           row.get("explanation", "").strip(),
        "ambiguity_level":       row["ambiguity_level"].strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-cell", type=int, default=3,
                        help="Eval examples per (predicate, level) cell (default 3)")
    parser.add_argument("--train-per-cell", type=int, default=None,
                        help="Cap each (predicate, level) cell in the TRAIN split to "
                             "this many rows, so train is balanced across predicate "
                             "AND ambiguity level. Default: keep every remaining row "
                             "(unbalanced, legacy behaviour).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Load & clean ──────────────────────────────────────────────────────────
    with open(SOURCE, newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))

    valid = [r for r in raw
             if r["relation_label"].strip() not in ("relation_label", "")
             and r["ambiguity_level"].strip() not in ("ambiguity_level", "")]

    print(f"[INFO] Valid rows: {len(valid)}  (removed {len(raw)-len(valid)} garbage header rows)")

    # ── Group by (predicate, level) ───────────────────────────────────────────
    cells: dict[tuple, list] = defaultdict(list)
    for r in valid:
        key = (r["relation_label"].strip().lower(), r["ambiguity_level"].strip())
        cells[key].append(r)

    # ── Stratified split (eval drawn from OSM-geocodable rows only) ──────────
    # Eval rows must pass the runtime OSM filter, so sample them exclusively
    # from candidates whose BOTH entities resolve in the warmed geocode cache.
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from osm_client import load_cache, is_geocodable
    cache = load_cache(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                     "results", "osm_cache.json"))
    if not cache:
        raise SystemExit("[ERROR] results/osm_cache.json missing/empty — warm it first "
                         "(warm_osm_cache.py); eval selection requires geocodability.")

    eval_rows, train_rows = [], []
    stats = {}
    for pred in PREDICATES:
        for lvl in LEVELS:
            pool = cells[(pred, lvl)]
            geo_pool = [r for r in pool
                        if is_geocodable(cache, r["source_entity"], r["target_entity"])]
            if len(geo_pool) < args.n_per_cell:
                raise ValueError(
                    f"Only {len(geo_pool)} geocodable examples for ({pred}, {lvl}) — "
                    f"need {args.n_per_cell}. Warm more entities of this cell "
                    f"(warm_osm_cache.py) and retry."
                )
            chosen = rng.sample(geo_pool, args.n_per_cell)
            chosen_set = set(id(r) for r in chosen)
            eval_rows.extend(chosen)

            remaining = [r for r in pool if id(r) not in chosen_set]
            if args.train_per_cell is not None:
                # Cap every cell to the same size so the train split is balanced
                # across BOTH predicate and ambiguity level. Prefer geocodable
                # rows so the KG arms keep as many usable examples as possible;
                # top up with ungeocodable rows only if the cell is short.
                if len(remaining) < args.train_per_cell:
                    raise ValueError(
                        f"Only {len(remaining)} rows left for ({pred}, {lvl}) after the "
                        f"eval draw — need {args.train_per_cell}. Lower --train-per-cell."
                    )
                geo = [r for r in remaining
                       if is_geocodable(cache, r["source_entity"], r["target_entity"])]
                geo_ids = set(id(r) for r in geo)
                non = [r for r in remaining if id(r) not in geo_ids]
                if len(geo) >= args.train_per_cell:
                    take = rng.sample(geo, args.train_per_cell)
                else:
                    take = geo + rng.sample(non, args.train_per_cell - len(geo))
                remaining = take

            train_rows.extend(remaining)
            stats[(pred, lvl)] = (len(chosen), len(remaining))

    n_eval  = len(eval_rows)
    n_train = len(train_rows)
    n_total = n_eval + n_train

    # ── Pretty-print split summary ────────────────────────────────────────────
    print(f"\n{'':=<70}")
    print(f"  Split summary  (n_per_cell={args.n_per_cell}  seed={args.seed})")
    print(f"{'':=<70}")
    header = f"  {'predicate':<12}" + "".join(f"  {l:<10}" for l in LEVELS)
    print(header)
    for pred in PREDICATES:
        row_str = f"  {pred:<12}"
        for lvl in LEVELS:
            e, t = stats[(pred, lvl)]
            row_str += f"  eval={e} tr={t}"
        print(row_str)
    print(f"{'':=<70}")
    print(f"  Total  →  eval: {n_eval}  train: {n_train}  (total valid: {n_total})")
    print(f"{'':=<70}\n")

    # ── Write eval CSV ────────────────────────────────────────────────────────
    eval_mapped  = [remap(r) for r in eval_rows]
    train_mapped = [remap(r) for r in train_rows]

    fieldnames = list(eval_mapped[0].keys())

    with open(OUT_EVAL, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(eval_mapped)
    print(f"[OK] Eval CSV   → {OUT_EVAL}  ({n_eval} rows)")

    # ── Write eval indices (all rows of the eval CSV) ─────────────────────────
    indices = list(range(n_eval))
    with open(OUT_IDX, "w", encoding="utf-8") as f:
        json.dump(indices, f)
    print(f"[OK] Eval idx   → {OUT_IDX}  ({n_eval} indices)")

    # ── Write train CSV ───────────────────────────────────────────────────────
    with open(OUT_TRAIN, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(train_mapped)
    print(f"[OK] Train CSV  → {OUT_TRAIN}  ({n_train} rows)")

    # ── Verification ──────────────────────────────────────────────────────────
    from collections import Counter
    eval_dist  = Counter(r["spatial_relation"] for r in eval_mapped)
    train_dist = Counter(r["spatial_relation"] for r in train_mapped)
    print(f"\n  Eval predicate distribution  (expected {args.n_per_cell * len(LEVELS)} each):")
    for p in PREDICATES:
        print(f"    {p:<12}: {eval_dist[p]:>3}")
    print(f"\n  Train predicate distribution:")
    for p in PREDICATES:
        print(f"    {p:<12}: {train_dist[p]:>3}")
    print()


if __name__ == "__main__":
    main()
