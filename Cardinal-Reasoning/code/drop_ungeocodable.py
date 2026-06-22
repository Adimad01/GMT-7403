"""
drop_ungeocodable.py
================================================================================
Restricts the evaluation to rows whose BOTH entities geocoded successfully in
results/osm_cache.json, and rewrites the result checkpoints to that subset.

Why: for rows where Nominatim returned nothing, the KG experiments (Exp 3-6)
effectively run WITHOUT OSM evidence, so including them is not a fair KG vs
no-KG comparison.  Dropping them — uniformly across all experiments — keeps
every experiment scored on the same well-grounded subset.

An entity is "ungeocodable" if it is absent from the cache or cached as null.
The drop set is computed once (from cache + dataset) and applied to every
voletc_*_ckpt.json in the results dir, so the kept index set stays identical
across all experiments.

Dry-run by default (reports only).  Pass --apply to rewrite the checkpoints
(originals are backed up to results/unfiltered_backup/ on first apply).

Usage:
  # Cardinal
  python drop_ungeocodable.py --dataset ../dataset/cardinal_direction_relations.csv \\
                              --eval-indices ../dataset/eval_40_balanced_indices.json
  # Topological (eval CSV already IS the eval set; indices are 0..104)
  python drop_ungeocodable.py --dataset ../dataset/topo_v2_eval.csv \\
                              --eval-indices ../dataset/topo_v2_eval_indices.json --apply
"""
import os
import csv
import json
import glob
import shutil
import argparse

# entity column name candidates (different domains remap differently)
_SRC_COLS = ["source_entity", "place_name_subject"]
_TGT_COLS = ["target_entity", "place_name_object"]


def _pick(fieldnames, candidates):
    for c in candidates:
        if c in fieldnames:
            return c
    raise SystemExit(f"[ERROR] none of {candidates} found in dataset columns {fieldnames}")


def _geocoded(cache, name):
    name = (name or "").strip()
    return bool(name) and cache.get(name) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-indices", required=True)
    ap.add_argument("--cache", default="results/osm_cache.json")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--apply", action="store_true",
                    help="Rewrite the checkpoints (default: dry-run report only)")
    args = ap.parse_args()

    for p in (args.dataset, args.eval_indices, args.cache):
        if not os.path.exists(p):
            raise SystemExit(f"[ERROR] not found: {p}")

    cache = json.load(open(args.cache))
    rows = list(csv.DictReader(open(args.dataset, newline="", encoding="utf-8")))
    src_col = _pick(rows[0].keys(), _SRC_COLS)
    tgt_col = _pick(rows[0].keys(), _TGT_COLS)

    # --- compute the drop set (over the eval indices) ---------------------
    eval_idx = json.load(open(args.eval_indices))
    drop = {}
    for i in eval_idx:
        if i >= len(rows):
            continue
        s, t = rows[i].get(src_col, ""), rows[i].get(tgt_col, "")
        miss = []
        if not _geocoded(cache, s):
            miss.append(f"src {s!r}")
        if not _geocoded(cache, t):
            miss.append(f"tgt {t!r}")
        if miss:
            drop[i] = "; ".join(miss)

    print(f"[DATA] eval rows: {len(eval_idx)}  |  entity cols: {src_col}/{tgt_col}")
    print(f"[DROP] ungeocodable eval rows: {len(drop)}")
    for i, why in sorted(drop.items()):
        print(f"   idx {i}: {why}")
    if not drop:
        print("[OK] nothing to drop — eval set is fully geocoded.")
        return

    drop_set = set(drop)
    ckpts = sorted(glob.glob(os.path.join(args.results_dir, "voletc_*_ckpt.json")))
    if not ckpts:
        print(f"[INFO] no checkpoints found in {args.results_dir}/")
        return

    backup_dir = os.path.join(args.results_dir, "unfiltered_backup")
    print(f"\n{'file':<60} {'before':>16} {'after':>16}")
    print("-" * 94)
    for path in ckpts:
        data = json.load(open(path))
        results = data.get("results", [])
        if not results:
            continue
        kept = [r for r in results if r.get("index") not in drop_set]
        b_n = len(results); b_acc = sum(1 for r in results if r.get("match")) / b_n * 100 if b_n else 0
        a_n = len(kept);    a_acc = sum(1 for r in kept if r.get("match")) / a_n * 100 if a_n else 0
        name = os.path.basename(path)
        print(f"{name:<60} {b_acc:6.1f}% ({b_n:>3}) {a_acc:6.1f}% ({a_n:>3})")

        if args.apply and a_n != b_n:
            os.makedirs(backup_dir, exist_ok=True)
            bak = os.path.join(backup_dir, name)
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            data["results"] = kept
            data["processed_indices"] = sorted(r["index"] for r in kept)
            data["dropped_ungeocodable"] = sorted(drop_set)
            json.dump(data, open(path, "w"), indent=2, ensure_ascii=False)

    if args.apply:
        print(f"\n[APPLIED] checkpoints rewritten; originals backed up in {backup_dir}/")
    else:
        print("\n[DRY-RUN] no files changed. Re-run with --apply to rewrite checkpoints.")


if __name__ == "__main__":
    main()
