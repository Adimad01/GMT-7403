"""
analyze_results.py
================================================================================
Unified analyzer for the 6-experiment × 3-strategy design.  Scans results/ for
the new exp1..exp6 checkpoints (voletc_<tag>_<strategy>_<suffix>_ckpt.json) and
prints the per-domain 6×3 accuracy matrix.

Domain-agnostic: drop a copy in any */code dir and run it there.

Usage:
    python analyze_results.py
    python analyze_results.py --results-dir results
"""
import os
import re
import glob
import json
import argparse

STRATS = ["cot", "tot", "got"]

# Canonical experiment order + friendly labels (unknown tags are appended).
EXP_ORDER = [
    ("exp1_base",          "Exp1  base / no-KG"),
    ("exp2_ft_nokg",       "Exp2  no-KG LoRA"),
    ("exp3_ft_osmkg",      "Exp3  OSM-KG LoRA (KG@train)"),
    ("exp4_base_kg_input", "Exp4  base + KG@input"),
    ("exp5_ft_kg_input",   "Exp5  no-KG LoRA + KG@input"),
    ("exp6_base_kg_rag",   "Exp6  base + KG@inference (RAG)"),
]
_CKPT_RE = re.compile(r"voletc_(?P<tag>.+?)_(?P<strat>cot|tot|got)_.*_ckpt\.json$")


def _accuracy(path: str):
    try:
        data = json.load(open(path))
    except Exception:
        return None
    results = data.get("results", [])
    if not results:
        return (0.0, 0)
    hits = sum(1 for r in results if r.get("match"))
    return (hits / len(results) * 100.0, len(results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    table: dict = {}   # tag -> {strat: (acc, n)}
    for path in glob.glob(os.path.join(args.results_dir, "voletc_*_ckpt.json")):
        m = _CKPT_RE.search(os.path.basename(path))
        if not m:
            continue
        acc = _accuracy(path)
        if acc is None:
            continue
        table.setdefault(m.group("tag"), {})[m.group("strat")] = acc

    if not table:
        print(f"[INFO] No result checkpoints found in {args.results_dir}/")
        return

    ordered = [(t, lbl) for t, lbl in EXP_ORDER if t in table]
    extras = [(t, t) for t in sorted(table) if t not in dict(EXP_ORDER)]
    rows = ordered + extras

    width = max(len(lbl) for _, lbl in rows)
    header = f"{'Experiment':<{width}} | " + " | ".join(f"{s.upper():>10}" for s in STRATS)
    print(header)
    print("-" * len(header))
    for tag, lbl in rows:
        cells = []
        for s in STRATS:
            if s in table[tag]:
                acc, n = table[tag][s]
                cells.append(f"{acc:6.1f}% ({n:>2})")
            else:
                cells.append(f"{'—':>10}")
        print(f"{lbl:<{width}} | " + " | ".join(f"{c:>10}" for c in cells))
    print("\n(accuracy %% with sample count; — = not yet run)")


if __name__ == "__main__":
    main()
