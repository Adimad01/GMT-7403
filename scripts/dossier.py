"""Everything the experiments have produced, on one page.

Written to be read by the person who ran them, as reference for describing
the work in their own words: the corpus as it stands, every arm with its
figure, what the checks found, and what has not been run.

    python3 scripts/dossier.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RELATIONS = ("topological", "cardinal", "relative")
STRATEGIES = ("zero_shot", "cot", "few_shot", "tot", "got")
ARMS = (("base", ""), ("+ adaptateur", "_lora"),
        ("+ base de connaissances", "_kg"), ("adaptateur + base", "_lora_kg"))


def acc(rel, strat, var=""):
    p = REPO / "results" / rel / strat / f"seed1{var}" / "predictions.jsonl"
    if not (p.parent / "run.json").exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows
          if r.get("status") == "ok" and r.get("ambiguity_level") != "Level 6"]
    return 100 * sum(1 for r in ok if r["correct"]) / len(ok) if ok else None


def main() -> int:
    print("=" * 72)
    print("  1. CORPUS")
    print("=" * 72)
    total = {}
    for rel in RELATIONS:
        counts = {}
        for split in ("corpus", "train", "eval"):
            f = REPO / "data" / rel / f"{split}.csv"
            counts[split] = sum(1 for _ in csv.DictReader(f.open(encoding="utf-8")))
        labels = {r["relation_label"] for r in
                  csv.DictReader((REPO / "data" / rel / "corpus.csv").open(encoding="utf-8"))}
        total[rel] = counts
        print(f"  {rel:<14}{counts['corpus']:>5} items   "
              f"{counts['train']:>5} entraînement   {counts['eval']:>4} évaluation   "
              f"{len(labels)} prédicats")
    print(f"  {'TOTAL':<14}{sum(c['corpus'] for c in total.values()):>5} items   "
          f"{sum(c['train'] for c in total.values()):>5}"
          f"   {sum(c['eval'] for c in total.values()):>16}")

    print("\n" + "=" * 72)
    print("  2. RÉSULTATS   (exactitude, niveau 6 exclu)")
    print("=" * 72)
    print(f"  {'':<26}" + "".join(f"{s:>12}" for s in STRATEGIES))
    for rel in RELATIONS:
        for label, var in ARMS:
            row = [acc(rel, s, var) for s in STRATEGIES]
            if not any(v is not None for v in row):
                continue
            cells = "".join(f"{v:>11.1f}%" if v is not None else f"{'—':>12}"
                            for v in row)
            print(f"  {rel[:11]:<12}{label:<14}{cells}")
        print()

    print("=" * 72)
    print("  3. CE QUI N'A PAS ÉTÉ MESURÉ")
    print("=" * 72)
    missing = []
    for label, var in ARMS:
        for rel in RELATIONS:
            for s in STRATEGIES:
                if s == "few_shot" and "_lora" in var:
                    continue          # démonstrations = jeu d'entraînement
                if acc(rel, s, var) is None:
                    missing.append(f"{rel}/{s} {label}")
    if missing:
        for m in missing:
            print(f"    {m}")
    else:
        print("    rien")
    print(f"\n    soit {len(missing)} cellule(s)")

    print("\n" + "=" * 72)
    print("  4. CONTRÔLES")
    print("=" * 72)
    print("    ligne de base lexicale (sac de mots, sans géométrie) :")
    print("       topologique 77.2 %   cardinal 28.3 %   relatif 41.6 %")
    print("    patrons partagés entre entraînement et évaluation : 1 sur 814")
    print("    villes dont la coordonnée désigne un autre lieu : 8")
    print("       -> 29 lignes d'évaluation sur 578 (5.0 %)")
    print("    base de connaissances : couverture 100 %, 0 fuite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
