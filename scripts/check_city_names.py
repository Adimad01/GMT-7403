"""Cities whose recorded coordinate belongs to a different place of that name.

The cardinal and relative ground truth was computed from a coordinate table,
and a short ambiguous name resolves to whichever homonym ranked highest:
Tanga to Tonga, Medina to Medina in Texas, Kolonia to Köln. The label is then
consistent with the coordinate and wrong for the name, so a model that knows
the real geography is marked incorrect.

That cuts the other way after fine-tuning. The adapter trains on the same
table, learns the corpus's own geography, and scores as if right -- so the
affected rows flatter fine-tuning and penalise the base model. This measures
the size of that, rather than leaving it to be assumed either way.

    python3 scripts/check_city_names.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Verified by hand against the place the name denotes. Not the full list: the
# neighbourhood heuristic in data_generation/find_misresolved_cities.py flags
# 25 of 296, of which these are the ones confirmed wrong rather than merely
# isolated -- Pago Pago and Nouméa sit far from their neighbours and are right.
MISRESOLVED = {
    "Tanga": "Tonga rather than Tanzania",
    "Tarawa": "Nigeria rather than Kiribati",
    "Kolonia": "Köln rather than Micronesia",
    "Medina": "Texas rather than Saudi Arabia",
    "Belem": "Bethlehem rather than Brazil",
    "Ibarra": "Spain rather than Ecuador",
    "Loja": "Spain rather than Ecuador",
    "Aba": "Hungary rather than Nigeria",
}


def main() -> int:
    print("  villes dont la coordonnée désigne un autre lieu :")
    for name, what in sorted(MISRESOLVED.items()):
        print(f"    {name:<10}{what}")

    for rel in ("cardinal", "relative"):
        ev = list(csv.DictReader(
            (REPO / "data" / rel / "eval.csv").open(encoding="utf-8")))
        cols = ["source_entity", "target_entity"]
        if rel == "relative":
            cols.append("observer_entity")
        tainted = {i for i, r in enumerate(ev)
                   if any((r.get(c) or "").replace("City of ", "") in MISRESOLVED
                          for c in cols)}

        def rows(variant):
            p = (REPO / "results" / rel / "zero_shot"
                 / f"seed1{variant}" / "predictions.jsonl")
            if not p.exists():
                return {}
            return {r["row_index"]: r for r in
                    map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r}

        base, tuned = rows(""), rows("_lora")
        print(f"\n  {rel} — {len(tainted)}/{len(ev)} lignes touchées "
              f"({len(tainted) / len(ev) * 100:.1f} %)")
        if not base or not tuned:
            print("    (résultats absents)")
            continue
        for label, idx in (("saines", set(base) - tainted), ("fautives", tainted)):
            idx = {i for i in idx
                   if base.get(i, {}).get("ambiguity_level") != "Level 6"}
            if not idx:
                continue
            b = sum(1 for i in idx if base[i]["correct"]) / len(idx) * 100
            t = sum(1 for i in idx if tuned[i]["correct"]) / len(idx) * 100
            print(f"    {label:<10}n={len(idx):<4} base {b:>5.1f} %  "
                  f"affiné {t:>5.1f} %   ({t - b:+.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
