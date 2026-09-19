"""Check the knowledge base before it is shown to a model.

The store exists to supply facts the model may lack. If it also supplies the
answer, the arm measures retrieval and nothing else -- so the first question
is always whether a node names the other entity of a pair it is asked about.

The node's own name is excluded from that test: both names are already in the
prompt as subject and object, so a node called "Lake District National Park"
containing "Lake District" adds nothing the model did not have.

    python3 scripts/check_kg.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RELATIONS = ("topological", "cardinal", "relative")
QUALIFIERS = ("City of ", "State of ", "Province of ", "Borough of ")


def short(name: str) -> str:
    for q in QUALIFIERS:
        if name.startswith(q):
            return name[len(q):]
    return name


def main() -> int:
    problems = []
    for rel in RELATIONS:
        path = REPO / "data" / rel / "kg_eval.json"
        if not path.exists():
            problems.append(f"{rel}: kg_eval.json absent")
            continue
        nodes = json.loads(path.read_text(encoding="utf-8"))["nodes"]
        rows = list(csv.DictReader(
            (REPO / "data" / rel / "eval.csv").open(encoding="utf-8")))
        cols = ["source_entity", "target_entity"]
        if rel == "relative":
            cols.append("observer_entity")

        uncovered = {r[c] for r in rows for c in cols if r.get(c)} - set(nodes)
        leaks = []
        for r in rows:
            s, t = r["source_entity"], r["target_entity"]
            for a, b in ((s, t), (t, s)):
                blob = json.dumps({k: v for k, v in nodes.get(a, {}).items()
                                   if k != "name"}, ensure_ascii=False).lower()
                if short(b).lower() in blob:
                    leaks.append((r["relation_label"], a, b))
                    break

        # A fact that locates nothing is worse than no fact: it fills the
        # prompt and tells the model where a place is not.
        vague = [n for n, f in nodes.items()
                 if f.get("main_part_share", 1.0) < 0.35]

        print(f"  {rel:<13}{len(nodes):>4} nœuds   "
              f"{len(uncovered)} entité(s) sans fait   "
              f"{len(leaks)} fuite(s)   "
              f"{len(vague)} centroïde(s) peu représentatif(s)")
        if uncovered:
            problems.append(f"{rel}: {len(uncovered)} entités sans fait "
                            f"({', '.join(sorted(uncovered)[:4])})")
        if leaks:
            problems.append(f"{rel}: {len(leaks)} ligne(s) dont un nœud nomme "
                            f"l'autre entité — ex. {leaks[0]}")
        if vague:
            print(f"    {'':<13}dont : {', '.join(vague[:4])}")

    print()
    if problems:
        print(f"  {len(problems)} problème(s) :")
        for p in problems:
            print(f"    ✗ {p}")
        return 1
    print("  la base couvre toutes les entités et ne révèle aucune réponse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
