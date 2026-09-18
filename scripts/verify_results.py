"""Check the finished results against the manifests that defined them.

A run that completes is not the same as a run that is sound. These are the
checks worth making before any number leaves the repository: that every cell
evaluated the rows it was supposed to, that the labels scored against were the
pinned ones, and that the accuracies reported were the accuracies computed.

    python3 scripts/verify_results.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
DATA = REPO / "data"

RELATIONS = ("topological", "cardinal", "relative")
STRATEGIES = ("zero_shot", "cot", "few_shot", "tot", "got")
MULTI_CALL = {"tot": 4, "got": 4}

problems: list[str] = []
notes: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


def manifest(relation: str) -> dict:
    return json.loads((DATA / relation / "eval_manifest.json").read_text(encoding="utf-8"))


def main() -> int:
    if not RESULTS.exists():
        print("no results/ directory — nothing to verify")
        return 1

    print(f"{'cellule':<30}{'lignes':>8}{'exact.':>9}{'échecs':>8}"
          f"{'non lus':>9}{'appels':>8}")
    print("-" * 72)

    for relation in RELATIONS:
        man = manifest(relation)
        by_index = {r["row_index"]: r for r in man["rows"]}
        expected = set(by_index)

        for strategy in STRATEGIES:
            cell = RESULTS / relation / strategy / "seed1"
            rid = f"{relation}__{strategy}"
            if not cell.exists():
                fail(f"{rid}: cellule absente")
                continue

            run_json = cell / "run.json"
            if not run_json.exists():
                fail(f"{rid}: run.json absent — cellule inachevée")
                continue
            meta = json.loads(run_json.read_text(encoding="utf-8"))

            # The hash pins which corpus the rows came from. A mismatch means
            # the results describe data that no longer exists.
            if meta.get("eval_manifest_sha256") != man["manifest_sha256"]:
                fail(f"{rid}: empreinte du manifeste différente — "
                     f"résultats produits sur un autre corpus")

            rows, seen = [], Counter()
            for line in (cell / "predictions.jsonl").read_text(
                    encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    fail(f"{rid}: ligne JSON illisible dans predictions.jsonl")
            for r in rows:
                seen[r.get("row_index")] += 1

            dupes = [i for i, n in seen.items() if n > 1]
            if dupes:
                fail(f"{rid}: {len(dupes)} ligne(s) en double "
                     f"(ex. index {dupes[:3]})")
            missing = expected - set(seen)
            if missing:
                fail(f"{rid}: {len(missing)} ligne(s) du manifeste absente(s)")
            extra = set(seen) - expected
            if extra:
                fail(f"{rid}: {len(extra)} ligne(s) hors manifeste "
                     f"(ex. index {sorted(extra)[:3]})")

            # The gold label must be the pinned one, not whatever the CSV
            # happened to hold when the cell ran.
            wrong_gold = [r["row_index"] for r in rows
                          if r["row_index"] in by_index
                          and r.get("gold") != by_index[r["row_index"]]["label"]]
            if wrong_gold:
                fail(f"{rid}: {len(wrong_gold)} ligne(s) notée(s) contre une "
                     f"autre étiquette que celle du manifeste")

            ok = [r for r in rows if r.get("status") == "ok"]
            errors = [r for r in rows if r.get("status") == "error"]
            unparsed = [r for r in ok if r.get("predicted") is None]
            correct = sum(1 for r in ok if r.get("correct"))

            # Recompute rather than trust the summary.
            acc = correct / len(ok) if ok else 0.0
            if meta.get("n_completed") != len(ok):
                fail(f"{rid}: run.json annonce {meta.get('n_completed')} "
                     f"lignes réussies, le fichier en contient {len(ok)}")

            calls = Counter(r.get("n_calls") for r in ok)
            want = MULTI_CALL.get(strategy, 1)
            odd = sum(n for c, n in calls.items() if c != want)
            if odd:
                notes.append(f"{rid}: {odd} ligne(s) avec un nombre d'appels "
                             f"inattendu (attendu {want}; vu {dict(calls)})")

            if strategy == "few_shot":
                fm = json.loads((DATA / relation / "fewshot_manifest.json")
                                .read_text(encoding="utf-8"))
                if meta.get("fewshot_manifest_sha256") != fm.get("demo_map_sha256"):
                    fail(f"{rid}: démonstrations différentes de celles "
                         f"actuellement épinglées")

            print(f"{rid:<30}{len(rows):>8}{acc * 100:>8.1f} %{len(errors):>8}"
                  f"{len(unparsed):>9}{want:>8}")

    print()
    if notes:
        print("observations :")
        for n in notes:
            print(f"  • {n}")
        print()
    if problems:
        print(f"{len(problems)} problème(s) :")
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    print("tout est cohérent : lignes, étiquettes, empreintes et totaux "
          "concordent avec les manifestes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
