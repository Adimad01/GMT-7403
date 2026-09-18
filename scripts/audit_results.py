"""Check the finished results for the things a percentage cannot show.

An accuracy figure hides most ways a run can be wrong: rows silently missing,
answers that never parsed and were scored as failures, a model that reached
its score by predicting one label for everything, or two arms that are not
comparable because they were scored against different data. This looks for
each of those, then runs the paired test the design actually allows.

    python3 scripts/audit_results.py
    python3 scripts/audit_results.py --relation relative --verbose
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
DATA = REPO / "data"
RELATIONS = ("topological", "cardinal", "relative")
STRATEGIES = ("zero_shot", "cot", "few_shot", "tot", "got")

# A strategy that issues several model calls per row should show it. A tree or
# graph arm reporting one call is not doing what its name claims.
EXPECTED_CALLS = {"zero_shot": 1, "cot": 1, "few_shot": 1, "tot": 4, "got": 4}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half) * 100, (centre + half) * 100


def chi2_sf_1df(x: float) -> float:
    """Upper tail of chi-square with one degree of freedom."""
    return math.erfc(math.sqrt(x / 2))


def mcnemar(a: dict[int, bool], b: dict[int, bool]) -> tuple[int, int, float]:
    """Paired comparison of two arms over the rows they share.

    The arms are evaluated on identical rows, so an unpaired test throws away
    that pairing and loses power. Only the rows where they disagree carry any
    information about which is better.
    """
    shared = a.keys() & b.keys()
    only_a = sum(1 for i in shared if a[i] and not b[i])
    only_b = sum(1 for i in shared if b[i] and not a[i])
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    stat = (abs(only_a - only_b) - 1) ** 2 / n     # continuity-corrected
    return only_a, only_b, chi2_sf_1df(stat)


def load(relation: str, strategy: str) -> tuple[list[dict], dict | None]:
    d = RESULTS / relation / strategy / "seed1"
    preds = d / "predictions.jsonl"
    if not preds.exists():
        return [], None
    rows = []
    for line in preds.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    meta = None
    if (d / "run.json").exists():
        try:
            meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
        except Exception:
            pass
    return rows, meta


def manifest(relation: str) -> dict | None:
    try:
        return json.loads((DATA / relation / "eval_manifest.json")
                          .read_text(encoding="utf-8"))
    except Exception:
        return None


def demo_manifest_hash(relation: str) -> str | None:
    try:
        return json.loads((DATA / relation / "fewshot_manifest.json")
                          .read_text(encoding="utf-8")).get("demo_map_sha256")
    except Exception:
        return None


def expected_rows(relation: str) -> int | None:
    man = manifest(relation)
    return len(man["rows"]) if man else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", choices=RELATIONS)
    ap.add_argument("--exclude-level", nargs="+", metavar="N", default=["6"],
                    help="ambiguity levels to leave out of the analysis "
                         "(default: 6). Level 6 varies composition, not "
                         "linguistic indirection, and its wording is "
                         "deliberately plain -- so it moves two variables at "
                         "once and does not belong on the same scale. Pass "
                         "--exclude-level none to include everything.")
    ap.add_argument("--verbose", action="store_true",
                    help="per-level accuracy and the confusion detail")
    args = ap.parse_args()
    relations = [args.relation] if args.relation else list(RELATIONS)
    dropped = set() if args.exclude_level == ["none"] else {
        f"Level {n}" for n in args.exclude_level}
    if dropped:
        print(f"  (niveaux exclus de l'analyse : "
              f"{', '.join(sorted(dropped))} — lignes conservées sur disque)\n")

    problems: list[str] = []
    cells: dict[tuple[str, str], dict] = {}

    # ---- integrity ----------------------------------------------------
    print("=" * 74)
    print("  INTÉGRITÉ")
    print("=" * 74)
    for rel in relations:
        man = manifest(rel)
        want = len(man["rows"]) if man else None
        # The pinned label for each row. Comparing cells only against each
        # other would pass happily if every one of them had been scored
        # against the same wrong data.
        gold_of = {r["row_index"]: r["label"] for r in man["rows"]} if man else {}
        disk_hash = man.get("manifest_sha256") if man else None
        disk_demo = demo_manifest_hash(rel)
        hashes, demo_hashes = set(), set()
        for strat in STRATEGIES:
            rows, meta = load(rel, strat)
            # run.json counts every row the cell ran, so its total must be
            # checked before any level is filtered out of the analysis.
            n_ok_all = sum(1 for r in rows if r.get("status") == "ok")
            if dropped:
                # The rows stay on disk; only the analysis leaves them out, so
                # the decision is reversible with a flag rather than a rerun.
                rows = [r for r in rows
                        if r.get("ambiguity_level") not in dropped]
            if not rows:
                problems.append(f"{rel}/{strat}: aucun résultat")
                continue
            idx = [r.get("row_index") for r in rows]
            dupes = [i for i, n in Counter(idx).items() if n > 1]
            # Rows excluded above are absent by choice, so completeness is
            # measured against what the manifest holds at the kept levels.
            if dropped and man:
                keep = {e["row_index"] for e in man["rows"]
                        if e.get("ambiguity_level") not in dropped}
                missing = keep - set(idx)
            else:
                missing = (set(range(want)) - set(idx)) if want else set()
            failed = [r for r in rows if r.get("status") != "ok"]
            # A scored-but-unparsed answer counts as wrong. If that is
            # common the number being reported is partly a parser score.
            unparsed = [r for r in rows
                        if r.get("status") == "ok" and r.get("predicted") is None]
            calls = [r.get("n_calls") for r in rows
                     if isinstance(r.get("n_calls"), int)]
            med_calls = sorted(calls)[len(calls) // 2] if calls else None

            if meta:
                if meta.get("eval_manifest_sha256"):
                    hashes.add(meta["eval_manifest_sha256"])
                if strat == "few_shot" and meta.get("fewshot_manifest_sha256"):
                    demo_hashes.add(meta["fewshot_manifest_sha256"])

            ok = [r for r in rows if r.get("status") == "ok"]
            k = sum(1 for r in ok if r.get("correct"))
            cells[(rel, strat)] = {
                "rows": rows, "ok": ok, "k": k, "n": len(ok),
                "correct_by_row": {r["row_index"]: bool(r.get("correct"))
                                   for r in ok},
            }

            flags = []
            if dupes:
                flags.append(f"{len(dupes)} doublons")
                problems.append(f"{rel}/{strat}: {len(dupes)} row_index en double")
            if missing:
                flags.append(f"{len(missing)} manquantes")
                problems.append(f"{rel}/{strat}: {len(missing)} lignes absentes")
            if failed:
                flags.append(f"{len(failed)} erreurs")
            if unparsed:
                flags.append(f"{len(unparsed)} non analysées")
                if len(unparsed) / max(len(rows), 1) > 0.02:
                    problems.append(
                        f"{rel}/{strat}: {len(unparsed)} réponses non analysées "
                        f"({len(unparsed)/len(rows)*100:.1f} %) comptées comme fausses")
            # Scored against the labels the manifest pins, not whatever the
            # CSV happened to hold when the cell ran.
            bad_gold = [r["row_index"] for r in rows
                        if r.get("row_index") in gold_of
                        and r.get("gold") != gold_of[r["row_index"]]]
            if bad_gold:
                flags.append(f"{len(bad_gold)} étiquettes divergentes")
                problems.append(
                    f"{rel}/{strat}: {len(bad_gold)} ligne(s) notée(s) contre une "
                    f"étiquette absente du manifeste (ex. index {bad_gold[:3]})")
            if meta and disk_hash and meta.get("eval_manifest_sha256") \
                    and meta["eval_manifest_sha256"] != disk_hash:
                flags.append("manifeste obsolète")
                problems.append(
                    f"{rel}/{strat}: produit sous un manifeste qui n'est plus "
                    f"celui du dépôt — les lignes évaluées ne sont plus celles-là")
            if strat == "few_shot" and meta and disk_demo \
                    and meta.get("fewshot_manifest_sha256") \
                    and meta["fewshot_manifest_sha256"] != disk_demo:
                flags.append("démonstrations obsolètes")
                problems.append(
                    f"{rel}/{strat}: démonstrations différentes de celles "
                    f"actuellement épinglées")
            # Recompute rather than trust the summary that reports itself.
            if meta and meta.get("n_completed") is not None \
                    and meta["n_completed"] != n_ok_all:
                flags.append("total incohérent")
                problems.append(
                    f"{rel}/{strat}: run.json annonce {meta['n_completed']} "
                    f"lignes réussies, le fichier en contient {n_ok_all}")
            if med_calls is not None and med_calls != EXPECTED_CALLS[strat]:
                flags.append(f"{med_calls} appels au lieu de {EXPECTED_CALLS[strat]}")
                problems.append(f"{rel}/{strat}: médiane {med_calls} appels, "
                                f"{EXPECTED_CALLS[strat]} attendus")
            print(f"  {rel:<12}{strat:<11}{len(rows):>4} lignes"
                  + (f"   ⚠ {', '.join(flags)}" if flags else "   ok"))

        if len(hashes) > 1:
            problems.append(f"{rel}: les cellules n'ont pas le même manifeste "
                            f"d'évaluation ({len(hashes)} empreintes) — non comparables")
        if len(demo_hashes) > 1:
            problems.append(f"{rel}: manifestes few-shot divergents")

    # ---- scores -------------------------------------------------------
    print()
    print("=" * 74)
    print("  EXACTITUDE   (IC de Wilson à 95 %)")
    print("=" * 74)
    for rel in relations:
        for strat in STRATEGIES:
            c = cells.get((rel, strat))
            if not c or not c["n"]:
                continue
            lo, hi = wilson(c["k"], c["n"])
            print(f"  {rel:<12}{strat:<11}{c['k'] / c['n'] * 100:>6.1f} %"
                  f"   [{lo:>5.1f} – {hi:>5.1f}]   n={c['n']}")
        print()

    # ---- is the model actually discriminating? ------------------------
    print("=" * 74)
    print("  RÉPARTITION DES PRÉDICTIONS")
    print("=" * 74)
    print("  Une exactitude peut venir d'un modèle qui répond presque toujours")
    print("  la même chose, si cette réponse domine la vérité terrain.\n")
    for rel in relations:
        for strat in STRATEGIES:
            c = cells.get((rel, strat))
            if not c or not c["n"]:
                continue
            # An unparsed answer is None, not a label. Counting it among the
            # labels used made cells look as though the model had invented a
            # category -- every cell with an unparsed row reported one more
            # label than exists, and only the two cells with none read right.
            pred = Counter(r.get("predicted") for r in c["ok"]
                           if r.get("predicted") is not None)
            gold = Counter(r.get("gold") for r in c["ok"])
            if not pred:
                print(f"  {rel:<12}{strat:<11} aucune réponse exploitable")
                continue
            top, n_top = pred.most_common(1)[0]
            share = n_top / c["n"]
            gold_share = gold[top] / c["n"] if top in gold else 0.0
            flag = ""
            # Predicting one label far more often than it occurs is the
            # signature of an arm that has collapsed onto a default.
            if share > 0.45 and share > gold_share * 1.6:
                flag = "   ⚠ effondrement sur une étiquette"
                problems.append(
                    f"{rel}/{strat}: prédit « {top} » sur {share*100:.0f} % des "
                    f"lignes alors qu'elle en couvre {gold_share*100:.0f} %")
            print(f"  {rel:<12}{strat:<11}{len(pred):>2} étiquettes utilisées / "
                  f"{len(gold)} présentes   dominante « {top} » "
                  f"{share*100:.0f} % (terrain {gold_share*100:.0f} %){flag}")
        print()

    # ---- the ambiguity ladder ----------------------------------------
    if args.verbose:
        print("=" * 74)
        print("  EXACTITUDE PAR NIVEAU")
        print("=" * 74)
        for rel in relations:
            for strat in STRATEGIES:
                c = cells.get((rel, strat))
                if not c or not c["n"]:
                    continue
                per = defaultdict(lambda: [0, 0])
                for r in c["ok"]:
                    lv = r.get("ambiguity_level", "?")
                    per[lv][1] += 1
                    per[lv][0] += bool(r.get("correct"))
                line = "  ".join(
                    f"{lv.replace('Level ', 'N')}:{k/n*100:>5.1f}%"
                    for lv, (k, n) in sorted(per.items()) if n)
                print(f"  {rel:<12}{strat:<11}{line}")
            print()

        print("=" * 74)
        print("  CONFUSIONS LES PLUS FRÉQUENTES")
        print("=" * 74)
        for rel in relations:
            conf = Counter()
            for strat in STRATEGIES:
                c = cells.get((rel, strat))
                if not c:
                    continue
                for r in c["ok"]:
                    if not r.get("correct"):
                        conf[(r.get("gold"), r.get("predicted"))] += 1
            print(f"  {rel}")
            for (g, p), n in conf.most_common(6):
                print(f"      {str(g):<14} pris pour {str(p):<14}{n:>5}")
            print()

    # ---- paired comparisons ------------------------------------------
    print("=" * 74)
    print("  COMPARAISONS APPARIÉES   (McNemar, mêmes lignes)")
    print("=" * 74)

    # Every strategy is compared against zero-shot in every relation, so this
    # is a family of tests, not one. Reading each at p<0.05 would expect about
    # one false positive per run by construction. Holm holds the chance of any
    # false positive across the family at 5 percent, and it is what decides
    # the verdict printed here -- the raw p is shown beside it, not instead.
    tests = []
    for rel in relations:
        base = cells.get((rel, "zero_shot"))
        if not base:
            continue
        for strat in STRATEGIES:
            if strat == "zero_shot":
                continue
            c = cells.get((rel, strat))
            if not c:
                continue
            gained, lost, pv = mcnemar(c["correct_by_row"], base["correct_by_row"])
            tests.append({"rel": rel, "strat": strat, "gained": gained,
                          "lost": lost, "p": pv})

    m = len(tests)
    holm_ok = {}
    still_below = True
    for i, t in enumerate(sorted(tests, key=lambda x: x["p"])):
        thr = 0.05 / (m - i)
        if still_below and t["p"] > thr:
            still_below = False          # Holm stops at the first failure
        holm_ok[(t["rel"], t["strat"])] = still_below and t["p"] <= thr

    last = None
    for t in tests:
        if last and t["rel"] != last:
            print()
        last = t["rel"]
        survives = holm_ok[(t["rel"], t["strat"])]
        if survives:
            verdict = "significatif"
        elif t["p"] < 0.05:
            verdict = "tendance (ne survit pas à Holm)"
        elif t["p"] >= 1:
            verdict = "identiques"
        else:
            verdict = "non significatif"
        sign = "+" if t["gained"] > t["lost"] else "−" if t["lost"] > t["gained"] else "="
        print(f"  {t['rel']:<12}{t['strat']:<11}vs zero_shot   "
              f"{sign}{abs(t['gained'] - t['lost']):>3} lignes nettes   "
              f"(gagne {t['gained']}, perd {t['lost']})   "
              f"p={t['p']:.3f}  {verdict}")
    print()
    n_raw = sum(1 for t in tests if t["p"] < 0.05)
    n_holm = sum(holm_ok.values())
    print(f"  {m} comparaisons. {n_raw} sous p<0.05 brut, {n_holm} après "
          f"correction de Holm.")
    if n_raw and not n_holm:
        print("  Aucun effet de stratégie n'est établi. Les tendances "
              "ci-dessus demandent")
        print("  d'être posées en hypothèse avant mesure, sur une seconde graine.")
    print()

    # ---- verdict ------------------------------------------------------
    print("=" * 74)
    if problems:
        print(f"  {len(problems)} POINT(S) À REGARDER")
        print("=" * 74)
        for p in problems:
            print(f"  • {p}")
    else:
        print("  RIEN D'ANORMAL DÉTECTÉ")
        print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
