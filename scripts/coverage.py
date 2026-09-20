"""What has been run, and what the design still leaves open.

The status grid shows what exists; this says what does not, across every axis
the design varies -- strategy, adapter, transfer pair, seed -- so a gap is
noticed here rather than in a viva.

    python3 scripts/coverage.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
ADAPTERS = REPO / "adapters"

RELATIONS = ("topological", "cardinal", "relative")
STRATEGIES = ("zero_shot", "cot", "few_shot", "tot", "got")
# Four model calls a row against one, so these cost roughly four times as much.
COSTLY = {"tot", "got"}
# Not a gap to close: few-shot draws its demonstrations from train.csv, which
# is the pool the adapter was fine-tuned on. The model has memorised them, so
# the arm cannot be compared with the base model's.
EXCLUDED_WHEN_TUNED = {"few_shot"}


def finished(rel: str, strat: str, variant: str = "", seed: int = 1) -> bool:
    return (RESULTS / rel / strat / f"seed{seed}{variant}" / "run.json").exists()


def main() -> int:
    print("=" * 70)
    print("  COUVERTURE EXPÉRIMENTALE")
    print("=" * 70)

    # The four arms this design crosses: plain, adapter, knowledge store, and
    # the two together. Listing only the first two made the knowledge cells
    # invisible in the count, so a grid that was three cells into a new axis
    # read as complete.
    ARMS = (("BASE", ""),
            ("AFFINÉ (adaptateur de la famille)", "_lora"),
            ("BASE + BASE DE CONNAISSANCES", "_kg"),
            ("AFFINÉ + BASE DE CONNAISSANCES", "_lora_kg"))
    for title, variant in ARMS:
        skip = EXCLUDED_WHEN_TUNED if "_lora" in variant else set()
        applicable = [s for s in STRATEGIES if s not in skip]
        done = [(r, s) for r in RELATIONS for s in applicable
                if finished(r, s, variant)]
        total = len(RELATIONS) * len(applicable)
        print(f"\n  {title} — {len(done)}/{total}")
        for rel in RELATIONS:
            missing = [s for s in applicable if not finished(rel, s, variant)]
            mark = "complet" if not missing else "manque : " + ", ".join(missing)
            print(f"    {rel:<14}{mark}")
        for st in sorted(skip):
            missing = [r for r in RELATIONS if not finished(r, st, variant)]
            state = "complet" if not missing else f"manque : {', '.join(missing)}"
            print(f"    {'':<14}{st} — {state}")
            print(f"    {'':<14}  décision, pas oubli : les démonstrations "
                  f"viennent de train.csv,")
            print(f"    {'':<14}  le jeu sur lequel l'adaptateur a été entraîné. "
                  f"Le score serait")
            print(f"    {'':<14}  optimiste. ~35 min par cellule si vous le "
                  f"voulez quand même.")


    print("\n  AXES NON EXPLORÉS")
    seeds = sorted({p.name for p in RESULTS.glob("*/*/seed*") if p.is_dir()})
    other = {s for s in seeds if not s.startswith("seed1")}
    print(f"    graines        seed 1 seulement"
          if not other else f"    graines        {sorted(other)}")
    print("    époques        aucune ablation 1 / 2 / 3")
    kg_built = all((REPO / "data" / r / "kg_eval.json").exists() for r in RELATIONS)
    print(f"    base KG        {'construite et vérifiée' if kg_built else 'absente'}"
          f" ; mode 'input' implémenté, 'rag' non")

    # What it would cost to close the gaps, at the rates already observed.
    print("\n  COÛT DE CE QUI MANQUE")
    # Measured rates: a plain cell runs about 35 minutes at one call, four
    # times that at four. An adapter answers in five tokens where the base
    # model writes up to a thousand, so its cells take a couple of minutes.
    total = 0.0
    for title, variant in ARMS:
        skip = EXCLUDED_WHEN_TUNED if "_lora" in variant else set()
        cheap = [(r, s) for r in RELATIONS for s in STRATEGIES
                 if s not in COSTLY and s not in skip
                 and not finished(r, s, variant)]
        dear = [(r, s) for r in RELATIONS for s in STRATEGIES
                if s in COSTLY and not finished(r, s, variant)]
        if not cheap and not dear:
            continue
        per_cheap = 2 / 60 if "_lora" in variant else 35 / 60
        hours = len(cheap) * per_cheap + len(dear) * per_cheap * 4
        total += hours
        print(f"    {title:<36}{len(cheap) + len(dear):>2} cellule(s)"
              f"   ~{hours:>5.1f} h")
    if total:
        print(f"    {'':<36}{'':>2}            ~{total:>5.1f} h au total")
    else:
        print("    aucune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
