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

    for title, variant in (("BASE", ""), ("AFFINÉ (adaptateur de la famille)", "_lora")):
        skip = EXCLUDED_WHEN_TUNED if variant else set()
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
    print("    graphe KG      non commencé")

    # What it would cost to close the gaps, at the rates already observed.
    print("\n  COÛT DES CELLULES AFFINÉES MANQUANTES")
    cheap = [(r, s) for r in RELATIONS for s in STRATEGIES
             if s not in COSTLY and s not in EXCLUDED_WHEN_TUNED
             and not finished(r, s, "_lora")]
    dear = [(r, s) for r in RELATIONS for s in STRATEGIES
            if s in COSTLY and not finished(r, s, "_lora")]
    if not cheap and not dear:
        print("    aucune")
        return 0
    if cheap:
        print(f"    {len(cheap)} à un appel       ~{len(cheap) * 35} min   "
              f"{', '.join(f'{r}/{s}' for r, s in cheap)}")
    if dear:
        print(f"    {len(dear)} à quatre appels  ~{len(dear) * 4.5:.0f} h   "
              f"{', '.join(f'{r}/{s}' for r, s in dear)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
