"""Cross-strategy comparison.

Builds the table the experiments exist to produce: every prompting strategy
against every spatial relation, with the uncertainty attached so a difference
can be judged rather than merely observed.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import LABELS, RELATIONS, RESULTS_DIR
from .metrics import compute, load_predictions, wilson
from .strategies import available


def resolution_floor(n: int) -> float:
    """Half-width of the widest interval this n can produce, in points.

    The practical precision limit of an evaluation set: a difference smaller
    than this cannot be resolved no matter which strategies are compared.
    """
    lo, hi = wilson(n // 2, n)
    return (hi - lo) / 2.0


def collect(relations=RELATIONS, strategies=None, seeds=None) -> dict:
    strategies = strategies or available()
    cells: dict[tuple[str, str, int], dict] = {}
    for rel in relations:
        for strat in strategies:
            base = RESULTS_DIR / rel / strat
            if not base.exists():
                continue
            for seed_dir in sorted(base.glob("seed*")):
                try:
                    seed = int(seed_dir.name.replace("seed", ""))
                except ValueError:
                    continue
                if seeds and seed not in seeds:
                    continue
                recs = load_predictions(seed_dir / "predictions.jsonl")
                if not recs:
                    continue
                cells[(rel, strat, seed)] = compute(recs, LABELS[rel])
    return cells


def _agg(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    m = sum(values) / len(values)
    if len(values) == 1:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return (m, var ** 0.5)


def render(cells: dict, metric: str = "accuracy") -> str:
    if not cells:
        return ("No results found under results/.\n"
                "Run an experiment first:  python3 -m spatial_eval.cli run --all\n")

    strategies = sorted({k[1] for k in cells})
    relations = [r for r in RELATIONS if any(k[0] == r for k in cells)]
    out: list[str] = []

    out.append("=" * 84)
    out.append(f"  STRATEGY COMPARISON  —  {metric}")
    out.append("=" * 84)

    label = {"accuracy": "accuracy (all rows)",
             "accuracy_by_fact": "accuracy (unique facts — independent observations)",
             "macro_f1": "macro-F1"}.get(metric, metric)
    out.append(f"  metric: {label}")
    out.append("  cells show mean across seeds ± sd, with the 95% interval of the "
               "pooled estimate")
    out.append("")

    head = f"  {'relation':<14}" + "".join(f"{s:>17}" for s in strategies)
    out.append(head)
    out.append("  " + "-" * (len(head) - 2))

    for rel in relations:
        ns = [c["n"] for k, c in cells.items() if k[0] == rel and c.get("n")]
        n = max(ns) if ns else 0
        row = f"  {rel:<14}"
        for strat in strategies:
            vals = [c[metric] for k, c in cells.items()
                    if k[0] == rel and k[1] == strat and metric in c]
            if not vals:
                row += f"{'—':>17}"
                continue
            m, sd = _agg(vals)
            cell = f"{m:.1f}" + (f"±{sd:.1f}" if len(vals) > 1 else "")
            row += f"{cell:>17}"
        out.append(row)
        floor = resolution_floor(n) if n else 0
        out.append(f"  {'':<14}" + f"  n={n}, resolution floor ±{floor:.1f}pp "
                                   f"— differences smaller than this are not resolvable")
    out.append("")

    # Seed coverage, so a single-seed result is never mistaken for a stable one.
    out.append("  " + "-" * 80)
    seeds_by = {}
    for (rel, strat, seed) in cells:
        seeds_by.setdefault((rel, strat), []).append(seed)
    single = [f"{r}/{s}" for (r, s), sd in sorted(seeds_by.items()) if len(sd) == 1]
    if single:
        out.append(f"  ⚠ single seed (run-to-run variance unmeasured): {', '.join(single)}")
    unparsed = [(k, c["unparsed"]) for k, c in cells.items() if c.get("unparsed")]
    if unparsed:
        out.append("  ⚠ unparseable completions: "
                   + ", ".join(f"{k[0]}/{k[1]}/s{k[2]}={v}" for k, v in sorted(unparsed)))
    failed = [(k, len(c["failed_rows"])) for k, c in cells.items() if c.get("failed_rows")]
    if failed:
        out.append("  ⚠ failed rows (rerun with `python3 -m spatial_eval.cli run --all`): "
                   + ", ".join(f"{k[0]}/{k[1]}/s{k[2]}={v}" for k, v in sorted(failed)))
    out.append("")
    return "\n".join(out)


def render_per_label(cells: dict) -> str:
    out = ["=" * 84, "  PER-LABEL F1  (each spatial relation, each strategy)", "=" * 84]
    strategies = sorted({k[1] for k in cells})
    for rel in [r for r in RELATIONS if any(k[0] == r for k in cells)]:
        out.append(f"\n  {rel.upper()}")
        head = f"  {'label':<16}" + "".join(f"{s:>12}" for s in strategies)
        out.append(head)
        out.append("  " + "-" * (len(head) - 2))
        for lab in LABELS[rel]:
            row = f"  {lab:<16}"
            for strat in strategies:
                vals = [c["per_label"][lab]["f1"]
                        for k, c in cells.items()
                        if k[0] == rel and k[1] == strat and "per_label" in c]
                row += f"{(sum(vals)/len(vals)):>12.3f}" if vals else f"{'—':>12}"
            out.append(row)
    return "\n".join(out) + "\n"


def write_csv(cells: dict, path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relation", "strategy", "seed", "n", "n_unique_facts",
                    "accuracy", "ci_low", "ci_high", "accuracy_by_fact",
                    "macro_f1", "unparsed", "failed"])
        for (rel, strat, seed), c in sorted(cells.items()):
            if not c.get("n"):
                continue
            w.writerow([rel, strat, seed, c["n"], c["n_unique_facts"],
                        c["accuracy"], c["accuracy_ci95"][0], c["accuracy_ci95"][1],
                        c["accuracy_by_fact"], c["macro_f1"], c["unparsed"],
                        len(c["failed_rows"])])


def write_json(cells: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"{r}/{s}/seed{sd}": c for (r, s, sd), c in sorted(cells.items())}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
