"""Cross-strategy comparison.

Builds the table the experiments exist to produce: every prompting strategy
against every spatial relation, with the uncertainty attached so a difference
can be judged rather than merely observed.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import LABELS, RELATIONS, RESULTS_DIR
from .metrics import (accuracy_excluding_unparsed, compute, holm_bonferroni,
                      load_predictions, mcnemar_exact, per_row_correct, wilson)
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

    # A checkpoint is written incrementally, so a cell still running looks
    # exactly like a finished small-n cell. Cells short of their relation's
    # full row count are the tell -- say so loudly, before any table is read.
    expected = {}
    for (rel, _s, _sd), c in cells.items():
        if c.get("n"):
            expected[rel] = max(expected.get(rel, 0), c["n"])
    partial = [(k, c["n"], expected[k[0]]) for k, c in cells.items()
               if c.get("n") and c["n"] < expected[k[0]]]
    if partial:
        out.append("  " + "!" * 80)
        out.append("  INCOMPLETE RESULTS — do not read the table below.")
        out.append(f"  {len(partial)} cell(s) have fewer rows than the relation's "
                   "full evaluation set, which means a run is still in progress "
                   "or died early:")
        for (rel, strat, seed), got, want in sorted(partial)[:8]:
            out.append(f"      {rel}/{strat}/seed{seed}: {got}/{want} rows")
        if len(partial) > 8:
            out.append(f"      ... and {len(partial) - 8} more")
        out.append("  Wait for the run to finish, then re-run evaluate and report.")
        out.append("  " + "!" * 80)
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


def _pooled_correct(rel: str, strat: str, seeds=None) -> dict[int, bool]:
    """row_index -> correct, pooling seeds by majority vote.

    Seeds are repeated measurements of the SAME rows, not extra rows. Treating
    each seed as an independent observation would multiply the effective sample
    size and manufacture significance, so they are collapsed to one verdict per
    row first.
    """
    from collections import defaultdict
    votes: dict[int, list[bool]] = defaultdict(list)
    base = RESULTS_DIR / rel / strat
    if not base.exists():
        return {}
    for seed_dir in sorted(base.glob("seed*")):
        if seeds:
            try:
                if int(seed_dir.name.replace("seed", "")) not in seeds:
                    continue
            except ValueError:
                continue
        for idx, ok in per_row_correct(
                load_predictions(seed_dir / "predictions.jsonl")).items():
            votes[idx].append(ok)
    return {i: (sum(v) * 2 > len(v)) for i, v in votes.items()}


def render_pairwise(relations=RELATIONS, seeds=None, alpha: float = 0.05) -> str:
    """Strategy-vs-strategy paired comparisons.

    Every strategy answers the same questions, so the comparison is paired and
    an exact McNemar test applies. That is far more sensitive than asking
    whether two confidence intervals overlap -- the resolution floor is the
    right caution for a single arm, but too conservative for a paired contrast.
    """
    out = ["=" * 90,
           "  PAIRWISE STRATEGY COMPARISON  (exact McNemar on shared rows, "
           "Holm-corrected)",
           "=" * 90,
           "  A>B / B>A counts rows where exactly one strategy was right.",
           "  Rows both got right, or both wrong, carry no information and drop out.",
           ""]
    strategies = available()
    for rel in relations:
        vectors = {s: _pooled_correct(rel, s, seeds) for s in strategies}
        vectors = {s: v for s, v in vectors.items() if v}
        if len(vectors) < 2:
            continue
        names = sorted(vectors)
        rows, pvals = [], []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared = sorted(set(vectors[a]) & set(vectors[b]))
                if not shared:
                    continue
                a_only = sum(1 for k in shared if vectors[a][k] and not vectors[b][k])
                b_only = sum(1 for k in shared if vectors[b][k] and not vectors[a][k])
                delta = 100.0 * (a_only - b_only) / len(shared)
                p = mcnemar_exact(a_only, b_only)
                rows.append([a, b, len(shared), a_only, b_only, delta, p])
                pvals.append(p)
        if not rows:
            continue
        for row, adj in zip(rows, holm_bonferroni(pvals)):
            row.append(adj)

        out.append(f"  {rel.upper()}   (n={rows[0][2]} shared rows)")
        out.append(f"    {'A':<11}{'B':<11}{'A>B':>5}{'B>A':>5}"
                   f"{'delta pp':>10}{'p':>9}{'p_holm':>9}   verdict")
        out.append("    " + "-" * 72)
        for a, b, n, ao, bo, d, p, adj in sorted(rows, key=lambda r: r[-1]):
            if adj < 0.001:
                mark = "*** "
            elif adj < 0.01:
                mark = "**  "
            elif adj < alpha:
                mark = "*   "
            else:
                mark = "n.s."
            better = a if d > 0 else b
            verdict = f"{mark} {better} better" if adj < alpha else f"{mark} indistinguishable"
            out.append(f"    {a:<11}{b:<11}{ao:>5}{bo:>5}{d:>+10.1f}"
                       f"{p:>9.4f}{adj:>9.4f}   {verdict}")
        n_sig = sum(1 for r in rows if r[-1] < alpha)
        out.append(f"    -> {n_sig}/{len(rows)} pairs separate after correction")
        out.append("")
    return "\n".join(out)


def render_parse_health(relations=RELATIONS) -> str:
    """Accuracy with and without unparseable completions.

    An unparseable answer is scored wrong, which mixes 'reasoned badly' with
    'answered in the wrong format'. Showing both figures says which one a
    strategy is actually suffering from.
    """
    out = ["=" * 90,
           "  PARSE HEALTH  —  does a strategy lose accuracy to formatting?",
           "=" * 90,
           f"    {'relation':<13}{'strategy':<12}{'unparsed':>9}"
           f"{'acc (all)':>11}{'acc (parsed)':>14}{'gap':>7}", ""]
    for rel in relations:
        for strat in available():
            base = RESULTS_DIR / rel / strat
            if not base.exists():
                continue
            recs = []
            for seed_dir in sorted(base.glob("seed*")):
                recs.extend(load_predictions(seed_dir / "predictions.jsonl"))
            if not recs:
                continue
            m = compute(recs, LABELS[rel])
            pu = accuracy_excluding_unparsed(recs)
            if not pu.get("n_parsed"):
                continue
            gap = pu["accuracy_parsed_only"] - m["accuracy"]
            flag = "  <-- formatting, not reasoning" if gap >= 3.0 else ""
            out.append(f"    {rel:<13}{strat:<12}{pu['n_unparsed']:>9}"
                       f"{m['accuracy']:>11.1f}{pu['accuracy_parsed_only']:>14.1f}"
                       f"{gap:>+7.1f}{flag}")
        out.append("")
    return "\n".join(out)
