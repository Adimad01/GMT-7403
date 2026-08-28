"""
stats_analysis.py
================================================================================
Statistical analysis for the 7-experiment x 3-strategy KG-integration ablation.

The previous analyzer printed a bare accuracy matrix. At these sample sizes
(Topological 105, Cardinal 40, Relative 25 eval rows) a bare percentage invites
over-reading: a single flipped prediction moves Relative accuracy by 4 points,
so most arm-to-arm gaps are indistinguishable from sampling noise.

This module reports what can actually be concluded:

  * Wilson 95% confidence intervals on every accuracy (better than the normal
    approximation at small n and near 0/100%).
  * Run-to-run variance across seeds, since generation uses do_sample=True.
  * Exact McNemar tests against a baseline arm. Every experiment scores the
    SAME eval rows, so comparisons are PAIRED -- a paired test is far more
    powerful here than comparing two independent proportions.
  * Holm-Bonferroni correction, because the design runs many comparisons and
    uncorrected p-values would manufacture significance.
  * A resolution floor per domain: the tightest CI the eval set can produce.
    If a claimed effect is smaller than this, the experiment cannot see it.

Dependencies: numpy + stdlib only (no scipy/statsmodels) so it runs unchanged
on the offline GPU server.

Usage
  python stats_analysis.py                          # all domains
  python stats_analysis.py --domain Topological-Reasoning
  python stats_analysis.py --baseline exp1_base --alpha 0.05
  python stats_analysis.py --by-level               # add per-ambiguity-level breakdown
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
Z95 = 1.959963984540054          # standard normal quantile for a 95% interval

DOMAINS = ["Topological-Reasoning", "Cardinal-Reasoning", "Relative-Reasoning"]
STRATS = ["cot", "tot", "got"]

EXP_LABELS = {
    "exp1_base":          "Exp1  base / no KG",
    "exp2_ft_nokg":       "Exp2  no-KG LoRA",
    "exp3_ft_osmkg":      "Exp3  OSM-KG LoRA (KG@train)",
    "exp4_base_kg_input": "Exp4  base + KG@input",
    "exp5_ft_kg_input":   "Exp5  no-KG LoRA + KG@input",
    "exp6_base_kg_rag":   "Exp6  base + KG@inference (RAG)",
    "exp7_base_graphrag": "Exp7  base + GraphRAG sub-graph",
}
EXP_ORDER = list(EXP_LABELS)

# voletc_<exp>[_fs<k>][_s<seed>]_<strategy>_<suffix>_ckpt.json
_CKPT_RE = re.compile(
    r"voletc_(?P<exp>.+?)"
    r"(?:_fs(?P<shots>\d+))?"
    r"(?:_s(?P<seed>\d+))?"
    r"_(?P<strat>cot|tot|got)"
    r"_.*_ckpt\.json$"
)


# ---------------------------------------------------------------------------
# STATISTICS  (numpy + stdlib only)
# ---------------------------------------------------------------------------
def wilson_interval(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as percentages.

    Preferred over the normal ("Wald") approximation, which produces intervals
    that fall outside [0, 1] and undercover badly when n is small or p is near
    an extreme -- both true of these eval sets.
    """
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half) * 100.0, min(1.0, centre + half) * 100.0)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes.

    b = rows the baseline got right and the variant got wrong
    c = rows the baseline got wrong and the variant got right

    Concordant pairs carry no information about which arm is better and drop
    out. Under H0 each discordant pair is a fair coin, so the count follows
    Binomial(b + c, 0.5). Computed exactly -- the chi-square approximation is
    unreliable when discordant pairs are few, which is the norm here.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values.

    Controls the family-wise error rate across the many comparisons this design
    generates, while being uniformly more powerful than plain Bonferroni.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)          # enforce monotonicity
        adj[idx] = min(1.0, running)
    return adj


def bootstrap_delta_ci(base: np.ndarray, variant: np.ndarray,
                       n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the paired accuracy difference (pp).

    Resamples ROWS (not predictions independently) so the pairing between the
    two arms is preserved -- that pairing is what makes the comparison sensitive.
    """
    n = len(base)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = (variant[idx].mean(axis=1) - base[idx].mean(axis=1)) * 100.0
    return (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def resolution_floor(n: int) -> float:
    """Half-width of the widest Wilson interval this n can produce (pp).

    The interval is widest at p = 0.5, so this is the best-case precision of the
    eval set. Differences smaller than roughly this cannot be resolved no matter
    which arms are compared.
    """
    lo, hi = wilson_interval(n // 2, n)
    return (hi - lo) / 2.0


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------
def parse_ckpt_name(fname: str) -> dict | None:
    m = _CKPT_RE.search(os.path.basename(fname))
    if not m:
        return None
    return {
        "exp":   m.group("exp"),
        "shots": int(m.group("shots") or 0),
        "seed":  int(m.group("seed") or 0),
        "strat": m.group("strat"),
    }


def load_runs(results_dir: str) -> list[dict]:
    """Load every checkpoint in a results dir into per-row correctness vectors."""
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_ckpt.json"))):
        meta = parse_ckpt_name(path)
        if not meta:
            continue
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        rows = data.get("results", [])
        if not rows:
            continue
        by_index = {r["index"]: bool(r.get("match")) for r in rows if "index" in r}
        meta.update({
            "path": path,
            "by_index": by_index,
            "n": len(by_index),
            "hits": sum(by_index.values()),
        })
        runs.append(meta)
    return runs


def load_eval_metadata(domain: str) -> dict[int, dict]:
    """Map eval row index -> {level, relation} for stratified breakdowns."""
    candidates = [
        (f"{domain}/dataset/topo_v2_eval.csv", "spatial_relation"),
        (f"{domain}/dataset/cardinal_direction_relations.csv", "relation_label"),
        (f"{domain}/dataset/relative_direction_relations.csv", "relation_label"),
    ]
    for path, label_col in candidates:
        if not os.path.exists(path):
            continue
        out = {}
        with open(path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if label_col not in row:
                    break
                out[i] = {"level": row.get("ambiguity_level", "?"),
                          "relation": row[label_col]}
        if out:
            return out
    return {}


def pool_seeds(runs: list[dict]) -> dict:
    """Group runs by (exp, strat, shots), keeping each seed separate."""
    grouped = defaultdict(list)
    for r in runs:
        grouped[(r["exp"], r["strat"], r["shots"])].append(r)
    return grouped


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------
def report_domain(domain: str, baseline: str, alpha: float,
                  by_level: bool, n_boot: int) -> None:
    results_dir = os.path.join(domain, "code", "results")
    runs = load_runs(results_dir)

    print("\n" + "=" * 78)
    print(f"  {domain}")
    print("=" * 78)

    if not runs:
        print("  No result checkpoints found — nothing to analyse yet.")
        return

    grouped = pool_seeds(runs)
    n_rows = max(r["n"] for r in runs)
    floor = resolution_floor(n_rows)
    seeds = sorted({r["seed"] for r in runs})

    print(f"  eval rows: {n_rows}    seeds present: {seeds}    "
          f"conditions: {len(grouped)}")

    # A checkpoint is written incrementally, so a run still in progress looks
    # exactly like a completed small-n run. Uneven row counts across conditions
    # are the tell -- surface them instead of silently reporting partial data.
    row_counts = sorted({r["n"] for r in runs})
    if len(row_counts) > 1:
        short = [f'{r["exp"]}/{r["strat"]}/s{r["seed"]}={r["n"]}'
                 for r in runs if r["n"] < n_rows]
        print(f"  ⚠ INCOMPLETE: row counts differ across runs {row_counts}. "
              f"Some runs are still in flight or died early:")
        for chunk in (short[i:i + 3] for i in range(0, min(len(short), 9), 3)):
            print("      " + "  ".join(chunk))
        if len(short) > 9:
            print(f"      ... and {len(short) - 9} more")
        print("    Numbers below mix complete and partial runs — do not report them.")
    print(f"  resolution floor: +/-{floor:.1f}pp  "
          f"(widest 95% CI half-width at n={n_rows})")
    if len(seeds) == 1:
        print("  ⚠ single seed — run-to-run variance is UNMEASURED. "
              "Re-run with --seed 1, 2, ... to quantify it.")

    # ---- accuracy table -----------------------------------------------------
    print(f"\n  ACCURACY  (% with Wilson 95% CI; +/-sd across seeds where >1)")
    print(f"  {'condition':<34}" + "".join(f"{s.upper():>21}" for s in STRATS))
    print("  " + "-" * (34 + 21 * len(STRATS)))

    exps = [e for e in EXP_ORDER if any(k[0] == e for k in grouped)]
    exps += sorted({k[0] for k in grouped} - set(EXP_ORDER))

    for shots in sorted({k[2] for k in grouped}):
        tag = "zero-shot" if shots == 0 else f"few-shot({shots})"
        print(f"\n  [{tag}]")
        for exp in exps:
            label = EXP_LABELS.get(exp, exp)
            line = f"  {label:<34}"
            for strat in STRATS:
                rs = grouped.get((exp, strat, shots), [])
                if not rs:
                    line += f"{'—':>21}"
                    continue
                accs = [r["hits"] / r["n"] * 100.0 for r in rs]
                hits = sum(r["hits"] for r in rs)
                n = sum(r["n"] for r in rs)
                lo, hi = wilson_interval(hits, n)
                cell = f"{np.mean(accs):.1f} [{lo:.0f}-{hi:.0f}]"
                if len(accs) > 1:
                    cell += f"±{np.std(accs, ddof=1):.1f}"
                line += f"{cell:>21}"
            print(line)

    # ---- paired tests vs baseline ------------------------------------------
    print(f"\n  PAIRED COMPARISONS vs {EXP_LABELS.get(baseline, baseline)}")
    print("  (exact McNemar on shared eval rows; Holm-corrected across this family)")

    comparisons = []
    for (exp, strat, shots), rs in sorted(grouped.items()):
        if exp == baseline:
            continue
        base_rs = grouped.get((baseline, strat, shots))
        if not base_rs:
            continue
        # Pair on the eval indices both arms actually scored.
        shared = sorted(set.intersection(
            *[set(r["by_index"]) for r in rs + base_rs]))
        if not shared:
            continue

        # Collapse seeds to ONE value per eval row before testing.
        #
        # Seeds are repeated measurements of the same 105 rows, not extra rows.
        # Crossing every baseline seed with every variant seed would replicate
        # each row seeds^2 times and inflate the effective sample size, which
        # shrinks p-values toward zero and manufactures significance. Averaging
        # per row keeps n at the true number of eval rows and uses the extra
        # seeds to reduce sampling noise, which is what they can legitimately do.
        b_arr = np.array([np.mean([r["by_index"][i] for r in base_rs])
                          for i in shared], dtype=float)
        v_arr = np.array([np.mean([r["by_index"][i] for r in rs])
                          for i in shared], dtype=float)
        # McNemar needs a binary outcome per row: take the majority verdict
        # across seeds. Rows where the seeds split evenly are dropped as
        # genuinely undecided rather than broken toward either arm.
        b_bin = np.where(b_arr > 0.5, 1, np.where(b_arr < 0.5, 0, -1))
        v_bin = np.where(v_arr > 0.5, 1, np.where(v_arr < 0.5, 0, -1))
        decided = (b_bin >= 0) & (v_bin >= 0)
        b_only = int(np.sum(decided & (b_bin == 1) & (v_bin == 0)))
        v_only = int(np.sum(decided & (b_bin == 0) & (v_bin == 1)))
        delta = (v_arr.mean() - b_arr.mean()) * 100.0
        p = mcnemar_exact(b_only, v_only)
        lo, hi = bootstrap_delta_ci(b_arr, v_arr, n_boot=n_boot)
        comparisons.append({
            "exp": exp, "strat": strat, "shots": shots, "delta": delta,
            "b_only": b_only, "v_only": v_only, "p": p, "lo": lo, "hi": hi,
        })

    if not comparisons:
        print("    (baseline arm absent — cannot compare)")
        return

    adj = holm_bonferroni([c["p"] for c in comparisons])
    for c, pa in zip(comparisons, adj):
        c["p_adj"] = pa

    print(f"\n  {'condition':<34}{'strat':<6}{'shots':<7}"
          f"{'delta pp':>10}{'95% CI':>16}{'discord':>10}{'p':>9}{'p_holm':>9}  sig")
    print("  " + "-" * 106)
    for c in sorted(comparisons, key=lambda x: (x["shots"], x["exp"], x["strat"])):
        sig = "***" if c["p_adj"] < 0.001 else \
              "**" if c["p_adj"] < 0.01 else \
              "*" if c["p_adj"] < alpha else "n.s."
        label = EXP_LABELS.get(c["exp"], c["exp"])
        ci_str = f"[{c['lo']:+.1f},{c['hi']:+.1f}]"
        discord = f"{c['b_only']}/{c['v_only']}"
        print(f"  {label:<34}{c['strat']:<6}{c['shots']:<7}"
              f"{c['delta']:>+10.1f}{ci_str:>16}{discord:>10}"
              f"{c['p']:>9.3f}{c['p_adj']:>9.3f}  {sig}")

    n_sig = sum(1 for c in comparisons if c["p_adj"] < alpha)
    print(f"\n  {n_sig}/{len(comparisons)} comparisons significant "
          f"after Holm correction (alpha={alpha}).")
    if n_sig == 0:
        print("  → No arm separates from the baseline once multiplicity is "
              "accounted for. At this n that is an honest 'no detectable "
              "effect', not evidence of no effect.")

    # ---- per-level breakdown ------------------------------------------------
    if by_level:
        meta = load_eval_metadata(domain)
        if not meta:
            print("\n  (per-level breakdown unavailable — eval metadata not found)")
            return
        levels = sorted({m["level"] for m in meta.values()})
        n_per_level = {l: sum(1 for m in meta.values() if m["level"] == l)
                       for l in levels}
        print("\n  ACCURACY BY AMBIGUITY LEVEL")
        print("  (seeds and strategies collapsed to one verdict per row, so n is the "
              "true\n   number of eval rows per level — not runs x rows)")
        print(f"  {'condition':<34}" + "".join(
            f"{l.replace('Level ', 'L') + f' (n={n_per_level[l]})':>16}"
            for l in levels))
        print("  " + "-" * (34 + 16 * len(levels)))
        for exp in exps:
            rs = [r for k, v in grouped.items() if k[0] == exp for r in v]
            if not rs:
                continue
            line = f"  {EXP_LABELS.get(exp, exp):<34}"
            for lvl in levels:
                idxs = [i for i, m in meta.items() if m["level"] == lvl]
                hits = n = 0
                for i in idxs:
                    verdicts = [r["by_index"][i] for r in rs if i in r["by_index"]]
                    if not verdicts:
                        continue
                    n += 1
                    if np.mean(verdicts) > 0.5:      # majority across runs
                        hits += 1
                if n == 0:
                    line += f"{'—':>16}"
                    continue
                lo, hi = wilson_interval(hits, n)
                line += f"{f'{100*hits/n:.0f} [{lo:.0f}-{hi:.0f}]':>16}"
            print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", action="append",
                    help="Domain dir to analyse (repeatable). Default: all three.")
    ap.add_argument("--baseline", default="exp1_base",
                    help="Experiment tag used as the comparison baseline.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--by-level", action="store_true",
                    help="Add the per-ambiguity-level accuracy breakdown.")
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="Bootstrap resamples for paired delta CIs.")
    args = ap.parse_args()

    for domain in (args.domain or DOMAINS):
        if not os.path.isdir(domain):
            print(f"[skip] {domain} — not found")
            continue
        report_domain(domain, args.baseline, args.alpha, args.by_level, args.n_boot)
    print()


if __name__ == "__main__":
    main()
