"""Evaluation metrics.

Accuracy alone is misleading on these sets: the evaluation rows are small, and
some rows assert the *same* fact more than once, so they are not independent
observations. Two things follow, and both are reported:

  * every accuracy carries a Wilson 95% interval, which behaves at small n and
    near 0/100% where the normal approximation does not;
  * accuracy is also reported over unique ``fact_id`` values, which is the
    honest effective sample size.

Macro-F1 is reported alongside accuracy because the label sets are balanced by
design -- if a strategy collapses onto one popular label, macro-F1 falls even
when accuracy does not.

Standard library only, so this runs unchanged on an offline server.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

Z95 = 1.959963984540054


def wilson(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion, as percentages."""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half) * 100.0, min(1.0, centre + half) * 100.0)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def load_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # A retried row appears twice; the last write wins.
    latest: dict[int, dict] = {}
    for rec in out:
        latest[rec["row_index"]] = rec
    return list(latest.values())


def compute(records: list[dict], labels: list[str]) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    n = len(ok)
    if n == 0:
        return {"n": 0, "note": "no completed predictions"}

    hits = sum(1 for r in ok if r.get("correct"))
    lo, hi = wilson(hits, n)

    # Clustered on fact_id: rows asserting the same fact are one observation,
    # scored by majority so a single fact cannot be counted several times.
    by_fact: dict[str, list[bool]] = defaultdict(list)
    for r in ok:
        by_fact[r.get("fact_id", str(r["row_index"]))].append(bool(r.get("correct")))
    fact_hits = sum(1 for v in by_fact.values() if sum(v) * 2 > len(v))
    f_lo, f_hi = wilson(fact_hits, len(by_fact))

    # Per-label precision / recall / F1
    per_label = {}
    for lab in labels:
        tp = sum(1 for r in ok if r.get("predicted") == lab and r.get("gold") == lab)
        fp = sum(1 for r in ok if r.get("predicted") == lab and r.get("gold") != lab)
        fn = sum(1 for r in ok if r.get("predicted") != lab and r.get("gold") == lab)
        p, rc, f1 = prf(tp, fp, fn)
        per_label[lab] = {"support": sum(1 for r in ok if r.get("gold") == lab),
                          "precision": round(p, 4), "recall": round(rc, 4),
                          "f1": round(f1, 4)}
    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(labels) if labels else 0.0

    per_level = {}
    for lvl in sorted({r.get("ambiguity_level", "") for r in ok if r.get("ambiguity_level")}):
        sub = [r for r in ok if r.get("ambiguity_level") == lvl]
        h = sum(1 for r in sub if r.get("correct"))
        l_lo, l_hi = wilson(h, len(sub))
        per_level[lvl] = {"n": len(sub), "accuracy": round(100 * h / len(sub), 2),
                          "ci95": [round(l_lo, 2), round(l_hi, 2)]}

    confusion = Counter((r.get("gold"), r.get("predicted")) for r in ok)

    return {
        "n": n,
        "n_unique_facts": len(by_fact),
        "accuracy": round(100 * hits / n, 2),
        "accuracy_ci95": [round(lo, 2), round(hi, 2)],
        "accuracy_by_fact": round(100 * fact_hits / len(by_fact), 2),
        "accuracy_by_fact_ci95": [round(f_lo, 2), round(f_hi, 2)],
        "macro_f1": round(macro_f1, 4),
        "unparsed": sum(1 for r in ok if r.get("predicted") is None),
        "unparsed_rate": round(100 * sum(1 for r in ok if r.get("predicted") is None) / n, 2),
        "parse_rules": dict(Counter(r.get("parse_rule") for r in ok)),
        "per_label": per_label,
        "per_level": per_level,
        "confusion": {f"{g}->{p}": c for (g, p), c in sorted(confusion.items(),
                                                            key=lambda x: -x[1])},
        "failed_rows": [r["row_index"] for r in records if r.get("status") == "error"],
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes.

    b = rows A got right and B got wrong;  c = the reverse.

    Rows both arms agree on carry no information about which is better and drop
    out. Under the null each discordant row is a fair coin, so the count is
    Binomial(b + c, 0.5). Computed exactly: the chi-square approximation is
    unreliable when discordant rows are few, which is the norm at these sizes.

    This is the right test here because every strategy scores the SAME rows.
    Comparing two independent confidence intervals instead would be far more
    conservative and would miss real differences.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values.

    Ten pairwise comparisons per relation would throw up false positives
    uncorrected; Holm controls the family-wise error rate and is uniformly more
    powerful than plain Bonferroni.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def per_row_correct(records: list[dict]) -> dict[int, bool]:
    """row_index -> was it right. Only completed rows."""
    return {r["row_index"]: bool(r.get("correct"))
            for r in records if r.get("status") == "ok"}


def accuracy_excluding_unparsed(records: list[dict]) -> dict:
    """Accuracy over rows the parser could actually read.

    An unparseable completion is scored wrong, which conflates two different
    failures: the model reasoned badly, or it reasoned fine and formatted the
    answer badly. Reporting both figures separates them.
    """
    ok = [r for r in records if r.get("status") == "ok"]
    parsed = [r for r in ok if r.get("predicted") is not None]
    if not parsed:
        return {"n_parsed": 0}
    hits = sum(1 for r in parsed if r.get("correct"))
    lo, hi = wilson(hits, len(parsed))
    return {"n_parsed": len(parsed),
            "n_unparsed": len(ok) - len(parsed),
            "accuracy_parsed_only": round(100 * hits / len(parsed), 2),
            "accuracy_parsed_only_ci95": [round(lo, 2), round(hi, 2)]}
