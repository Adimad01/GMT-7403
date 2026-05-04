"""
Compare all voletc_dynamic_* result files and produce a summary comparison table.
Outputs: comparison_dynamic_results.txt
"""

import re
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_FILE = RESULTS_DIR / "comparison_dynamic_results.txt"

PREDICATES = ["within", "contains", "touches", "crosses", "disjoint", "overlaps", "equals"]
VALID_PREDICATES = set(PREDICATES)

FILES = {
    ("OSM",      "CoT"): "voletc_dynamic_osm_neighborhood_details_spatial_relation_16_sample_cot_neighborhood_details_spatial_relation_16_sample.txt",
    ("OSM",      "ToT"): "voletc_dynamic_osm_neighborhood_details_spatial_relation_16_sample_tot_neighborhood_details_spatial_relation_16_sample.txt",
    ("OSM",      "GoT"): "voletc_dynamic_osm_neighborhood_details_spatial_relation_16_sample_got_neighborhood_details_spatial_relation_16_sample.txt",
    ("Wikidata", "CoT"): "voletc_dynamic_wikidata_neighborhood_details_spatial_relation_16_sample_wikidata_cot_neighborhood_details_spatial_relation_16_sample_wikidata.txt",
    ("Wikidata", "ToT"): "voletc_dynamic_wikidata_neighborhood_details_spatial_relation_16_sample_wikidata_tot_neighborhood_details_spatial_relation_16_sample_wikidata.txt",
    ("Wikidata", "GoT"): "voletc_dynamic_wikidata_neighborhood_details_spatial_relation_16_sample_wikidata_got_neighborhood_details_spatial_relation_16_sample_wikidata.txt",
}

RESULT_RE = re.compile(
    r"RESULT \| Expected=(\w+) \| Predicted=(\w+) \| (CORRECT|WRONG)"
)


def parse_file(path: Path) -> dict:
    """Parse a dynamic-KG result log file and return aggregated accuracy statistics."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = RESULT_RE.search(line)
            if m:
                expected  = m.group(1).lower()
                predicted = m.group(2).lower()
                outcome   = m.group(3)
                rows.append((expected, predicted, outcome))

    total   = len(rows)
    correct = sum(1 for _, _, o in rows if o == "CORRECT")
    wrong   = sum(1 for _, _, o in rows if o == "WRONG")
    invalid = sum(1 for _, p, _ in rows if p not in VALID_PREDICATES)

    # per-predicate stats
    per_pred: dict[str, dict] = {}
    for pred in PREDICATES:
        support   = [(e, p, o) for e, p, o in rows if e == pred]
        n         = len(support)
        n_correct = sum(1 for _, _, o in support if o == "CORRECT")
        n_invalid = sum(1 for _, p, _ in support if p not in VALID_PREDICATES)
        per_pred[pred] = {
            "support":  n,
            "correct":  n_correct,
            "invalid":  n_invalid,
            "accuracy": (n_correct / n * 100) if n > 0 else float("nan"),
        }

    return {
        "total":    total,
        "correct":  correct,
        "wrong":    wrong,
        "invalid":  invalid,
        "accuracy": (correct / total * 100) if total > 0 else float("nan"),
        "per_pred": per_pred,
    }


def bar(pct: float, width: int = 20) -> str:
    """Return a Unicode block-character progress bar string for a percentage value."""
    if pct != pct:  # nan
        return " " * width
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def fmt(val: float) -> str:
    """Format a float as a percentage string, returning 'N/A' for NaN values."""
    if val != val:
        return "  N/A  "
    return f"{val:6.2f}%"


def build_report(stats: dict) -> str:
    """Build the full plain-text comparison report string from parsed statistics."""
    lines = []
    W = 96

    def sep(char="═"):
        lines.append(char * W)

    def title(text):
        lines.append(f"  {text}")

    sep()
    title("DYNAMIC KG APPROACHES — COMPARISON TABLE")
    title("Topological Relation Classification (7 DE-9IM predicates)")
    title(f"KG Sources: OSM (OpenStreetMap API) | Wikidata API")
    title(f"Strategies : CoT | ToT | GoT")
    sep()
    lines.append("")

    # ── Section 1: Overall accuracy ──────────────────────────────────────────
    sep("─")
    title("SECTION 1 — Overall Accuracy")
    sep("─")
    hdr = f"  {'KG Source':<10} {'Strategy':<8} {'Total':>6} {'Correct':>8} {'Wrong':>7} {'Invalid':>8}  {'Accuracy':>9}  {'Bar (20)'}"
    lines.append(hdr)
    lines.append("  " + "─" * (W - 2))

    best_acc = max(v["accuracy"] for v in stats.values())

    for (kg, strat), info in stats.items():
        marker = " ◀ BEST" if abs(info["accuracy"] - best_acc) < 0.01 else ""
        row = (
            f"  {kg:<10} {strat:<8} {info['total']:>6} {info['correct']:>8} "
            f"{info['wrong']:>7} {info['invalid']:>8}  "
            f"{fmt(info['accuracy']):>9}  {bar(info['accuracy'])}{marker}"
        )
        lines.append(row)

    lines.append("")

    # ── Section 2: Per-predicate accuracy by KG source ───────────────────────
    for kg_src in ["OSM", "Wikidata"]:
        sep("─")
        title(f"SECTION 2 — Per-Predicate Accuracy  [{kg_src}]")
        sep("─")

        strategies = ["CoT", "ToT", "GoT"]
        col_w = 18

        # header
        hdr = f"  {'Predicate':<12}"
        for strat in strategies:
            hdr += f"  {strat:>{col_w}}"
        lines.append(hdr)
        lines.append("  " + "─" * (W - 2))

        for pred in PREDICATES:
            row = f"  {pred:<12}"
            for strat in strategies:
                info = stats[(kg_src, strat)]
                pp   = info["per_pred"][pred]
                if pp["support"] == 0:
                    cell = f"{'N/A':>{col_w}}"
                else:
                    cell = f"{pp['correct']:>3}/{pp['support']:>3} ({pp['accuracy']:5.1f}%){' ':>{col_w - 16}}"
                row += f"  {cell}"
            lines.append(row)

        lines.append("")

    # ── Section 3: OSM vs Wikidata delta ─────────────────────────────────────
    sep("─")
    title("SECTION 3 — OSM vs Wikidata Δ Accuracy (OSM − Wikidata)")
    sep("─")
    hdr = f"  {'Strategy':<10}  {'OSM':>9}  {'Wikidata':>9}  {'Δ (pp)':>9}"
    lines.append(hdr)
    lines.append("  " + "─" * (W - 2))

    for strat in ["CoT", "ToT", "GoT"]:
        osm_acc  = stats[("OSM",      strat)]["accuracy"]
        wiki_acc = stats[("Wikidata", strat)]["accuracy"]
        delta    = osm_acc - wiki_acc
        sign     = "+" if delta >= 0 else ""
        lines.append(
            f"  {strat:<10}  {fmt(osm_acc):>9}  {fmt(wiki_acc):>9}  {sign}{delta:+.2f} pp"
        )

    lines.append("")

    # ── Section 4: Per-predicate delta (CoT only as reference) ───────────────
    sep("─")
    title("SECTION 4 — Per-Predicate Δ Accuracy OSM vs Wikidata (all strategies)")
    sep("─")

    col_w2 = 14
    hdr = f"  {'Predicate':<12}"
    for strat in ["CoT", "ToT", "GoT"]:
        hdr += f"  {'Δ ' + strat:>{col_w2}}"
    lines.append(hdr)
    lines.append("  " + "─" * (W - 2))

    for pred in PREDICATES:
        row = f"  {pred:<12}"
        for strat in ["CoT", "ToT", "GoT"]:
            osm_acc  = stats[("OSM",      strat)]["per_pred"][pred]["accuracy"]
            wiki_acc = stats[("Wikidata", strat)]["per_pred"][pred]["accuracy"]
            if osm_acc != osm_acc or wiki_acc != wiki_acc:
                cell = f"{'N/A':>{col_w2}}"
            else:
                delta = osm_acc - wiki_acc
                cell  = f"{delta:+.1f} pp{' ':>{col_w2 - 9}}"
            row += f"  {cell}"
        lines.append(row)

    lines.append("")

    # ── Section 5: Strategy comparison within each KG ────────────────────────
    sep("─")
    title("SECTION 5 — Strategy Comparison Within Each KG Source")
    sep("─")
    hdr = f"  {'KG':<10}  {'CoT':>9}  {'ToT':>9}  {'GoT':>9}  {'Best Strategy'}"
    lines.append(hdr)
    lines.append("  " + "─" * (W - 2))

    for kg_src in ["OSM", "Wikidata"]:
        accs  = {s: stats[(kg_src, s)]["accuracy"] for s in ["CoT", "ToT", "GoT"]}
        best  = max(accs, key=accs.get)
        lines.append(
            f"  {kg_src:<10}  {fmt(accs['CoT']):>9}  {fmt(accs['ToT']):>9}  "
            f"{fmt(accs['GoT']):>9}  {best}"
        )

    lines.append("")
    sep()
    title("END OF REPORT")
    sep()

    return "\n".join(lines)


def main():
    """Parse all configured result files and write the comparison report to disk."""
    stats = {}
    for (kg, strat), fname in FILES.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"[WARN] Missing: {fname}")
            continue
        stats[(kg, strat)] = parse_file(path)
        print(f"[OK] Parsed {kg:10s} {strat}: {stats[(kg,strat)]['total']} rows, "
              f"{stats[(kg,strat)]['accuracy']:.2f}% accuracy")

    report = build_report(stats)
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(f"\n[DONE] Report saved → {OUTPUT_FILE}")
    print(report)


if __name__ == "__main__":
    main()
