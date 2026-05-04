"""
Evaluation of OSM vs Wikidata coordinate-enriched metric distance estimation.
Parses the six result files (osm/wikidata × cot/tot/got) and reports:
  - per-pair details
  - aggregate metrics: MAE, RMSE, MdAE, MAPE, prediction rate,
    within-50 km / within-10% accuracy
  - side-by-side OSM vs Wikidata comparison per strategy
"""

import re
import math
import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
STRATEGIES = ["cot", "tot", "got"]
SOURCES = ["osm", "wikidata"]


def parse_results_file(path: Path) -> list[dict]:
    """
    Parses a metric_eval_<source>_<strategy>.txt file.
    Returns a list of dicts with keys:
      pair_id, city_a, city_b, true_dist, predicted, abs_error, success
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"---\s+Pair\s+(\d+)\s+/\s+\d+\s+---", text)
    # blocks: ['', '1', '<content>', '2', '<content>', ...]
    records = []
    for i in range(1, len(blocks), 2):
        pair_id = int(blocks[i])
        body = blocks[i + 1]

        est = re.search(r"Estimating:\s+(.+?)\s+to\s+(.+?)\n", body)
        true_m = re.search(r"True Distance:\s+([\d.]+)\s+km", body)
        if not est or not true_m:
            continue

        city_a = est.group(1).strip()
        city_b = est.group(2).strip()
        true_dist = float(true_m.group(1))

        pred_m = re.search(
            r"✅ Prediction:\s+([\d.]+)\s+km\s+\|\s+Error:\s+([\d.]+)\s+km", body
        )
        if pred_m:
            predicted = float(pred_m.group(1))
            abs_error = float(pred_m.group(2))
            success = True
        else:
            predicted = None
            abs_error = None
            success = False

        records.append(
            dict(
                pair_id=pair_id,
                city_a=city_a,
                city_b=city_b,
                true_dist=true_dist,
                predicted=predicted,
                abs_error=abs_error,
                success=success,
            )
        )
    return records


def compute_metrics(records: list[dict]) -> dict:
    """Compute MAE, RMSE, MdAE, MAPE, and accuracy metrics from a list of prediction records."""
    valid = [r for r in records if r["success"]]
    n_total = len(records)
    n_valid = len(valid)

    if n_valid == 0:
        return dict(
            n_total=n_total, n_valid=0, pred_rate=0.0,
            mae=None, rmse=None, mdae=None, mape=None,
            within_50=None, within_10pct=None,
        )

    errors = [r["abs_error"] for r in valid]
    trues = [r["true_dist"] for r in valid]
    pct_errors = [e / t * 100 for e, t in zip(errors, trues)]

    mae = sum(errors) / n_valid
    rmse = math.sqrt(sum(e**2 for e in errors) / n_valid)
    mdae = sorted(errors)[n_valid // 2] if n_valid % 2 == 1 else (
        sorted(errors)[n_valid // 2 - 1] + sorted(errors)[n_valid // 2]
    ) / 2
    mape = sum(pct_errors) / n_valid
    within_50 = sum(1 for e in errors if e <= 50) / n_valid * 100
    within_10pct = sum(1 for p in pct_errors if p <= 10.0) / n_valid * 100

    return dict(
        n_total=n_total,
        n_valid=n_valid,
        pred_rate=n_valid / n_total * 100,
        mae=mae,
        rmse=rmse,
        mdae=mdae,
        mape=mape,
        within_50=within_50,
        within_10pct=within_10pct,
    )


def fmt(val, decimals=2, suffix=""):
    """Format a numeric value to a fixed number of decimals, returning 'N/A' for None."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}{suffix}"


def print_divider(width=110):
    """Print a horizontal divider line of the given width."""
    print("─" * width)


def print_header(title, width=110):
    """Print a section header between two divider lines."""
    print_divider(width)
    print(f"  {title}")
    print_divider(width)


def main():
    """Load OSM and Wikidata result files, print comparison tables, and save a CSV."""
    # ── Load all data ──────────────────────────────────────────────────────────
    data: dict[tuple[str, str], list[dict]] = {}
    for source in SOURCES:
        for strat in STRATEGIES:
            path = RESULTS_DIR / f"metric_eval_{source}_{strat}.txt"
            if not path.exists():
                print(f"[WARN] Missing: {path}")
                data[(source, strat)] = []
            else:
                data[(source, strat)] = parse_results_file(path)

    # ── Per-pair detail table ─────────────────────────────────────────────────
    print_header("PER-PAIR DETAILS — OSM vs WIKIDATA (all strategies)")
    col_w = [4, 32, 10, 12, 12, 12, 12, 12, 12]
    header = (
        f"{'#':>{col_w[0]}}  {'Pair':<{col_w[1]}}  {'True(km)':>{col_w[2]}}"
        f"  {'OSM-CoT':>{col_w[3]}}  {'OSM-ToT':>{col_w[4]}}  {'OSM-GoT':>{col_w[5]}}"
        f"  {'WD-CoT':>{col_w[6]}}  {'WD-ToT':>{col_w[7]}}  {'WD-GoT':>{col_w[8]}}"
    )
    print(header)
    print_divider()

    # Build a merged index from all pairs seen across files
    pairs_index: dict[tuple[str, str], dict] = {}
    for (source, strat), records in data.items():
        for r in records:
            key = (r["city_a"], r["city_b"])
            if key not in pairs_index:
                pairs_index[key] = {"true_dist": r["true_dist"], "id": r["pair_id"]}

    sorted_pairs = sorted(pairs_index.items(), key=lambda x: x[1]["id"])

    for (city_a, city_b), meta in sorted_pairs:
        pair_label = f"{city_a} → {city_b}"
        if len(pair_label) > col_w[1]:
            pair_label = pair_label[: col_w[1] - 1] + "…"
        true_d = meta["true_dist"]

        def get_err(source, strat):
            """Return the formatted absolute error for a city pair in a given source/strategy."""
            for r in data[(source, strat)]:
                if r["city_a"] == city_a and r["city_b"] == city_b:
                    if r["success"]:
                        return f"{r['abs_error']:.1f}"
                    return "FAIL"
            return "—"

        row = (
            f"{meta['id']:>{col_w[0]}}  {pair_label:<{col_w[1]}}  {true_d:>{col_w[2]}.1f}"
            f"  {get_err('osm','cot'):>{col_w[3]}}"
            f"  {get_err('osm','tot'):>{col_w[4]}}"
            f"  {get_err('osm','got'):>{col_w[5]}}"
            f"  {get_err('wikidata','cot'):>{col_w[6]}}"
            f"  {get_err('wikidata','tot'):>{col_w[7]}}"
            f"  {get_err('wikidata','got'):>{col_w[8]}}"
        )
        print(row)

    print_divider()
    print("  Values are absolute errors (km). FAIL = no valid prediction extracted.\n")

    # ── Aggregate metrics per strategy ────────────────────────────────────────
    print_header("AGGREGATE METRICS — OSM vs WIKIDATA per Strategy")

    metrics_all: dict[tuple[str, str], dict] = {}
    for source in SOURCES:
        for strat in STRATEGIES:
            metrics_all[(source, strat)] = compute_metrics(data[(source, strat)])

    col = 14
    label_w = 18
    strat_header = (
        f"{'Metric':<{label_w}}"
        + "".join(
            f"  {'OSM-' + s.upper():>{col}}  {'WD-' + s.upper():>{col}}"
            for s in STRATEGIES
        )
    )
    print(strat_header)
    print_divider()

    metric_rows = [
        ("Valid / Total",   lambda m: f"{m['n_valid']}/{m['n_total']}",             False),
        ("Pred. Rate (%)",  lambda m: fmt(m["pred_rate"], 1, "%"),                   False),
        ("MAE (km)",        lambda m: fmt(m["mae"], 2),                               True),
        ("RMSE (km)",       lambda m: fmt(m["rmse"], 2),                              True),
        ("MdAE (km)",       lambda m: fmt(m["mdae"], 2),                              True),
        ("MAPE (%)",        lambda m: fmt(m["mape"], 2, "%"),                         True),
        ("Within 50 km (%)",lambda m: fmt(m["within_50"], 1, "%"),                    False),
        ("Within 10% (%)",  lambda m: fmt(m["within_10pct"], 1, "%"),                 False),
    ]

    for label, fmt_fn, lower_better in metric_rows:
        row = f"{label:<{label_w}}"
        for strat in STRATEGIES:
            osm_m = metrics_all[("osm", strat)]
            wd_m = metrics_all[("wikidata", strat)]
            osm_val = fmt_fn(osm_m)
            wd_val = fmt_fn(wd_m)
            row += f"  {osm_val:>{col}}  {wd_val:>{col}}"
        print(row)

    print_divider()

    # ── Per-source overall (all strategies combined) ──────────────────────────
    print_header("OVERALL SUMMARY — OSM vs WIKIDATA (all 3 strategies combined)")

    combined_osm = [r for strat in STRATEGIES for r in data[("osm", strat)]]
    combined_wd = [r for strat in STRATEGIES for r in data[("wikidata", strat)]]
    m_osm = compute_metrics(combined_osm)
    m_wd = compute_metrics(combined_wd)

    col2 = 20
    print(f"{'Metric':<{label_w}}  {'OSM':>{col2}}  {'Wikidata':>{col2}}")
    print_divider()
    for label, fmt_fn, _ in metric_rows:
        print(f"{label:<{label_w}}  {fmt_fn(m_osm):>{col2}}  {fmt_fn(m_wd):>{col2}}")
    print_divider()

    # ── Save per-pair CSV ─────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "evaluation_osm_wikidata.csv"
    fieldnames = [
        "pair_id", "city_a", "city_b", "true_dist_km",
        "osm_cot_pred", "osm_cot_err",
        "osm_tot_pred", "osm_tot_err",
        "osm_got_pred", "osm_got_err",
        "wd_cot_pred",  "wd_cot_err",
        "wd_tot_pred",  "wd_tot_err",
        "wd_got_pred",  "wd_got_err",
    ]

    def get_pred_err(source, strat, city_a, city_b):
        """Return the predicted distance and absolute error for a city pair."""
        for r in data[(source, strat)]:
            if r["city_a"] == city_a and r["city_b"] == city_b:
                return r["predicted"], r["abs_error"]
        return None, None

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (city_a, city_b), meta in sorted_pairs:
            row = {
                "pair_id": meta["id"],
                "city_a": city_a,
                "city_b": city_b,
                "true_dist_km": meta["true_dist"],
            }
            for src, strat, prefix in [
                ("osm", "cot", "osm_cot"),
                ("osm", "tot", "osm_tot"),
                ("osm", "got", "osm_got"),
                ("wikidata", "cot", "wd_cot"),
                ("wikidata", "tot", "wd_tot"),
                ("wikidata", "got", "wd_got"),
            ]:
                pred, err = get_pred_err(src, strat, city_a, city_b)
                row[f"{prefix}_pred"] = pred if pred is not None else ""
                row[f"{prefix}_err"] = err if err is not None else ""
            writer.writerow(row)

    print(f"\n  CSV saved → {csv_path}")


if __name__ == "__main__":
    main()
