"""
evaluate_osm_wikidata.py — Coordinate Prediction: Base (OSM-KG) vs Wikidata
============================================================================
Parses the three wikidata result files (CoT / ToT / GoT), each of which embeds:
  • Old Base Error — fixed reference baseline (OSM-KG, no Wikidata enrichment)
  • New Wiki Error — Wikidata-augmented prediction for that strategy

Computes per-city and aggregate metrics, flags hemisphere-flip failures,
and writes a per-city CSV for further analysis.
"""

import re
import math
import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
METRICS_DIR = Path(__file__).parent / "eval_outputs_real"
STRATEGIES   = ["cot", "tot", "got"]

HEMISPHERE_FLIP_THRESHOLD_KM = 5_000   # errors above this flag sign-flip failures

# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_result_file(path: Path) -> list[dict]:
    """
    Returns one dict per city block:
      city, true_lat, true_lon,
      wiki_pred_lat, wiki_pred_lon,
      base_err_km, wiki_err_km,
      wiki_valid (bool)
    """
    text  = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"---\s+City\s+(\d+)\s+/\s+\d+\s+---", text)

    records = []
    for i in range(1, len(blocks), 2):
        city_id = int(blocks[i])
        body    = blocks[i + 1]

        city_m = re.search(r"Predicting coordinates for:\s+(.+)", body)
        true_m = re.search(r"True Coordinates:\s+Lat\s+([-\d.]+),\s+Lon\s+([-\d.]+)", body)
        pred_m = re.search(r"✅ FINAL PREDICTION:\s+\(([-\d.]+),\s+([-\d.]+)\)", body)
        cmp_m  = re.search(
            r"Old Base Error:\s+([\d.]+)\s+km\s+\|\s+New Wiki Error:\s+([\d.]+)\s+km",
            body,
        )

        if not (city_m and true_m):
            continue

        city     = city_m.group(1).strip()
        true_lat = float(true_m.group(1))
        true_lon = float(true_m.group(2))

        wiki_pred_lat = float(pred_m.group(1)) if pred_m else None
        wiki_pred_lon = float(pred_m.group(2)) if pred_m else None
        base_err      = float(cmp_m.group(1))  if cmp_m  else None
        wiki_err      = float(cmp_m.group(2))  if cmp_m  else None
        wiki_valid    = pred_m is not None and cmp_m is not None

        records.append(dict(
            city_id=city_id, city=city,
            true_lat=true_lat, true_lon=true_lon,
            wiki_pred_lat=wiki_pred_lat, wiki_pred_lon=wiki_pred_lon,
            base_err_km=base_err, wiki_err_km=wiki_err,
            wiki_valid=wiki_valid,
        ))
    return records


def is_hemisphere_flip(r: dict) -> bool:
    """True when the LLM predicted correct magnitude but wrong sign."""
    if not r["wiki_valid"] or r["wiki_pred_lat"] is None:
        return False
    lat_flip = (
        abs(r["wiki_pred_lat"]) > 1 and
        r["true_lat"] * r["wiki_pred_lat"] < 0 and
        abs(abs(r["wiki_pred_lat"]) - abs(r["true_lat"])) < 5
    )
    lon_flip = (
        abs(r["wiki_pred_lon"]) > 1 and
        r["true_lon"] * r["wiki_pred_lon"] < 0 and
        abs(abs(r["wiki_pred_lon"]) - abs(r["true_lon"])) < 10
    )
    return (lat_flip or lon_flip) or (r["wiki_err_km"] is not None and
                                       r["wiki_err_km"] >= HEMISPHERE_FLIP_THRESHOLD_KM)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _sorted_median(vals: list[float]) -> float:
    """Return the median of a list of floats using a simple sort-based approach."""
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def compute_metrics(errors_km: list[float], n_total: int) -> dict:
    """Compute MAE, RMSE, MdAE, and within-threshold accuracy from a list of km errors."""
    n = len(errors_km)
    if n == 0:
        return dict(n_total=n_total, n_valid=0, pred_rate=0.0,
                    mae=None, rmse=None, mdae=None,
                    within_50=None, within_100=None, within_200=None)
    mae    = sum(errors_km) / n
    rmse   = math.sqrt(sum(e**2 for e in errors_km) / n)
    mdae   = _sorted_median(errors_km)
    w50    = sum(1 for e in errors_km if e <=  50) / n * 100
    w100   = sum(1 for e in errors_km if e <= 100) / n * 100
    w200   = sum(1 for e in errors_km if e <= 200) / n * 100
    return dict(n_total=n_total, n_valid=n, pred_rate=n / n_total * 100,
                mae=mae, rmse=rmse, mdae=mdae,
                within_50=w50, within_100=w100, within_200=w200)


def fmt(val, dec=1, suffix=""):
    """Format a numeric value to a fixed number of decimal places, or return 'N/A' if None."""
    return "N/A" if val is None else f"{val:.{dec}f}{suffix}"


# ── Printing helpers ──────────────────────────────────────────────────────────

DIV = "─" * 116

def section(title):
    """Print a decorated section header to stdout."""
    print(f"\n{DIV}\n  {title}\n{DIV}")


# ── Full-dataset base metrics ─────────────────────────────────────────────────

def load_full_base_metrics() -> dict:
    """Reads metrics_eval_*_outputs.json for the full 1 058-city baseline."""
    result = {}
    for strat in STRATEGIES:
        path = METRICS_DIR / f"metrics_eval_{strat}_outputs.json"
        if path.exists():
            result[strat] = json.loads(path.read_text())
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Parse Wikidata result files, print comparison tables, and save a per-city CSV."""
    # Load per-city data for all 3 strategies
    data: dict[str, list[dict]] = {}
    for strat in STRATEGIES:
        path = RESULTS_DIR / f"coord_eval_wikidata_{strat}.txt"
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            data[strat] = []
        else:
            data[strat] = parse_result_file(path)

    full_base = load_full_base_metrics()

    # ── 1. Per-city detail table ──────────────────────────────────────────────
    section("PER-CITY DETAIL — Base (OSM-KG) error vs Wikidata error (km)")

    # Use CoT city list as the reference (all 3 strategies share the same 10 cities)
    ref_records = data["cot"]

    # Column widths
    cw = [4, 22, 10, 10, 11, 11, 11, 11, 11, 11]
    hdr = (
        f"{'#':>{cw[0]}}  {'City':<{cw[1]}}  "
        f"{'TrueLat':>{cw[2]}}  {'TrueLon':>{cw[3]}}  "
        f"{'Base(OSM)':>{cw[4]}}  "
        f"{'WD-CoT':>{cw[5]}}  {'WD-ToT':>{cw[6]}}  {'WD-GoT':>{cw[7]}}  "
        f"{'Flip?':>{cw[8]}}"
    )
    print(hdr)
    print(DIV)

    flip_markers: dict[str, dict[str, bool]] = {s: {} for s in STRATEGIES}
    for r in ref_records:
        city = r["city"]
        base_e = fmt(r["base_err_km"])

        wiki_errs = {}
        flips = {}
        for strat in STRATEGIES:
            match = next((x for x in data[strat] if x["city"] == city), None)
            if match:
                wiki_errs[strat] = fmt(match["wiki_err_km"])
                flips[strat] = is_hemisphere_flip(match)
                flip_markers[strat][city] = flips[strat]
            else:
                wiki_errs[strat] = "—"
                flips[strat] = False

        any_flip = any(flips.values())
        flip_tag = "★" if any_flip else ""

        row = (
            f"{r['city_id']:>{cw[0]}}  {city:<{cw[1]}}  "
            f"{r['true_lat']:>{cw[2]}.4f}  {r['true_lon']:>{cw[3]}.4f}  "
            f"{base_e:>{cw[4]}}  "
            f"{wiki_errs['cot']:>{cw[5]}}  "
            f"{wiki_errs['tot']:>{cw[6]}}  "
            f"{wiki_errs['got']:>{cw[7]}}  "
            f"{flip_tag:>{cw[8]}}"
        )
        print(row)

    print(DIV)
    print("  ★ = hemisphere sign-flip failure (lat/lon sign reversed, error > 5 000 km)")

    # ── 2. Aggregate metrics per strategy ─────────────────────────────────────
    section("AGGREGATE METRICS — Base (OSM-KG) vs Wikidata, per strategy (10 cities)")

    # Base errors are the same reference across all 3 files; use CoT file values.
    base_errors_all = [r["base_err_km"] for r in ref_records if r["base_err_km"] is not None]
    base_m_all = compute_metrics(base_errors_all, len(ref_records))

    col = 13
    lw  = 20
    strat_hdr = f"{'Metric':<{lw}}" + "".join(
        f"  {'Base(OSM)':>{col}}  {'WD-' + s.upper():>{col}}"
        for s in STRATEGIES
    )
    print(strat_hdr)
    print(DIV)

    def strat_metrics(strat, exclude_flips=False):
        """Collect valid Wikidata errors for a strategy, optionally excluding flip failures."""
        errs = []
        for r in data[strat]:
            if not r["wiki_valid"] or r["wiki_err_km"] is None:
                continue
            if exclude_flips and is_hemisphere_flip(r):
                continue
            errs.append(r["wiki_err_km"])
        return compute_metrics(errs, len(data[strat]))

    metric_defs = [
        ("Valid / Total",    lambda m: f"{m['n_valid']}/{m['n_total']}"),
        ("Pred. Rate (%)",   lambda m: fmt(m["pred_rate"],  1, "%")),
        ("MAE (km)",         lambda m: fmt(m["mae"],        1)),
        ("RMSE (km)",        lambda m: fmt(m["rmse"],       1)),
        ("MdAE (km)",        lambda m: fmt(m["mdae"],       1)),
        ("Within  50 km (%)",lambda m: fmt(m["within_50"],  1, "%")),
        ("Within 100 km (%)",lambda m: fmt(m["within_100"], 1, "%")),
        ("Within 200 km (%)",lambda m: fmt(m["within_200"], 1, "%")),
    ]

    # Full table (all predictions, including flips)
    print("  — Including hemisphere-flip failures —")
    for label, fn in metric_defs:
        row = f"{label:<{lw}}"
        for strat in STRATEGIES:
            m_wiki = strat_metrics(strat, exclude_flips=False)
            row += f"  {fn(base_m_all):>{col}}  {fn(m_wiki):>{col}}"
        print(row)

    print()
    print("  — Excluding hemisphere-flip failures (★) —")
    for label, fn in metric_defs:
        row = f"{label:<{lw}}"
        for strat in STRATEGIES:
            base_errs_excl = [
                r["base_err_km"] for r in ref_records
                if r["base_err_km"] is not None and not flip_markers[strat].get(r["city"], False)
            ]
            m_base_excl = compute_metrics(base_errs_excl, len(ref_records))
            m_wiki = strat_metrics(strat, exclude_flips=True)
            row += f"  {fn(m_base_excl):>{col}}  {fn(m_wiki):>{col}}"
        print(row)

    # ── 3. Head-to-head win/loss per city ─────────────────────────────────────
    section("HEAD-TO-HEAD — Wikidata wins (↓) / loses (↑) vs Base (OSM-KG)")

    cw2 = [22, 12, 12, 12, 12, 12, 12, 12]
    hdr2 = (
        f"{'City':<{cw2[0]}}  {'BaseErr':>{cw2[1]}}"
        + "".join(
            f"  {'WD-' + s.upper():>{cw2[i+2]}}  {'Δ':>{cw2[i+2]}}"
            for i, s in enumerate(STRATEGIES)
        )
    )
    print(hdr2)
    print(DIV)

    for r in ref_records:
        city   = r["city"]
        base_e = r["base_err_km"]
        row    = f"{city:<{cw2[0]}}  {fmt(base_e):>{cw2[1]}}"
        for strat in STRATEGIES:
            match = next((x for x in data[strat] if x["city"] == city), None)
            if match and match["wiki_err_km"] is not None and base_e is not None:
                we  = match["wiki_err_km"]
                delta = we - base_e
                tag   = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
                w = cw2[2] - 2
                row  += f"  {fmt(we):>{cw2[2]}}  {tag}{fmt(abs(delta)):>{w}}"
            else:
                row += f"  {'—':>{cw2[2]}}  {'—':>{cw2[2]}}"
        print(row)

    print(DIV)
    print("  ↓ = Wikidata better than base   ↑ = base better   = = tie (±0)")

    # ── 4. Full-dataset baseline context ──────────────────────────────────────
    section("FULL-DATASET BASELINE — Base (OSM-KG) on 1 058 cities (all strategies)")

    fw = 14
    print(f"{'Metric':<{lw}}" + "".join(f"  {'Base-' + s.upper():>{fw}}" for s in STRATEGIES))
    print(DIV)

    full_metric_defs = [
        ("Valid / Total",    lambda m: f"{m.get('valid_outputs','?')}/{m.get('total','?')}"),
        ("Within 100 km (%)",lambda m: fmt(m.get("accuracy_100km", 0) * 100, 1, "%")),
        ("Avg Error (km)",   lambda m: fmt(m.get("average_error_km"), 1)),
    ]
    for label, fn in full_metric_defs:
        row = f"{label:<{lw}}"
        for strat in STRATEGIES:
            m = full_base.get(strat, {})
            row += f"  {fn(m):>{fw}}"
        print(row)

    # ── 5. Save CSV ────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "evaluation_osm_wikidata.csv"
    fieldnames = [
        "city_id", "city", "true_lat", "true_lon",
        "base_err_km",
        "wd_cot_pred_lat", "wd_cot_pred_lon", "wd_cot_err_km", "wd_cot_flip",
        "wd_tot_pred_lat", "wd_tot_pred_lon", "wd_tot_err_km", "wd_tot_flip",
        "wd_got_pred_lat", "wd_got_pred_lon", "wd_got_err_km", "wd_got_flip",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ref_records:
            city = r["city"]
            row = {
                "city_id":    r["city_id"],
                "city":       city,
                "true_lat":   r["true_lat"],
                "true_lon":   r["true_lon"],
                "base_err_km": r["base_err_km"] if r["base_err_km"] is not None else "",
            }
            for strat in STRATEGIES:
                match = next((x for x in data[strat] if x["city"] == city), None)
                prefix = f"wd_{strat}"
                if match:
                    row[f"{prefix}_pred_lat"] = match["wiki_pred_lat"] if match["wiki_pred_lat"] is not None else ""
                    row[f"{prefix}_pred_lon"] = match["wiki_pred_lon"] if match["wiki_pred_lon"] is not None else ""
                    row[f"{prefix}_err_km"]   = match["wiki_err_km"]   if match["wiki_err_km"]   is not None else ""
                    row[f"{prefix}_flip"]     = int(is_hemisphere_flip(match))
                else:
                    row[f"{prefix}_pred_lat"] = ""
                    row[f"{prefix}_pred_lon"] = ""
                    row[f"{prefix}_err_km"]   = ""
                    row[f"{prefix}_flip"]     = ""
            writer.writerow(row)

    print(f"\n  CSV saved → {csv_path}\n")


if __name__ == "__main__":
    main()
