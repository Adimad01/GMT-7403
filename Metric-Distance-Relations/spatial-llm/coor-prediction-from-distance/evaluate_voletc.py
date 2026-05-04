"""
evaluate_voletc.py — Volet C Metric Distance Evaluation & Comparison
=====================================================================
Reads Volet C checkpoint files and compares with Volet B baselines.
Produces a unified summary table with MAE, RMSE, median error, and
prediction rate for each strategy.

Usage:
  python evaluate_voletc.py
  python evaluate_voletc.py --results-dir ./results --voletb-dir ./outputs
"""

import os
import re
import json
import argparse
import numpy as np
from datetime import datetime


# ─── Configuration ────────────────────────────────────────────────────────────
VOLETC_RESULTS_DIR = "results"
VOLETB_OUTPUTS_DIR = "outputs"
REPORTS_DIR = "results/reports"
MAX_EARTH_DISTANCE_KM = 20_100

# Volet B file pattern: gen_dis_{model}_{p_type}_{shots}[_suffix].json
VOLETB_PATTERN = re.compile(r"gen_dis_(.+?)_(p\d+)_(zero-shot|\d+-shot).*?\.json")
# Volet C checkpoint pattern: voletc_{tag}_{strategy}_ckpt.json
VOLETC_PATTERN = re.compile(r"voletc_(.+?)_(cot|tot|got)_ckpt\.json")


# ─── Volet B distance extractor (from evaluate_distances.py) ──────────────────
def _clean_number(s):
    """Strip whitespace, commas, and non-breaking spaces from a numeric string."""
    if s is None:
        return ""
    return re.sub(r"[,\s\u00a0\u202f]", "", s)


def _extract_voletb_distance(text):
    """Extract distance from Volet B model output."""
    if not text:
        return None
    if isinstance(text, list):
        text = " ".join(text)

    unit_pat = re.compile(
        r"([\d][\d,\s\u00a0\u202f]*\.?[\d]*)\s*(?:km|kilometers)", re.I
    )
    num_pat = re.compile(r"([\d][\d,\s\u00a0\u202f]*\.?[\d]*)")

    for raw in unit_pat.findall(text):
        cleaned = _clean_number(raw)
        try:
            val = float(cleaned)
            if 0 < val <= MAX_EARTH_DISTANCE_KM:
                return val
        except ValueError:
            pass

    for raw in num_pat.findall(text):
        cleaned = _clean_number(raw)
        if not cleaned:
            continue
        try:
            val = float(cleaned)
            if 50 <= val <= MAX_EARTH_DISTANCE_KM:
                return val
        except ValueError:
            pass

    return None


# ─── Load Volet C results ─────────────────────────────────────────────────────
def load_voletc_results(results_dir):
    """Load all Volet C checkpoint files."""
    entries = []
    if not os.path.exists(results_dir):
        return entries

    for fname in sorted(os.listdir(results_dir)):
        match = VOLETC_PATTERN.match(fname)
        if not match:
            continue
        model_tag, strategy = match.groups()
        fpath = os.path.join(results_dir, fname)

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠️  Could not load {fname}: {e}")
            continue

        results = data.get("results", [])
        if not results:
            continue

        errors = [r["error"] for r in results if r.get("error") is not None]
        total = len(results)
        valid = len(errors)

        if errors:
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean(np.array(errors) ** 2))
            median = np.median(errors)
        else:
            mae = rmse = median = float("nan")

        entries.append({
            "volet": "C",
            "model": model_tag,
            "strategy": strategy.upper(),
            "config": f"KG+{strategy.upper()}",
            "total": total,
            "valid": valid,
            "prediction_rate": (valid / total * 100) if total > 0 else 0,
            "mae": mae,
            "rmse": rmse,
            "median": median,
            "file": fname,
        })

    return entries


# ─── Load Volet B results ─────────────────────────────────────────────────────
def load_voletb_results(outputs_dir):
    """Load Volet B output files and compute metrics."""
    entries = []
    if not os.path.exists(outputs_dir):
        return entries

    for fname in sorted(os.listdir(outputs_dir)):
        match = VOLETB_PATTERN.match(fname)
        if not match:
            continue
        model_name, p_type, shots = match.groups()
        fpath = os.path.join(outputs_dir, fname)

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not data:
            continue

        errors = []
        valid = 0
        total = len(data)

        for item in data:
            gt = item.get("distance")
            if gt is None:
                continue
            output_text = item.get("output", "")
            if isinstance(output_text, list):
                output_text = " ".join(str(x) for x in output_text)
            else:
                output_text = str(output_text)

            pred = _extract_voletb_distance(output_text)
            if pred is None:
                fb = item.get("predicted_dis")
                if fb is not None:
                    try:
                        val = float(fb)
                        if 0 < val <= MAX_EARTH_DISTANCE_KM:
                            pred = val
                    except (ValueError, TypeError):
                        pass

            if pred is not None:
                errors.append(abs(pred - gt))
                valid += 1

        if errors:
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean(np.array(errors) ** 2))
            median = np.median(errors)
        else:
            mae = rmse = median = float("nan")

        entries.append({
            "volet": "B",
            "model": model_name,
            "strategy": f"{p_type.upper()}/{shots}",
            "config": f"{p_type.upper()}_{shots}",
            "total": total,
            "valid": valid,
            "prediction_rate": (valid / total * 100) if total > 0 else 0,
            "mae": mae,
            "rmse": rmse,
            "median": median,
            "file": fname,
        })

    return entries


# ─── Unified Summary ──────────────────────────────────────────────────────────
def print_summary(voletb_entries, voletc_entries):
    """Print a unified comparison table."""
    all_entries = voletb_entries + voletc_entries

    if not all_entries:
        print("No results found.")
        return

    print(f"\n{'=' * 100}")
    print(f"  UNIFIED COMPARISON — VOLET B (Baselines) vs VOLET C (KG + Reasoning)")
    print(f"{'=' * 100}")
    print(
        f"  {'Volet':<7} {'Model':<15} {'Config':<18} "
        f"{'Rate':>6}   {'MAE':>10}   {'RMSE':>10}   {'Median':>10}   "
        f"{'Valid':>6}"
    )
    print(f"  {'-' * 90}")

    # Group by volet
    for volet_label, entries in [("B", voletb_entries), ("C", voletc_entries)]:
        if not entries:
            continue
        # Sort by MAE
        sorted_entries = sorted(entries, key=lambda x: x["mae"] if not np.isnan(x["mae"]) else 1e9)
        for e in sorted_entries:
            print(
                f"  {e['volet']:<7} {e['model']:<15} {e['config']:<18} "
                f"{e['prediction_rate']:>5.1f}%  "
                f"{e['mae']:>8.1f}km  "
                f"{e['rmse']:>8.1f}km  "
                f"{e['median']:>8.1f}km  "
                f"{e['valid']:>3d}/{e['total']}"
            )
        print(f"  {'-' * 90}")

    # Best of each volet
    print(f"\n  📊 BEST RESULTS:")
    for volet_label, entries in [("B", voletb_entries), ("C", voletc_entries)]:
        if not entries:
            continue
        valid_entries = [e for e in entries if not np.isnan(e["mae"])]
        if valid_entries:
            best = min(valid_entries, key=lambda x: x["mae"])
            print(
                f"     Volet {volet_label}: {best['config']:<18} "
                f"MAE={best['mae']:.1f}km  RMSE={best['rmse']:.1f}km  "
                f"Median={best['median']:.1f}km"
            )

    print(f"{'=' * 100}\n")


def save_report(voletb_entries, voletc_entries, reports_dir):
    """Save a JSON report for further analysis."""
    os.makedirs(reports_dir, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "volet_b": voletb_entries,
        "volet_c": voletc_entries,
    }

    report_path = os.path.join(reports_dir, "voletc_comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"  📄 Report saved: {report_path}")

    # Also save a text summary
    txt_path = os.path.join(reports_dir, "voletc_comparison_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"VOLET C EVALUATION SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        all_entries = voletb_entries + voletc_entries
        f.write(
            f"{'Volet':<7} {'Model':<15} {'Config':<18} "
            f"{'Rate':>6}   {'MAE':>10}   {'RMSE':>10}   {'Median':>10}   "
            f"{'Valid':>6}\n"
        )
        f.write("-" * 95 + "\n")
        for e in sorted(all_entries, key=lambda x: (x["volet"], x["mae"] if not np.isnan(x["mae"]) else 1e9)):
            f.write(
                f"{e['volet']:<7} {e['model']:<15} {e['config']:<18} "
                f"{e['prediction_rate']:>5.1f}%  "
                f"{e['mae']:>8.1f}km  "
                f"{e['rmse']:>8.1f}km  "
                f"{e['median']:>8.1f}km  "
                f"{e['valid']:>3d}/{e['total']}\n"
            )

    print(f"  📄 Text summary saved: {txt_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    """Parse arguments, load Volet B and C results, print comparison, and save reports."""
    parser = argparse.ArgumentParser(
        description="Evaluate Volet C results and compare with Volet B baselines."
    )
    parser.add_argument("--results-dir", default=VOLETC_RESULTS_DIR,
                        help="Directory with Volet C checkpoint files")
    parser.add_argument("--voletb-dir", default=VOLETB_OUTPUTS_DIR,
                        help="Directory with Volet B output files")
    parser.add_argument("--reports-dir", default=REPORTS_DIR,
                        help="Directory for output reports")
    args = parser.parse_args()

    print(f"📂 Scanning Volet C results: {args.results_dir}")
    voletc = load_voletc_results(args.results_dir)
    print(f"   Found {len(voletc)} Volet C result files.")

    print(f"📂 Scanning Volet B baselines: {args.voletb_dir}")
    voletb = load_voletb_results(args.voletb_dir)
    print(f"   Found {len(voletb)} Volet B result files.")

    print_summary(voletb, voletc)
    save_report(voletb, voletc, args.reports_dir)


if __name__ == "__main__":
    main()
