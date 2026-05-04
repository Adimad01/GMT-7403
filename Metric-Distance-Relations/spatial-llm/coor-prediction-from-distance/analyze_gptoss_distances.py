"""
analyze_gptoss_distances.py — Distance Prediction Evaluation
=============================================================
Evaluates Base and Fine-tuned GPT-OSS across all prompt types
(p1–p4) and shot settings (zero-shot, 3-shot).

Predictions are re-extracted from the model output text (not from
the pre-filled predicted_dis column, which is unreliable for these files).

Saves one txt results file per model:
  outputs/results_base_gptoss.txt
  outputs/results_finetuned_gptoss.txt

Run from coor-prediction-from-distance/:
    python analyze_gptoss_distances.py
"""

import json
import math
import re
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = Path(__file__).parent / "outputs"
MAX_EARTH_KM = 20_100  # no two points on Earth farther apart

PROMPT_DESCRIPTIONS = {
    "p1": "City names only          — answer-only format",
    "p2": "City names + coordinates — answer-only format",
    "p3": "City names only          — reasoning format",
    "p4": "City names + coordinates — reasoning format",
}

MODELS = {
    "Base GPT-OSS": {
        "tag": "gptoss",
        "out": OUTPUTS_DIR / "results_base_gptoss.txt",
    },
    "Fine-tuned GPT-OSS": {
        "tag": "gptoss_finetuned",
        "out": OUTPUTS_DIR / "results_finetuned_gptoss.txt",
    },
}

CONFIGS = [
    ("p1", "zero-shot"),
    ("p1", "3-shot"),
    ("p2", "zero-shot"),
    ("p2", "3-shot"),
    ("p3", "zero-shot"),
    ("p3", "3-shot"),
    ("p4", "zero-shot"),
    ("p4", "3-shot"),
]


# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE EXTRACTION  (re-extracted from output text, not predicted_dis)
# ─────────────────────────────────────────────────────────────────────────────
_CLEAN_RE   = re.compile(r"[,\s  ]")
_UNIT_RE    = re.compile(r"([\d][\d,\s  ]*\.?[\d]*)\s*(?:km|kilometers)", re.IGNORECASE)
_NUMBER_RE  = re.compile(r"([\d][\d,\s  ]*\.?[\d]*)")


def _to_float(raw: str):
    """Convert a raw numeric string to float after stripping whitespace/commas."""
    cleaned = _CLEAN_RE.sub("", raw)
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_distance(text: str):
    """
    Return the best distance estimate from model output text.
    Priority:
      1. First number followed by km/kilometers within Earth range
      2. First number in [50, MAX_EARTH_KM]
      3. Last number within Earth range (fallback)
    Returns (value, strategy) or (None, None).
    """
    if not text:
        return None, None
    if isinstance(text, list):
        text = " ".join(text)

    # Strategy 1: number + unit
    for raw in _UNIT_RE.findall(text):
        val = _to_float(raw)
        if val and 0 < val <= MAX_EARTH_KM:
            return val, "unit"

    all_nums = _NUMBER_RE.findall(text)

    # Strategy 2: first plausible distance number
    for raw in all_nums:
        val = _to_float(raw)
        if val and 50 <= val <= MAX_EARTH_KM:
            return val, "first>=50"

    # Strategy 3: last number within range
    for raw in reversed(all_nums):
        val = _to_float(raw)
        if val and 0 < val <= MAX_EARTH_KM:
            return val, "fallback_last"

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADER & EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────
def load_file(model_tag: str, p_type: str, shots: str):
    """Load a generated-distance JSON output file for a given model/prompt/shots combo."""
    path = OUTPUTS_DIR / f"gen_dis_{model_tag}_{p_type}_{shots}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_config(records: list) -> dict:
    """Compute MAE, RMSE, MAPE, R2, and error stats for a list of prediction records."""
    actuals, predicted, failed = [], [], 0
    strategies = {}

    for r in records:
        ground_truth = r.get("distance")
        raw_output   = r.get("output", "")
        text = raw_output[0] if isinstance(raw_output, list) else str(raw_output)

        val, strat = extract_distance(text)

        # last-resort: fall back to pre-filled predicted_dis if output empty
        if val is None:
            fb = r.get("predicted_dis")
            if fb is not None:
                fv = float(fb)
                if 0 < fv <= MAX_EARTH_KM:
                    val, strat = fv, "fallback_col"

        if val is not None and ground_truth is not None:
            actuals.append(ground_truth)
            predicted.append(val)
            strategies[strat] = strategies.get(strat, 0) + 1
        else:
            failed += 1

    n_valid = len(actuals)
    n_total = len(records)

    if n_valid == 0:
        return {
            "n": n_total, "n_valid": 0, "n_failed": failed,
            "mae": float("nan"), "rmse": float("nan"),
            "mape": float("nan"), "r2": float("nan"),
            "median_err": float("nan"), "max_err": float("nan"),
            "std_err": float("nan"), "strategies": {},
        }

    a = np.array(actuals)
    p = np.array(predicted)
    errors = np.abs(p - a)

    mae        = float(np.mean(errors))
    rmse       = float(np.sqrt(np.mean((p - a) ** 2)))
    mape       = float(np.mean(errors / np.where(a == 0, 1, a)) * 100)
    ss_res     = float(np.sum((p - a) ** 2))
    ss_tot     = float(np.sum((a - np.mean(a)) ** 2))
    r2         = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    median_err = float(np.median(errors))
    max_err    = float(np.max(errors))
    std_err    = float(np.std(errors))

    return {
        "n": n_total, "n_valid": n_valid, "n_failed": failed,
        "mae": mae, "rmse": rmse, "mape": mape, "r2": r2,
        "median_err": median_err, "max_err": max_err, "std_err": std_err,
        "strategies": strategies,
        "_actuals": a, "_predicted": p, "_errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def fmt(val, suffix="", width=9) -> str:
    """Format a float value for table display, returning an em-dash for NaN."""
    if math.isnan(val):
        return f"{'—':>{width}}"
    return f"{val:{width}.1f}{suffix}"


def build_report(model_name: str, model_tag: str) -> str:
    """Build a formatted text report for all prompt/shot configs of a given model."""
    SEP  = "═" * 80
    DASH = "─" * 80

    results = []
    for p_type, shots in CONFIGS:
        records = load_file(model_tag, p_type, shots)
        if records is None:
            results.append((p_type, shots, None))
        else:
            results.append((p_type, shots, evaluate_config(records)))

    # ── header ────────────────────────────────────────────────────────────────
    lines = [
        SEP,
        f"  Distance Prediction Results — {model_name}",
        f"  Task   : Estimate geographic distance (km) between city pairs",
        f"  Method : Predictions re-extracted from model output text",
        SEP,
        "",
        "  Prompt Types:",
    ]
    for pt, desc in PROMPT_DESCRIPTIONS.items():
        lines.append(f"    {pt.upper()} — {desc}")

    # ── main results table ────────────────────────────────────────────────────
    lines += [
        "",
        DASH,
        f"  {'Config':<16} {'N':>5} {'Valid':>6}  {'MAE km':>9}  {'RMSE km':>9}"
        f"  {'MAPE %':>8}  {'R²':>7}  {'Median km':>10}",
        "  " + "─" * 72,
    ]

    valid_rows = [(pt, sh, m) for pt, sh, m in results if m is not None and not math.isnan(m["mae"])]

    for p_type, shots, m in results:
        cfg = f"{p_type.upper()} {shots}"
        if m is None:
            lines.append(f"  {cfg:<16}  — file not found")
            continue
        r2_str  = f"{m['r2']:>7.4f}" if not math.isnan(m["r2"]) else f"{'—':>7}"
        pct_val = m["n_valid"] / m["n"] * 100 if m["n"] > 0 else 0
        lines.append(
            f"  {cfg:<16} {m['n']:>5} {m['n_valid']:>5} "
            f"({pct_val:>4.0f}%)  "
            f"{fmt(m['mae']):>9}  {fmt(m['rmse']):>9}  "
            f"{fmt(m['mape']):>7}%  {r2_str}  {fmt(m['median_err']):>10}"
        )

    # ── best config ───────────────────────────────────────────────────────────
    if valid_rows:
        best = min(valid_rows, key=lambda x: x[2]["mae"])
        b_pt, b_sh, b_m = best
        r2_str = f"{b_m['r2']:.4f}" if not math.isnan(b_m["r2"]) else "—"
        lines += [
            "  " + "─" * 72,
            f"  ◀ Best config by MAE : {b_pt.upper()} {b_sh}  "
            f"(MAE={b_m['mae']:.1f} km  |  RMSE={b_m['rmse']:.1f} km  |  R²={r2_str})",
        ]

    # ── error distribution for best config ───────────────────────────────────
    if valid_rows:
        b_pt, b_sh, b_m = best
        errors = b_m.get("_errors")
        if errors is not None and len(errors):
            lines += [
                "",
                DASH,
                f"  Error Distribution — Best Config ({b_pt.upper()} {b_sh})",
                "  " + "─" * 72,
            ]
            buckets = [
                ("<  100 km",       errors <  100),
                ("100–500 km",     (errors >= 100)  & (errors <  500)),
                ("500–1000 km",    (errors >= 500)  & (errors < 1000)),
                ("1000–2000 km",   (errors >= 1000) & (errors < 2000)),
                ("> 2000 km",       errors >= 2000),
            ]
            for label, mask in buckets:
                count = int(mask.sum())
                pct   = count / len(errors) * 100
                bar   = "█" * int(pct / 2.5)
                lines.append(f"  {label:<16}  {count:>4} ({pct:>5.1f}%)  {bar}")

            lines += [
                "",
                f"  Min error : {errors.min():.1f} km",
                f"  Max error : {errors.max():.1f} km",
                f"  Std dev   : {errors.std():.1f} km",
            ]

    # ── extraction strategy breakdown for best config ─────────────────────────
    if valid_rows and b_m.get("strategies"):
        lines += [
            "",
            DASH,
            f"  Extraction Strategies — Best Config ({b_pt.upper()} {b_sh})",
            "  " + "─" * 50,
        ]
        strat_labels = {
            "unit":          "Number followed by 'km/kilometers'",
            "first>=50":     "First plausible numeric token (≥50 km)",
            "fallback_last": "Last numeric token (fallback)",
            "fallback_col":  "Pre-computed predicted_dis column",
        }
        for strat, count in sorted(b_m["strategies"].items(), key=lambda x: -x[1]):
            label = strat_labels.get(strat, strat)
            lines.append(f"  {label:<40} : {count:>4}")

    lines += ["", SEP]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Generate and save distance evaluation reports for all configured models."""
    for model_name, cfg in MODELS.items():
        report = build_report(model_name, cfg["tag"])
        cfg["out"].write_text(report, encoding="utf-8")
        print(f"Saved → {cfg['out'].name}")
        print(report)


if __name__ == "__main__":
    main()
