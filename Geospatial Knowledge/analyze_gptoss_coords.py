"""
analyze_gptoss_coords.py — Coordinate Prediction Evaluation
============================================================
Evaluates Base GPT-OSS and Fine-tuned GPT-OSS (LoRA) on the task of
predicting geographic coordinates (latitude, longitude) for world cities.

Metric: Haversine distance (km) between predicted and ground-truth coordinates.

Saves one txt results file per model:
  outputs/results_base_gptoss.txt
  outputs/results_finetuned_gptoss.txt

Run from Geospatial Knowledge/:
    python analyze_gptoss_coords.py
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

PROMPT_DESCRIPTIONS = {
    "type0": "\"What are the geo-coordinates of [City]?\"",
    "type1": "\"What are the latitude and longitude of [City]?\"",
}

MODELS = {
    "Base GPT-OSS": {
        "tag": "gptoss-base",
        "out": OUTPUTS_DIR / "results_base_gptoss.txt",
    },
    "Fine-tuned GPT-OSS (LoRA)": {
        "tag": "gptoss-lora",
        "out": OUTPUTS_DIR / "results_finetuned_gptoss.txt",
    },
}

CONFIGS = [
    ("type0", "zero-shot"),
    ("type0", "3-shot"),
    ("type1", "zero-shot"),
    ("type1", "3-shot"),
]

# Error thresholds for accuracy buckets
THRESHOLDS_KM = [10, 50, 100, 250, 500]


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE PARSING
# ─────────────────────────────────────────────────────────────────────────────
_COORD_RE = re.compile(
    r"([+-]?\d{1,3}(?:\.\d+)?)\s*,\s*([+-]?\d{1,3}(?:\.\d+)?)"
)


def parse_coords(text: str):
    """
    Extract (lat, lon) from model output string.
    Returns (lat, lon) or (None, None) on failure.
    """
    if not text:
        return None, None
    m = _COORD_RE.search(str(text).strip())
    if not m:
        return None, None
    try:
        lat = float(m.group(1))
        lon = float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except ValueError:
        pass
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# HAVERSINE DISTANCE
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2) -> float:
    """Compute the Haversine great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADER
# ─────────────────────────────────────────────────────────────────────────────
def find_file(model_tag: str, p_type: str, shots: str) -> Path | None:
    """Return the latest matching results JSON file for the given model/type/shots combo."""
    # files are timestamped; glob by model/type/shots pattern
    pattern = f"results_{model_tag}_{shots}_{p_type}_*.json"
    matches = sorted(OUTPUTS_DIR.glob(pattern))
    return matches[-1] if matches else None  # latest if multiple


def load_and_eval(model_tag: str, p_type: str, shots: str) -> dict | None:
    """Load a results file and compute distance-error statistics for one configuration."""
    path = find_file(model_tag, p_type, shots)
    if path is None:
        return None

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    records = raw["data"] if isinstance(raw, dict) else raw

    errors_km = []
    lat_errors, lon_errors = [], []
    failed = 0

    for r in records:
        true_lat = r.get("Latitude")
        true_lon = r.get("Longitude")
        pred_lat, pred_lon = parse_coords(r.get("output", ""))

        if pred_lat is None or true_lat is None:
            failed += 1
            continue

        dist = haversine(true_lat, true_lon, pred_lat, pred_lon)
        errors_km.append(dist)
        lat_errors.append(abs(pred_lat - true_lat))
        lon_errors.append(abs(pred_lon - true_lon))

    n_total = len(records)
    n_valid = len(errors_km)

    if n_valid == 0:
        return {
            "n": n_total, "n_valid": 0, "n_failed": failed,
            "mae_km": float("nan"), "rmse_km": float("nan"),
            "median_km": float("nan"), "min_km": float("nan"),
            "max_km": float("nan"), "std_km": float("nan"),
            "mae_lat": float("nan"), "mae_lon": float("nan"),
            "thresholds": {}, "path": path.name,
        }

    e = np.array(errors_km)
    return {
        "n":         n_total,
        "n_valid":   n_valid,
        "n_failed":  failed,
        "mae_km":    float(np.mean(e)),
        "rmse_km":   float(np.sqrt(np.mean(e ** 2))),
        "median_km": float(np.median(e)),
        "min_km":    float(np.min(e)),
        "max_km":    float(np.max(e)),
        "std_km":    float(np.std(e)),
        "mae_lat":   float(np.mean(lat_errors)),
        "mae_lon":   float(np.mean(lon_errors)),
        "thresholds": {t: int((e <= t).sum()) for t in THRESHOLDS_KM},
        "_errors":   e,
        "path":      path.name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def nan_fmt(val, fmt_spec="9.1f") -> str:
    """Format a float with the given spec, substituting an em-dash for NaN values."""
    if math.isnan(val):
        return f"{'—':>9}"
    return f"{val:{fmt_spec}}"


def build_report(model_name: str, model_tag: str) -> str:
    """Build a full text report of coordinate prediction results for one model."""
    SEP  = "═" * 80
    DASH = "─" * 80

    results = []
    for p_type, shots in CONFIGS:
        m = load_and_eval(model_tag, p_type, shots)
        results.append((p_type, shots, m))

    # ── header ────────────────────────────────────────────────────────────────
    lines = [
        SEP,
        f"  Coordinate Prediction Results — {model_name}",
        f"  Task   : Predict (latitude, longitude) for world cities",
        f"  Metric : Haversine distance (km) vs ground-truth coordinates",
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
        f"  {'Config':<18} {'N':>5} {'Valid':>6}  {'MAE km':>9}  {'RMSE km':>9}"
        f"  {'Median km':>10}  {'MAE lat°':>9}  {'MAE lon°':>9}",
        "  " + "─" * 76,
    ]

    valid_rows = [(pt, sh, m) for pt, sh, m in results if m and not math.isnan(m["mae_km"])]

    for p_type, shots, m in results:
        cfg = f"{p_type.upper()} {shots}"
        if m is None:
            lines.append(f"  {cfg:<18}  — file not found")
            continue
        pct = m["n_valid"] / m["n"] * 100 if m["n"] > 0 else 0
        lines.append(
            f"  {cfg:<18} {m['n']:>5} {m['n_valid']:>5} ({pct:>4.0f}%)"
            f"  {nan_fmt(m['mae_km']):>9}"
            f"  {nan_fmt(m['rmse_km']):>9}"
            f"  {nan_fmt(m['median_km']):>10}"
            f"  {nan_fmt(m['mae_lat'], '9.4f'):>9}"
            f"  {nan_fmt(m['mae_lon'], '9.4f'):>9}"
        )

    # ── best config ───────────────────────────────────────────────────────────
    if valid_rows:
        best = min(valid_rows, key=lambda x: x[2]["mae_km"])
        b_pt, b_sh, b_m = best
        lines += [
            "  " + "─" * 76,
            f"  ◀ Best config by MAE : {b_pt.upper()} {b_sh}  "
            f"(MAE={b_m['mae_km']:.1f} km  |  Median={b_m['median_km']:.1f} km  |  "
            f"RMSE={b_m['rmse_km']:.1f} km)",
        ]

    # ── accuracy-at-threshold table ───────────────────────────────────────────
    if valid_rows:
        lines += [
            "",
            DASH,
            f"  Accuracy at Distance Thresholds  (% of cities within X km)",
            f"  {'Config':<18}  "
            + "  ".join(f"≤{t:>4}km" for t in THRESHOLDS_KM),
            "  " + "─" * 76,
        ]
        for p_type, shots, m in results:
            cfg = f"{p_type.upper()} {shots}"
            if not m or math.isnan(m["mae_km"]):
                continue
            pcts = [
                f"{m['thresholds'][t] / m['n_valid'] * 100:>6.1f}%"
                for t in THRESHOLDS_KM
            ]
            lines.append(f"  {cfg:<18}  " + "   ".join(pcts))

    # ── error distribution for best config ───────────────────────────────────
    if valid_rows:
        errors = b_m.get("_errors")
        if errors is not None and len(errors):
            lines += [
                "",
                DASH,
                f"  Error Distribution — Best Config ({b_pt.upper()} {b_sh})",
                "  " + "─" * 76,
            ]
            buckets = [
                ("<   10 km",  errors <   10),
                ("10–50 km",  (errors >=  10) & (errors <   50)),
                ("50–100 km", (errors >=  50) & (errors <  100)),
                ("100–250 km",(errors >= 100) & (errors <  250)),
                ("250–500 km",(errors >= 250) & (errors <  500)),
                ("> 500 km",   errors >= 500),
            ]
            for label, mask in buckets:
                count = int(mask.sum())
                pct   = count / len(errors) * 100
                bar   = "█" * int(pct / 2.5)
                lines.append(f"  {label:<14}  {count:>5} ({pct:>5.1f}%)  {bar}")

            lines += [
                "",
                f"  Min error : {errors.min():.2f} km",
                f"  Max error : {errors.max():.2f} km",
                f"  Std dev   : {errors.std():.2f} km",
            ]

    lines += ["", SEP]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Generate and save text reports for all configured models."""
    for model_name, cfg in MODELS.items():
        report = build_report(model_name, cfg["tag"])
        cfg["out"].write_text(report, encoding="utf-8")
        print(f"Saved → {cfg['out'].name}")
        print(report)


if __name__ == "__main__":
    main()
