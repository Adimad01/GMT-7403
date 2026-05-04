import json
import os
import re
import numpy as np
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────
OUTPUTS_DIR            = r"outputs"
REPORTS_DIR            = r"results/reports"
DEBUG_SAMPLES_PER_FILE = 5
MAX_EARTH_DISTANCE_KM  = 20_100   # hard cap — no two points on Earth are farther apart

# Regex to capture: gen_dis_{model}_{p_type}_{shots}[_optional_suffix].json
FILE_PATTERN = re.compile(r"gen_dis_(.+?)_(p\d+)_(zero-shot|\d+-shot).*?\.json")
# ──────────────────────────────────────────────────────────────────────────────

def clean_number_string(s):
    """Strip commas, spaces, and non-breaking space characters from a numeric string."""
    if s is None:
        return ""
    return re.sub(r"[,\s\u00a0\u202f]", "", s)


def extract_distance(text):
    """
    Extract a plausible distance from model output.
    Strategy 1 : first number followed by km/kilometers AND within Earth range.
    Strategy 2 : first numeric token in [50, MAX_EARTH_DISTANCE_KM].
    Strategy 3 : last numeric token (fallback), capped to Earth range.
    Returns (value, raw_token, cleaned_token, strategy) or (None, ...) if nothing valid.
    """
    if not text:
        return None, None, None, None
    if isinstance(text, list):
        text = " ".join(text)

    unit_pattern   = re.compile(r"([\d][\d,\s\u00a0\u202f]*\.?[\d]*)\s*(?:km|kilometers)", re.IGNORECASE)
    number_pattern = re.compile(r"([\d][\d,\s\u00a0\u202f]*\.?[\d]*)")

    # Strategy 1: number + unit, within Earth range
    for raw in unit_pattern.findall(text):
        cleaned = clean_number_string(raw)
        try:
            val = float(cleaned)
            if 0 < val <= MAX_EARTH_DISTANCE_KM:
                return val, raw, cleaned, "unit"
        except ValueError:
            pass

    # Strategy 2 & 3: scan all numbers
    all_numbers = number_pattern.findall(text)
    if all_numbers:
        # Strategy 2: first number in plausible distance range
        for raw in all_numbers:
            cleaned = clean_number_string(raw)
            if not cleaned:
                continue
            try:
                val = float(cleaned)
                if 50 <= val <= MAX_EARTH_DISTANCE_KM:
                    return val, raw, cleaned, "first>=50"
            except ValueError:
                pass

        # Strategy 3: last number, only if within Earth range
        for raw in reversed(all_numbers):
            cleaned = clean_number_string(raw)
            if not cleaned:
                continue
            try:
                val = float(cleaned)
                if 0 < val <= MAX_EARTH_DISTANCE_KM:
                    return val, raw, cleaned, "fallback_last"
            except ValueError:
                pass

    return None, None, None, None


def evaluate_model(file_path, output_dir, model_name, p_type, shots):
    """Evaluate distance predictions from a JSON output file and save a text report."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    print(f"Loading predictions from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    absolute_errors   = []
    valid_predictions = 0
    failed_extraction = 0
    total_samples     = len(data)
    debug_logs        = []

    for i, item in enumerate(data):
        ground_truth = item.get('distance')
        raw_output   = item.get('output', "")
        output_text  = raw_output[0] if isinstance(raw_output, list) else str(raw_output)

        predicted_val, raw_tok, cleaned_tok, strategy = extract_distance(output_text)

        # Fallback to predicted_dis column if output was empty
        if predicted_val is None and (not output_text or not output_text.strip()):
            fallback_pred = item.get('predicted_dis')
            if fallback_pred is not None:
                val = float(fallback_pred)
                if 0 < val <= MAX_EARTH_DISTANCE_KM:
                    predicted_val = val
                    raw_tok       = str(fallback_pred)
                    cleaned_tok   = clean_number_string(raw_tok)
                    strategy      = "fallback_predicted_dis"

        if predicted_val is not None:
            error = abs(predicted_val - ground_truth)
            absolute_errors.append(error)
            valid_predictions += 1
            if len(debug_logs) < DEBUG_SAMPLES_PER_FILE:
                snippet = output_text.strip().replace("\n", " ")
                snippet = snippet[:180] + ("..." if len(snippet) > 180 else "")
                debug_logs.append(
                    f"[{i}] strategy={strategy} raw='{raw_tok}' cleaned='{cleaned_tok}' "
                    f"-> {predicted_val} | snippet: {snippet}"
                )
        else:
            failed_extraction += 1

    p_rate       = (valid_predictions / total_samples) * 100 if total_samples > 0 else 0
    mean_error   = np.mean(absolute_errors)                          if absolute_errors else 0
    median_error = np.median(absolute_errors)                        if absolute_errors else 0
    rmse         = np.sqrt(np.mean(np.array(absolute_errors) ** 2)) if absolute_errors else 0

    base_name    = os.path.basename(file_path)
    timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "=" * 60,
        f"  EVALUATION REPORT — {model_name.upper()}",
        "=" * 60,
        f"  File Name          : {base_name}",
        f"  Prompt Type        : {p_type.upper()}",
        f"  Shot Setting       : {shots}",
        f"  Evaluation Time    : {timestamp}",
        "-" * 60,
        f"  Total Samples      : {total_samples}",
        f"  Valid Predictions  : {valid_predictions}",
        f"  Failed Extractions : {failed_extraction}",
        f"  Prediction Rate    : {p_rate:.2f}%",
        "-" * 60,
        f"  Mean  Abs Error    : {mean_error:.2f} km",
        f"  RMSE               : {rmse:.2f} km",
        "=" * 60,
    ]

    if debug_logs:
        report_lines += ["", "DEBUG SAMPLES (extracted distances)", "-" * 60] + debug_logs

    report_content = "\n".join(report_lines)
    
    # Save text report
    output_path = os.path.join(output_dir, f"report_{os.path.splitext(base_name)[0]}.txt")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except Exception as e:
        print(f"\n>> Error saving report: {e}\n")

    return {
        "file": base_name, 
        "model": model_name,
        "p_type": p_type, 
        "shots": shots,
        "total": total_samples, 
        "valid": valid_predictions,
        "failed": failed_extraction,
        "prediction_rate": p_rate,
        "mean_error": mean_error, 
        "median_error": median_error, 
        "rmse": rmse,
    }


def discover_output_files(directory):
    """Dynamically scan directory for any valid generated distance JSON files."""
    if not os.path.exists(directory):
        return []
    
    found = []
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith(".json"):
            continue
        
        match = FILE_PATTERN.match(fname)
        if match:
            model_name, p_type, shots = match.groups()
            found.append((os.path.join(directory, fname), model_name, p_type, shots))
            
    return found


def print_unified_summary(summary):
    """Prints a single unified table grouped by model name, sorted by RMSE."""
    if not summary:
        print("No evaluation data to summarize.")
        return

    print("\n" + "=" * 90)
    print(f"  COMPARISON SUMMARY — ALL MODELS  (Sorted by Model, then RMSE)")
    print("=" * 90)
    print(f"  {'Model Name':<20} {'P-Type':<8} {'Shots':<12} {'Rate':>6}   {'Mean Err':>12}   {'RMSE':>12}")
    print("-" * 90)

    # Find unique models to group them nicely
    models = sorted(list(set(m["model"] for m in summary)))
    
    for current_model in models:
        # Filter and sort entries for this specific model
        group = sorted(
            [m for m in summary if m["model"] == current_model],
            key=lambda x: x["rmse"]   # best (lowest RMSE) first
        )
        
        for m in group:
            print(f"  {m['model']:<20} {m['p_type'].upper():<8} {m['shots']:<12} "
                  f"{m['prediction_rate']:>5.1f}%  "
                  f"{m['mean_error']:>10.1f}km  "
                  f"{m['rmse']:>10.1f}km")
        
        # Print a subtle divider between different models
        print("  " + "-" * 88)
        
    print("=" * 90)


if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)

    all_files = discover_output_files(OUTPUTS_DIR)

    if not all_files:
        print(f"No valid output files found in '{OUTPUTS_DIR}'. Make sure filenames start with 'gen_dis_'")
    else:
        print(f"Found {len(all_files)} output file(s) to evaluate.\n")

        summary = []
        for path, model_name, p_type, shots in all_files:
            m = evaluate_model(path, REPORTS_DIR, model_name, p_type, shots)
            if m:
                summary.append(m)

        # Print the final unified table
        print_unified_summary(summary)