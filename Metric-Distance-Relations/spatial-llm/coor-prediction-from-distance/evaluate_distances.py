import json
import os
import re
import numpy as np
from datetime import datetime

# Configuration
OUTPUTS_DIR = r"outputs"
RESULTS_DIR = r"results"
FILENAME_PREFIX = "gen_dis_"  # evaluate only generated distance files
DEBUG_SAMPLES_PER_FILE = 5      # how many extracted values to print for inspection

def clean_number_string(s):
    """Normalize number tokens (strip commas, spaces, NBSP/thin spaces)."""
    if s is None:
        return ""
    return re.sub(r"[,\s\u00a0\u202f]", "", s)

def extract_distance(text):
    """
    Extract distance plus trace info from model output.
    Strategy 1: first number followed by km/kilometers.
    Strategy 2: first numeric token >= 50 (likely distance), else last numeric token.
    Returns (value, raw_token, cleaned_token, strategy) or (None, None, None, None).
    """
    if not text:
        return None, None, None, None

    if isinstance(text, list):
        text = " ".join(text)

    unit_pattern = re.compile(r"([\d][\d,\s\u00a0\u202f]*\.?[\d]*)\s*(?:km|kilometers)", re.IGNORECASE)
    number_pattern = re.compile(r"([\d][\d,\s\u00a0\u202f]*\.?[\d]*)")

    unit_matches = unit_pattern.findall(text)
    if unit_matches:
        raw = unit_matches[0]
        cleaned = clean_number_string(raw)
        try:
            return float(cleaned), raw, cleaned, "unit"
        except ValueError:
            pass

    all_numbers = number_pattern.findall(text)
    if all_numbers:
        for raw in all_numbers:
            cleaned = clean_number_string(raw)
            if not cleaned:
                continue
            try:
                val = float(cleaned)
            except ValueError:
                continue
            if val >= 50:  # likely a distance, not a lat/lon
                return val, raw, cleaned, "first>=50"
        raw = all_numbers[-1]
        cleaned = clean_number_string(raw)
        try:
            return float(cleaned), raw, cleaned, "fallback_last"
        except ValueError:
            pass

    return None, None, None, None

def evaluate_model(file_path, output_dir):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Loading predictions from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    absolute_errors = []
    valid_predictions = 0
    total_samples = len(data)
    debug_logs = []
    
    # Process samples
    for i, item in enumerate(data):
        ground_truth = item.get('distance')

        # Handle both string outputs and list outputs gracefully
        raw_output = item.get('output', "")
        if isinstance(raw_output, list):
            output_text = raw_output[0] if raw_output else ""
        else:
            output_text = str(raw_output)

        predicted_val, raw_tok, cleaned_tok, strategy = extract_distance(output_text)

        # Fallback: if output is empty or extraction failed, use predicted_dis when available
        if predicted_val is None and (not output_text or not output_text.strip()):
            fallback_pred = item.get('predicted_dis')
            if fallback_pred is not None:
                predicted_val = float(fallback_pred)
                raw_tok = str(fallback_pred)
                cleaned_tok = clean_number_string(raw_tok)
                strategy = "fallback_predicted_dis"

        if predicted_val is not None:
            error = abs(predicted_val - ground_truth)
            absolute_errors.append(error)
            valid_predictions += 1
            if len(debug_logs) < DEBUG_SAMPLES_PER_FILE:
                snippet = output_text.strip().replace("\n", " ")
                snippet = snippet[:180] + ("..." if len(snippet) > 180 else "")
                debug_logs.append(
                    f"[{i}] strategy={strategy} raw='{raw_tok}' cleaned='{cleaned_tok}' "
                    f"-> {predicted_val} | output_snippet: {snippet}"
                )

    # Metrics Calculation
    p_rate = (valid_predictions / total_samples) * 100 if total_samples > 0 else 0
    mean_error = np.mean(absolute_errors) if absolute_errors else 0
    median_error = np.median(absolute_errors) if absolute_errors else 0
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Construct Report String
    report_lines = []
    report_lines.append("="*50)
    report_lines.append(f" EVALUATION REPORT")
    report_lines.append("="*50)
    report_lines.append(f"File Name        : {os.path.basename(file_path)}")
    report_lines.append(f"Evaluation Time  : {timestamp}")
    report_lines.append("-" * 50)
    report_lines.append(f"Total Samples    : {total_samples}")
    report_lines.append(f"Valid Predictions: {valid_predictions}")
    report_lines.append(f"Prediction Rate  : {p_rate:.2f}%")
    report_lines.append("-" * 50)
    report_lines.append(f"Mean Error Dist  : {mean_error:.2f} km")
    report_lines.append(f"Median Error Dist: {median_error:.2f} km")
    report_lines.append("="*50)

    if debug_logs:
        report_lines.append("")
        report_lines.append("DEBUG SAMPLES (extracted distances)")
        report_lines.append("-" * 50)
        report_lines.extend(debug_logs)
    
    report_content = "\n".join(report_lines)

    # 1. Print to Console
    print(report_content)

    # 2. Save to File
    base_name = os.path.basename(file_path)
    file_root, _ = os.path.splitext(base_name)
    report_filename = f"report_{file_root}.txt"
    output_path = os.path.join(output_dir, report_filename)
    
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n>> Report successfully saved to: {output_path}\n")
    except Exception as e:
        print(f"\n>> Error saving report: {e}\n")

    return {
        "file": os.path.basename(file_path),
        "total": total_samples,
        "valid": valid_predictions,
        "prediction_rate": p_rate,
        "mean_error": mean_error,
        "median_error": median_error,
    }


def discover_output_files(directory, prefix=FILENAME_PREFIX):
    if not os.path.exists(directory):
        return []
    return [
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.startswith(prefix) and fname.lower().endswith(".json")
    ]

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory: {RESULTS_DIR}")

    json_files = discover_output_files(OUTPUTS_DIR)

    if not json_files:
        print(f"No files found in {OUTPUTS_DIR} with prefix '{FILENAME_PREFIX}'")
    else:
        summary = []
        for path in json_files:
            metrics = evaluate_model(path, RESULTS_DIR)
            if metrics:
                summary.append(metrics)

        if summary:
            print("\nSUMMARY")
            print("=" * 50)
            for m in summary:
                print(f"{m['file']}: total={m['total']}, valid={m['valid']}, "
                      f"rate={m['prediction_rate']:.2f}%, "
                      f"mean_err={m['mean_error']:.2f} km, "
                      f"median_err={m['median_error']:.2f} km")