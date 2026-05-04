#!/usr/bin/env python3
"""
Model Evaluation Script: Qwen 14B & GPT-OSS 20B
Visualizations: Accuracy/F1 Lines, Confusion Matrices (Best Temp only), and Global Heatmap.
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

# ─────────────────────── CONFIGURATION ───────────────────────
GROUND_TRUTH   = "evaluation_results/answers_eval_subset.jsonl"
OUTPUT_DIR     = "evaluation_results"
MAX_IDS        = 1400

VALID_DIRECTIONS = [
    "north", "south", "east", "west",
    "north-east", "north-west", "south-east", "south-west"
]

MODELS_CONFIG = {
    "qwen_base_0shot":   {"file": "experiment_results_qwen14b_base_0shot.jsonl",      "label": "Qwen Base 0-shot"},
    "qwen_base_3shot":   {"file": "experiment_results_qwen14b_base_3shot.jsonl",      "label": "Qwen Base 3-shot"},
    "qwen_ft_0shot":     {"file": "experiment_results_qwen14b_finetuned_0shot.jsonl", "label": "Qwen FT 0-shot"},
    "qwen_ft_3shot":     {"file": "experiment_results_qwen14b_finetuned_3shot.jsonl", "label": "Qwen FT 3-shot"},
    "gptoss_base_0shot": {"file": "experiment_results_gptoss20b_base_0shot.jsonl",      "label": "GPT-OSS Base 0-shot"},
    "gptoss_base_3shot": {"file": "experiment_results_gptoss20b_base_3shot.jsonl",      "label": "GPT-OSS Base 3-shot"},
    "gptoss_ft_0shot":   {"file": "experiment_results_gptoss20b_finetuned_0shot.jsonl", "label": "GPT-OSS FT 0-shot"},
    "gptoss_ft_3shot":   {"file": "experiment_results_gptoss20b_finetuned_3shot.jsonl", "label": "GPT-OSS FT 3-shot"},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# HELPERS & EXTRACTION
# ══════════════════════════════════════════════════════════════

def load_ground_truth(max_ids=MAX_IDS):
    """Loads ground-truth direction labels from the evaluation JSONL file."""
    gt = {}
    if not os.path.exists(GROUND_TRUTH): return gt
    with open(GROUND_TRUTH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            gt[str(obj["id"])] = obj.get("absoluteAnswer", "").lower().strip()
    return gt

def load_model_results(filepath):
    """Loads model prediction results grouped by question ID and temperature."""
    results, temps = defaultdict(dict), set()
    if not os.path.exists(filepath): return results, []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q_id, temp = str(obj["request"]["id"]), float(obj["request"]["temperature"])
            results[q_id][temp] = obj["response"]["content"][0]["text"].lower().strip()
            temps.add(temp)
    return results, sorted(temps)

def extract_direction(text):
    """Extracts and normalizes a cardinal direction string from raw model output."""
    text = text.lower().strip()
    for v, c in [("northeast", "north-east"), ("northwest", "north-west"),
                 ("southeast", "south-east"), ("southwest", "south-west")]:
        text = text.replace(v, c)
    
    if "assistantfinal" in text:
        final_part = text.split("assistantfinal")[-1].strip().split("\n")[0]
        candidate = re.sub(r'[^a-z\-]', '', final_part)
        if candidate in VALID_DIRECTIONS: return candidate

    pattern = r'\b(' + '|'.join(VALID_DIRECTIONS) + r')\b'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else "invalid_format"

# ══════════════════════════════════════════════════════════════
# NEW PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, model_label, temp, out_dir):
    """Plots and saves a confusion matrix heatmap for a model at its best temperature."""
    labels = VALID_DIRECTIONS + ["invalid_format"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    # Updated title to explicitly state it's the Best Temp
    plt.title(f"Confusion Matrix: {model_label}\n(Best Temp: {temp})")
    plt.ylabel('Actual Direction')
    plt.xlabel('Predicted Direction')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/CM_BEST_{model_label.replace(' ', '_')}.png", dpi=300)
    plt.close()

def plot_global_heatmap(all_data, metric="accuracy"):
    """Creates a Heatmap of Models vs Temperatures"""
    df_data = []
    for model_key, temps_dict in all_data.items():
        for t, m in temps_dict.items():
            df_data.append({"Model": MODELS_CONFIG[model_key]["label"], "Temp": t, "Score": m[metric]})
    
    df = pd.DataFrame(df_data).pivot(index="Model", columns="Temp", values="Score")
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title(f"Global Model Stability - {metric.capitalize()} Across Temperatures")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/GLOBAL_Stability_Heatmap.png", dpi=300)
    plt.close()

# ══════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════

def main():
    """Evaluates all configured models, plots confusion matrices, and generates a global heatmap."""
    gt = load_ground_truth()
    all_metrics = {}

    for key, info in MODELS_CONFIG.items():
        file_path = os.path.join(OUTPUT_DIR, info["file"])
        results, temps = load_model_results(file_path)
        if not results: continue

        metrics_by_temp = {}
        for temp in temps:
            y_true, y_pred = [], []
            for q_id, gt_raw in gt.items():
                if q_id in results and temp in results[q_id]:
                    y_true.append(gt_raw)
                    y_pred.append(extract_direction(results[q_id][temp]))

            if y_true:
                metrics_by_temp[temp] = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                    "y_true": y_true, "y_pred": y_pred
                }

        all_metrics[key] = metrics_by_temp

        # --- MODIFIED SECTION: Evaluate best temp and plot CM ---
        if metrics_by_temp:
            # Find the temperature that produced the highest accuracy
            best_temp = max(metrics_by_temp.keys(), key=lambda t: metrics_by_temp[t]["accuracy"])
            
            # Extract the actuals and predictions for that specific best temperature
            best_y_true = metrics_by_temp[best_temp]["y_true"]
            best_y_pred = metrics_by_temp[best_temp]["y_pred"]
            
            # Plot the matrix
            plot_confusion_matrix(best_y_true, best_y_pred, info["label"], best_temp, OUTPUT_DIR)
            print(f"  -> Generated Best Temp CM for {info['label']} (Temp: {best_temp})")

    # Generate Global Heatmap
    if all_metrics:
        plot_global_heatmap(all_metrics, "accuracy")
    
    print(f"\n✅ Evaluation Complete. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()