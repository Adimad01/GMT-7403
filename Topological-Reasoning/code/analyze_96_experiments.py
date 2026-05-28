"""
analyze_96_experiments.py
================================================================================
Unified comparison of all 6 experiments on the 96 balanced test examples.

Loads results from:
  Exp 1 — GPTOSS Base                               → kg_eval_gptoss_base_96_none.csv
  Exp 2 — GPTOSS Fine-tuné                           → kg_eval_gptoss_finetuned_96_none.csv
  Exp 3 — GPTOSS Fine-tuné + KG en entrée           → kg_eval_gptoss_finetuned_kg_input_96_osm.csv
  Exp 4 — GPTOSS Fine-tuné + KG ft                  → kg_eval_gptoss_osm_kg_ft_96_osm.csv
  Exp 5 — GPTOSS Fine-tuné + Enriched inference     → voletc_dynamic_osm_finetuned_gpu_*_ckpt.json
  Exp 6 — GPTOSS + Enriched inference               → voletc_dynamic_osm_improved_version_*_ckpt.json

Usage:
  cd /path/to/Topological-Reasoning/code
  python analyze_96_experiments.py [--results-dir results] [--output-dir results]
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

VALID_PREDICATES = ["contains", "within", "touches", "crosses", "disjoint", "overlaps"]
INDICES_FILE = "../dataset/eval_96_balanced_indices.json"

# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------

def _load_indices(path: str):
    with open(path) as f:
        return set(json.load(f))


def load_csv_results(csv_path: str, indices: set = None) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if indices and "index" in df.columns:
        df = df[df["index"].isin(indices)]
    return df


def load_ckpt_results(ckpt_path: str) -> pd.DataFrame:
    if not os.path.exists(ckpt_path):
        return None
    with open(ckpt_path) as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        return None
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, label: str) -> dict:
    if df is None or len(df) == 0:
        return {"label": label, "n": 0, "accuracy": None, "per_predicate": {}}

    df = df.copy()
    df["match"] = df["match"].astype(str).str.lower().map({"true": True, "false": False})
    overall_acc = df["match"].mean()

    per_pred = {}
    for pred in VALID_PREDICATES:
        sub = df[df["expected"] == pred]
        if len(sub) > 0:
            per_pred[pred] = sub["match"].mean()
        else:
            per_pred[pred] = None

    return {
        "label": label,
        "n": len(df),
        "accuracy": overall_acc,
        "per_predicate": per_pred,
    }


# ---------------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------------

def print_comparison_table(experiments: list):
    print("\n" + "=" * 100)
    print(f"  RESULTS — 96 BALANCED TEST EXAMPLES (16 per predicate, OSM KG)")
    print("=" * 100)

    header = f"{'Experiment':<52} {'N':>4} {'Accuracy':>10}  " + \
             "  ".join(f"{p[:7]:>7}" for p in VALID_PREDICATES)
    print(header)
    print("-" * 100)

    for exp in experiments:
        if exp["accuracy"] is None:
            acc_str = "   N/A"
        else:
            acc_str = f"{exp['accuracy'] * 100:>9.1f}%"

        pred_cols = ""
        for p in VALID_PREDICATES:
            v = exp["per_predicate"].get(p)
            if v is None:
                pred_cols += "      N/A"
            else:
                pred_cols += f"  {v * 100:>6.1f}%"

        print(f"{exp['label']:<52} {exp['n']:>4} {acc_str}  {pred_cols}")

    print("=" * 100)


# ---------------------------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------------------------

def plot_confusion_matrix(df: pd.DataFrame, title: str, save_path: str):
    if df is None or len(df) == 0:
        return
    df = df.copy()
    df["match"] = df["match"].astype(str).str.lower().map({"true": True, "false": False})
    matrix = np.zeros((len(VALID_PREDICATES), len(VALID_PREDICATES)), dtype=int)
    pred_idx = {p: i for i, p in enumerate(VALID_PREDICATES)}

    for _, row in df.iterrows():
        e = str(row.get("expected", "")).lower().strip()
        p = str(row.get("predicted", "")).lower().strip()
        if e in pred_idx and p in pred_idx:
            matrix[pred_idx[e]][pred_idx[p]] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(VALID_PREDICATES)))
    ax.set_yticks(range(len(VALID_PREDICATES)))
    ax.set_xticklabels(VALID_PREDICATES, rotation=45, ha="right")
    ax.set_yticklabels(VALID_PREDICATES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10, pad=10)

    thresh = matrix.max() / 2
    for i in range(len(VALID_PREDICATES)):
        for j in range(len(VALID_PREDICATES)):
            ax.text(j, i, str(matrix[i][j]),
                    ha="center", va="center",
                    color="white" if matrix[i][j] > thresh else "black",
                    fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Saved: {save_path}")


# ---------------------------------------------------------------------------
# BAR CHART
# ---------------------------------------------------------------------------

def plot_accuracy_bar(experiments: list, save_path: str):
    labels = [e["label"] for e in experiments if e["accuracy"] is not None]
    accs   = [e["accuracy"] * 100 for e in experiments if e["accuracy"] is not None]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    bars = ax.bar(range(len(labels)), accs, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Overall Accuracy — 6 Experiments on 96 Balanced Examples (OSM KG)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Saved: {save_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir",  default="results")
    args = parser.parse_args()

    rd = args.results_dir
    od = args.output_dir
    os.makedirs(od, exist_ok=True)

    indices = None
    if os.path.exists(INDICES_FILE):
        indices = _load_indices(INDICES_FILE)

    # ----------------------------------------------------------------
    # Experiments 1-4 — CSV format
    # ----------------------------------------------------------------
    csv_specs = [
        ("1 — GPTOSS Base",                     "kg_eval_gptoss_base_96_none.csv"),
        ("2 — GPTOSS Fine-tuné",                 "kg_eval_gptoss_finetuned_96_none.csv"),
        ("3 — Fine-tuné + KG en entrée",         "kg_eval_gptoss_finetuned_kg_input_96_osm.csv"),
        ("4 — Fine-tuné + KG ft (OSM)",          "kg_eval_gptoss_osm_kg_ft_96_osm.csv"),
    ]

    experiments = []
    for label, fname in csv_specs:
        path = os.path.join(rd, fname)
        df = load_csv_results(path, indices)
        if df is None:
            print(f"[MISSING] {path}")
        else:
            print(f"[OK] {path}  ({len(df)} rows)")
        exp = compute_metrics(df, label)
        experiments.append(exp)

    # ----------------------------------------------------------------
    # Experiment 5 — GPU enriched inference (fine-tuned)
    # ----------------------------------------------------------------
    tag5 = "dynamic_osm_finetuned_gpu"
    strategies_exp5 = {}
    for strat in ("cot", "tot", "got"):
        ckpt = os.path.join(rd, f"voletc_{tag5}_{strat}_neighborhood_details_spatial_relation_16_sample_ckpt.json")
        df_s = load_ckpt_results(ckpt)
        if df_s is None:
            print(f"[MISSING] {ckpt}")
        else:
            print(f"[OK] {ckpt}  ({len(df_s)} rows)")
        strategies_exp5[strat] = df_s

    for strat, df_s in strategies_exp5.items():
        label = f"5 — Fine-tuné + Enriched ({strat.upper()}, GPU)"
        exp = compute_metrics(df_s, label)
        experiments.append(exp)

    # ----------------------------------------------------------------
    # Experiment 6 — Ollama enriched inference (base model)
    # ----------------------------------------------------------------
    tag6 = "dynamic_osm_improved_version"
    strategies_exp6 = {}
    for strat in ("cot", "tot", "got"):
        ckpt = os.path.join(rd, f"voletc_{tag6}_{strat}_neighborhood_details_spatial_relation_16_sample_ckpt.json")
        df_s = load_ckpt_results(ckpt)
        if df_s is None:
            print(f"[MISSING] {ckpt}")
        else:
            print(f"[OK] {ckpt}  ({len(df_s)} rows)")
        strategies_exp6[strat] = df_s

    for strat, df_s in strategies_exp6.items():
        label = f"6 — GPTOSS + Enriched ({strat.upper()}, Ollama)"
        exp = compute_metrics(df_s, label)
        experiments.append(exp)

    # ----------------------------------------------------------------
    # Print table
    # ----------------------------------------------------------------
    print_comparison_table(experiments)

    # ----------------------------------------------------------------
    # Confusion matrices for complete experiments
    # ----------------------------------------------------------------
    print("\nGenerating confusion matrices...")

    cm_configs = [
        ("1 — GPTOSS Base",                     load_csv_results(os.path.join(rd, "kg_eval_gptoss_base_96_none.csv"), indices),                          "cm_96_exp1_base"),
        ("2 — GPTOSS Fine-tuné",                 load_csv_results(os.path.join(rd, "kg_eval_gptoss_finetuned_96_none.csv"), indices),                     "cm_96_exp2_finetuned"),
        ("3 — Fine-tuné + KG en entrée",         load_csv_results(os.path.join(rd, "kg_eval_gptoss_finetuned_kg_input_96_osm.csv"), indices),             "cm_96_exp3_ft_kg_input"),
        ("4 — Fine-tuné + KG ft (OSM)",          load_csv_results(os.path.join(rd, "kg_eval_gptoss_osm_kg_ft_96_osm.csv"), indices),                      "cm_96_exp4_osm_kg_ft"),
        ("5 — Fine-tuné + CoT GPU",              strategies_exp5.get("cot"),                                                                               "cm_96_exp5_ft_cot_gpu"),
        ("5 — Fine-tuné + ToT GPU",              strategies_exp5.get("tot"),                                                                               "cm_96_exp5_ft_tot_gpu"),
        ("5 — Fine-tuné + GoT GPU",              strategies_exp5.get("got"),                                                                               "cm_96_exp5_ft_got_gpu"),
        ("6 — GPTOSS + CoT (Ollama)",            strategies_exp6.get("cot"),                                                                               "cm_96_exp6_cot_ollama"),
        ("6 — GPTOSS + ToT (Ollama)",            strategies_exp6.get("tot"),                                                                               "cm_96_exp6_tot_ollama"),
        ("6 — GPTOSS + GoT (Ollama)",            strategies_exp6.get("got"),                                                                               "cm_96_exp6_got_ollama"),
    ]

    for title, df_cm, fname in cm_configs:
        if df_cm is not None and len(df_cm) > 0:
            plot_confusion_matrix(df_cm, title, os.path.join(od, f"{fname}.png"))

    # ----------------------------------------------------------------
    # Overall accuracy bar chart
    # ----------------------------------------------------------------
    print("\nGenerating accuracy bar chart...")
    plot_accuracy_bar(experiments, os.path.join(od, "acc_96_all_experiments.png"))

    # ----------------------------------------------------------------
    # Save summary CSV
    # ----------------------------------------------------------------
    rows = []
    for exp in experiments:
        row = {"experiment": exp["label"], "n": exp["n"],
               "accuracy": round(exp["accuracy"] * 100, 2) if exp["accuracy"] is not None else None}
        for p in VALID_PREDICATES:
            v = exp["per_predicate"].get(p)
            row[p] = round(v * 100, 2) if v is not None else None
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(od, "summary_96_experiments.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
