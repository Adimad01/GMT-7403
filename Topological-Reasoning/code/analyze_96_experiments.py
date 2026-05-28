"""
analyze_96_experiments.py
================================================================================
Unified comparison of all 6 experiments × 3 strategies (CoT, ToT, GoT) on the
96 balanced test examples.

Each experiment uses a different model/adapter configuration; all 6 are evaluated
with CoT, ToT, and GoT — producing 18 result sets total.

Experiment configurations:
  Exp 1 — GPTOSS Base           tag: exp1_base_gpu
  Exp 2 — GPTOSS Fine-tuné      tag: exp2_finetuned_topo_gpu
  Exp 3 — Fine-tuné + KG entrée tag: exp3_finetuned_kg_in_gpu
  Exp 4 — Fine-tuné + KG ft     tag: exp4_finetuned_osm_kg_gpu
  Exp 5 — Fine-tuné + Enriched  tag: exp5_finetuned_enriched_gpu
  Exp 6 — Base + Enriched Ollama tag: dynamic_osm_improved_version  ✅ done

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

VALID_PREDICATES = ["contains", "within", "touches", "crosses", "disjoint", "overlaps"]
STRATEGIES       = ["cot", "tot", "got"]
SUFFIX           = "neighborhood_details_spatial_relation_16_sample"

# ---------------------------------------------------------------------------
# Experiment registry — (label, model_tag, adapter_note)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    ("Exp1 — Base",                 "exp1_base_gpu",              "no adapter"),
    ("Exp2 — Fine-tuné",            "exp2_finetuned_topo_gpu",    "topo adapter"),
    ("Exp3 — FT + KG entrée",       "exp3_finetuned_kg_in_gpu",   "topo adapter + KG at inf."),
    ("Exp4 — FT + KG ft",           "exp4_finetuned_osm_kg_gpu",  "osm-kg adapter"),
    ("Exp5 — FT + Enriched GPU",    "exp5_finetuned_enriched_gpu","topo adapter"),
    ("Exp6 — Base + Ollama",        "dynamic_osm_improved_version","no adapter (Ollama)"),
]


# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------

def load_ckpt(ckpt_path: str) -> pd.DataFrame | None:
    if not os.path.exists(ckpt_path):
        return None
    with open(ckpt_path) as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        return None
    return pd.DataFrame({
        "index":     [r["index"]     for r in results],
        "expected":  [r["expected"]  for r in results],
        "predicted": [r["predicted"] for r in results],
        "match":     [bool(r["match"]) for r in results],
    })


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, label: str) -> dict:
    if df is None or len(df) == 0:
        return {"label": label, "n": 0, "accuracy": None, "per_predicate": {p: None for p in VALID_PREDICATES}}

    overall_acc = df["match"].mean()
    per_pred = {}
    for pred in VALID_PREDICATES:
        sub = df[df["expected"] == pred]
        per_pred[pred] = sub["match"].mean() if len(sub) > 0 else None

    return {"label": label, "n": len(df), "accuracy": overall_acc, "per_predicate": per_pred}


# ---------------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------------

def print_table(results_matrix: dict):
    """results_matrix[exp_label][strategy] = metrics dict"""
    col_w = 14
    pred_w = 8

    for strat in STRATEGIES:
        print("\n" + "=" * 110)
        print(f"  STRATEGY: {strat.upper()}  —  96 BALANCED TEST EXAMPLES (OSM KG)")
        print("=" * 110)
        header = f"{'Experiment':<42} {'N':>4} {'Accuracy':>10}  " + \
                 "  ".join(f"{p[:7]:>{pred_w}}" for p in VALID_PREDICATES)
        print(header)
        print("-" * 110)

        for exp_label, _, _ in EXPERIMENTS:
            m = results_matrix.get(exp_label, {}).get(strat)
            if m is None:
                print(f"{exp_label:<42} {'—':>4} {'N/A':>10}")
                continue
            acc_str  = f"{m['accuracy']*100:>9.1f}%" if m["accuracy"] is not None else "     N/A"
            per_cols = "  ".join(
                f"{v*100:>{pred_w}.1f}%" if v is not None else f"{'N/A':>{pred_w}}"
                for v in (m["per_predicate"].get(p) for p in VALID_PREDICATES)
            )
            print(f"{exp_label:<42} {m['n']:>4} {acc_str}  {per_cols}")

        print("=" * 110)

    # Summary: best strategy per experiment
    print("\n" + "=" * 70)
    print("  BEST STRATEGY PER EXPERIMENT (overall accuracy)")
    print("=" * 70)
    for exp_label, _, adapter_note in EXPERIMENTS:
        best_strat, best_acc = None, -1
        for strat in STRATEGIES:
            m = results_matrix.get(exp_label, {}).get(strat)
            if m and m["accuracy"] is not None and m["accuracy"] > best_acc:
                best_acc   = m["accuracy"]
                best_strat = strat
        if best_strat:
            print(f"  {exp_label:<42} → {best_strat.upper():3s}  {best_acc*100:.1f}%  [{adapter_note}]")
        else:
            print(f"  {exp_label:<42} → N/A (not yet run)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------------------------

def plot_confusion_matrix(df: pd.DataFrame, title: str, save_path: str):
    if df is None or len(df) == 0:
        return
    matrix   = np.zeros((len(VALID_PREDICATES), len(VALID_PREDICATES)), dtype=int)
    pred_idx = {p: i for i, p in enumerate(VALID_PREDICATES)}

    for _, row in df.iterrows():
        e = str(row.get("expected",  "")).lower().strip()
        p = str(row.get("predicted", "")).lower().strip()
        if e in pred_idx and p in pred_idx:
            matrix[pred_idx[e]][pred_idx[p]] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(VALID_PREDICATES)))
    ax.set_yticks(range(len(VALID_PREDICATES)))
    ax.set_xticklabels(VALID_PREDICATES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(VALID_PREDICATES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9, pad=8)
    thresh = matrix.max() / 2
    for i in range(len(VALID_PREDICATES)):
        for j in range(len(VALID_PREDICATES)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    color="white" if matrix[i][j] > thresh else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# GROUPED BAR CHART
# ---------------------------------------------------------------------------

def plot_grouped_bar(results_matrix: dict, save_path: str):
    exp_labels  = [e[0] for e in EXPERIMENTS]
    x           = np.arange(len(exp_labels))
    width       = 0.25
    offsets     = [-width, 0, width]
    colors      = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, strat in enumerate(STRATEGIES):
        accs = []
        for exp_label in exp_labels:
            m = results_matrix.get(exp_label, {}).get(strat)
            accs.append(m["accuracy"] * 100 if (m and m["accuracy"] is not None) else 0)
        bars = ax.bar(x + offsets[i], accs, width, label=strat.upper(),
                      color=colors[i], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, accs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Accuracy per Experiment × Strategy  (96 balanced examples, OSM KG)")
    ax.legend(title="Strategy", loc="upper left")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir",  default="results")
    args = parser.parse_args()
    rd, od = args.results_dir, args.output_dir
    os.makedirs(od, exist_ok=True)

    # ----------------------------------------------------------------
    # Load all 18 result sets
    # ----------------------------------------------------------------
    results_matrix = {}   # [exp_label][strategy] = metrics dict
    all_dfs        = {}   # [exp_label][strategy] = DataFrame (for confusion matrices)

    for exp_label, model_tag, _ in EXPERIMENTS:
        results_matrix[exp_label] = {}
        all_dfs[exp_label]        = {}
        for strat in STRATEGIES:
            ckpt = os.path.join(rd, f"voletc_{model_tag}_{strat}_{SUFFIX}_ckpt.json")
            df   = load_ckpt(ckpt)
            status = f"{len(df)}/96" if df is not None else "MISSING"
            print(f"[{'OK' if df is not None and len(df)==96 else 'PARTIAL' if df is not None else 'MISSING':7s}] {model_tag}_{strat}  ({status})")
            results_matrix[exp_label][strat] = compute_metrics(df, f"{exp_label} / {strat.upper()}")
            all_dfs[exp_label][strat]        = df

    # ----------------------------------------------------------------
    # Print comparison tables
    # ----------------------------------------------------------------
    print_table(results_matrix)

    # ----------------------------------------------------------------
    # Confusion matrices
    # ----------------------------------------------------------------
    print("\nGenerating confusion matrices...")
    for exp_label, model_tag, _ in EXPERIMENTS:
        for strat in STRATEGIES:
            df = all_dfs[exp_label][strat]
            if df is not None and len(df) > 0:
                safe_tag = model_tag.replace("/", "_")
                fname    = os.path.join(od, f"cm_{safe_tag}_{strat}.png")
                plot_confusion_matrix(df, f"{exp_label} / {strat.upper()}", fname)

    # ----------------------------------------------------------------
    # Grouped bar chart (all experiments × strategies)
    # ----------------------------------------------------------------
    print("\nGenerating grouped accuracy chart...")
    plot_grouped_bar(results_matrix, os.path.join(od, "acc_96_experiments_by_strategy.png"))

    # ----------------------------------------------------------------
    # Summary CSV
    # ----------------------------------------------------------------
    rows = []
    for exp_label, model_tag, adapter_note in EXPERIMENTS:
        for strat in STRATEGIES:
            m = results_matrix[exp_label][strat]
            row = {
                "experiment": exp_label,
                "model_tag":  model_tag,
                "adapter":    adapter_note,
                "strategy":   strat.upper(),
                "n":          m["n"],
                "accuracy":   round(m["accuracy"] * 100, 2) if m["accuracy"] is not None else None,
            }
            for p in VALID_PREDICATES:
                v = m["per_predicate"].get(p)
                row[p] = round(v * 100, 2) if v is not None else None
            rows.append(row)

    summary_df  = pd.DataFrame(rows)
    summary_path = os.path.join(od, "summary_96_6exp_3strat.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary CSV → {summary_path}")


if __name__ == "__main__":
    main()
