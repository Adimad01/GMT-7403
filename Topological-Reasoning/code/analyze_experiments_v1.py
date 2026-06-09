"""
analyze_experiments_v1.py
================================================================================
Unified comparison of 5 experiments × 3 strategies (CoT, ToT, GoT) on the
96 vol-1 balanced test examples (16/predicate × 6 predicates). All experiments
on A100 80GB.

Experiment configurations:
  vol1-01 — Base (no adapter)                          tag: exp1_base_gpu          512 tok
  vol1-02 — Topo-LoRA                                  tag: exp2_finetuned_topo_gpu 512 tok
  vol1-03 — OSM-KG LoRA  (KG in training + inference)  tag: exp3_finetuned_kg_in_gpu 1024 tok
  vol1-04 — OSM-KG LoRA  (KG en training seulement)     tag: exp4_osm_kg_ft_only_gpu   512 tok
  vol1-05 — Topo-LoRA + extended budget                tag: exp5_finetuned_enriched_gpu 1024 tok

Usage:
    cd /path/to/Topological-Reasoning/code
    python analyze_experiments_v1.py [--results-dir results] [--output-dir results]
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

VALID_PREDICATES = ["contains", "within", "touches", "crosses", "disjoint", "overlaps", "equals"]
STRATEGIES       = ["cot", "tot", "got"]
SUFFIX           = "neighborhood_details_spatial_relation_16_sample"
N_EVAL           = 112

# ---------------------------------------------------------------------------
# Experiment registry — (label, model_tag, adapter_note)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    ("Base GPT-OSS-20B",                          "exp1_base_gpu",               "no adapter, 512 tok"),
    ("Topo-LoRA",                                 "exp2_finetuned_topo_gpu",     "topo adapter, 512 tok"),
    ("OSM-KG LoRA  (KG at inference)",            "exp3_finetuned_kg_in_gpu",    "osm-kg adapter, 1024 tok"),
    ("OSM-KG LoRA  (KG en training seulement)",   "exp4_osm_kg_ft_only_gpu",     "osm-kg adapter, 512 tok, no KG at inference"),
    ("Topo-LoRA + extended budget",               "exp5_finetuned_enriched_gpu", "topo adapter, 1024 tok"),
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
        "index":     [r["index"]         for r in results],
        "expected":  [r["expected"]      for r in results],
        "predicted": [r.get("predicted") for r in results],
        "match":     [bool(r["match"])   for r in results],
    })


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, label: str) -> dict:
    if df is None or len(df) == 0:
        return {"label": label, "n": 0, "accuracy": None,
                "per_predicate": {p: None for p in VALID_PREDICATES}}

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
    pred_w = 8

    for strat in STRATEGIES:
        header_preds = "  ".join(f"{p[:7]:>{pred_w}}" for p in VALID_PREDICATES)
        total_w = 46 + 4 + 10 + 2 + len(VALID_PREDICATES) * (pred_w + 2)

        print("\n" + "=" * total_w)
        print(f"  STRATEGY: {strat.upper()}  —  {N_EVAL} vol-1 TEST EXAMPLES  (6 predicates, 16 each)")
        print("=" * total_w)
        header = (f"{'Experiment':<44} {'N':>4} {'Accuracy':>10}  " + header_preds)
        print(header)
        print("-" * total_w)

        for exp_label, _, _ in EXPERIMENTS:
            m = results_matrix.get(exp_label, {}).get(strat)
            if m is None or m["n"] == 0:
                print(f"{exp_label:<44} {'—':>4} {'N/A':>10}")
                continue
            acc_str  = f"{m['accuracy']*100:>9.1f}%" if m["accuracy"] is not None else "     N/A"
            per_cols = "  ".join(
                f"{v*100:>{pred_w}.1f}%" if v is not None else f"{'N/A':>{pred_w}}"
                for v in (m["per_predicate"].get(p) for p in VALID_PREDICATES)
            )
            print(f"{exp_label:<44} {m['n']:>4} {acc_str}  {per_cols}")

        print("=" * total_w)

    # Summary: best strategy per experiment
    print("\n" + "=" * 72)
    print("  BEST STRATEGY PER EXPERIMENT (overall accuracy)")
    print("=" * 72)
    for exp_label, _, adapter_note in EXPERIMENTS:
        best_strat, best_acc = None, -1
        for strat in STRATEGIES:
            m = results_matrix.get(exp_label, {}).get(strat)
            if m and m["accuracy"] is not None and m["accuracy"] > best_acc:
                best_acc   = m["accuracy"]
                best_strat = strat
        if best_strat:
            print(f"  {exp_label:<44} → {best_strat.upper():3s}  {best_acc*100:.1f}%  [{adapter_note}]")
        else:
            print(f"  {exp_label:<44} → N/A (not yet run)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CONFUSION MATRIX  (6 × 6)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(df: pd.DataFrame, title: str, save_path: str):
    if df is None or len(df) == 0:
        return
    matrix   = np.zeros((len(VALID_PREDICATES), len(VALID_PREDICATES)), dtype=int)
    pred_idx = {p: i for i, p in enumerate(VALID_PREDICATES)}

    unknown_expected = {}
    for _, row in df.iterrows():
        e = str(row.get("expected",  "")).lower().strip()
        p = str(row.get("predicted", "")).lower().strip()
        if e in pred_idx:
            if p in pred_idx:
                matrix[pred_idx[e]][pred_idx[p]] += 1
            else:
                unknown_expected[e] = unknown_expected.get(e, 0) + 1

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

    if unknown_expected:
        note = "Unparsed: " + ", ".join(f"{k}:{v}" for k, v in unknown_expected.items())
        ax.set_xlabel(f"Predicted\n({note})", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# GROUPED BAR CHART
# ---------------------------------------------------------------------------

def plot_grouped_bar(results_matrix: dict, save_path: str):
    exp_labels = [e[0] for e in EXPERIMENTS]
    x          = np.arange(len(exp_labels))
    width      = 0.25
    offsets    = [-width, 0, width]
    colors     = ["#4C72B0", "#DD8452", "#55A868"]

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
    ax.set_title(
        f"Accuracy per Experiment × Strategy  ({N_EVAL} vol-1 examples · 6 predicates · A100 80GB)",
        fontsize=10,
    )
    ax.legend(title="Strategy", loc="upper right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# PER-PREDICATE BREAKDOWN CHART
# ---------------------------------------------------------------------------

def plot_per_predicate(results_matrix: dict, strat: str, save_path: str):
    exp_labels = [e[0] for e in EXPERIMENTS]
    x          = np.arange(len(VALID_PREDICATES))
    width      = 0.16
    n_exp      = len(exp_labels)
    offsets    = [width * (i - (n_exp - 1) / 2) for i in range(n_exp)]
    colors     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, exp_label in enumerate(exp_labels):
        m    = results_matrix.get(exp_label, {}).get(strat)
        vals = [
            (m["per_predicate"].get(p) or 0) * 100
            if (m and m["accuracy"] is not None) else 0
            for p in VALID_PREDICATES
        ]
        bars = ax.bar(x + offsets[i], vals, width, label=exp_label,
                      color=colors[i], edgecolor="black", linewidth=0.4)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(VALID_PREDICATES, fontsize=9)
    ax.set_ylabel("Accuracy per predicate (%)")
    ax.set_ylim(0, 120)
    ax.set_title(f"Per-predicate accuracy — {strat.upper()}  (vol-1 · 16 examples/predicate)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
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
    # Load all checkpoints
    # ----------------------------------------------------------------
    results_matrix = {}
    all_dfs        = {}

    for exp_label, model_tag, _ in EXPERIMENTS:
        results_matrix[exp_label] = {}
        all_dfs[exp_label]        = {}
        for strat in STRATEGIES:
            ckpt   = os.path.join(rd, f"voletc_{model_tag}_{strat}_{SUFFIX}_ckpt.json")
            df     = load_ckpt(ckpt)
            n      = len(df) if df is not None else 0
            status = "OK" if n == N_EVAL else ("PARTIAL" if n > 0 else "MISSING")
            print(f"[{status:7s}] {model_tag}_{strat}  ({n}/{N_EVAL})")
            results_matrix[exp_label][strat] = compute_metrics(df, f"{exp_label} / {strat.upper()}")
            all_dfs[exp_label][strat]        = df

    # ----------------------------------------------------------------
    # Comparison tables
    # ----------------------------------------------------------------
    print_table(results_matrix)

    # ----------------------------------------------------------------
    # Confusion matrices (6 × 6)
    # ----------------------------------------------------------------
    print("\nGenerating confusion matrices...")
    for exp_label, model_tag, _ in EXPERIMENTS:
        for strat in STRATEGIES:
            df = all_dfs[exp_label][strat]
            if df is not None and len(df) > 0:
                safe_tag = model_tag.replace("/", "_")
                fname    = os.path.join(od, f"cm_v1_{safe_tag}_{strat}.png")
                plot_confusion_matrix(df, f"{exp_label} / {strat.upper()}", fname)

    # ----------------------------------------------------------------
    # Grouped bar chart
    # ----------------------------------------------------------------
    print("\nGenerating grouped accuracy chart...")
    plot_grouped_bar(
        results_matrix,
        os.path.join(od, "acc_v1_5exp_3strat.png"),
    )

    # ----------------------------------------------------------------
    # Per-predicate breakdown (one chart per strategy)
    # ----------------------------------------------------------------
    print("\nGenerating per-predicate breakdown charts...")
    for strat in STRATEGIES:
        plot_per_predicate(
            results_matrix, strat,
            os.path.join(od, f"per_pred_v1_{strat}.png"),
        )

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

    summary_df   = pd.DataFrame(rows)
    summary_path = os.path.join(od, "summary_v1_5exp_3strat.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary CSV → {summary_path}")


if __name__ == "__main__":
    main()
