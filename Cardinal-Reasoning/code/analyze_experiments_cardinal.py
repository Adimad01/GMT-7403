"""
analyze_experiments_cardinal.py
================================================================================
Unified comparison of 5 cardinal direction experiments × 3 strategies
(CoT, ToT, GoT) on 80 eval examples (10/direction × 8 directions).

Experiment configurations:
  card-01 — Base (no adapter)                          tag: exp1_card_base_gpu          512 tok
  card-02 — Cardinal-LoRA                              tag: exp2_card_topo_gpu          512 tok
  card-03 — Cardinal-KG LoRA (KG in train+inference)   tag: exp3_card_kg_gpu           1024 tok
  card-04 — Wikidata-KG LoRA (cross-task transfer)     tag: exp4_card_wikidata_gpu     1024 tok
  card-05 — Cardinal-LoRA + extended budget             tag: exp5_card_enriched_gpu     1024 tok

Usage:
    cd /path/to/Cardinal-Reasoning/code
    python analyze_experiments_cardinal.py [--results-dir results] [--output-dir results]
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

DIRECTIONS = [
    "north", "south", "east", "west",
    "north-east", "north-west", "south-east", "south-west",
]
STRATEGIES = ["cot", "tot", "got"]
SUFFIX     = "cardinal_direction_80_sample"
N_EVAL     = 80

EXPERIMENTS = [
    ("Base GPT-OSS-20B",                         "exp1_card_base_gpu",       "no adapter, 512 tok"),
    ("Cardinal-LoRA",                             "exp2_card_topo_gpu",       "cardinal adapter, 512 tok"),
    ("Cardinal-KG LoRA  (KG at inference)",       "exp3_card_kg_gpu",         "cardinal-kg adapter, 1024 tok"),
    ("Wikidata-KG LoRA  (cross-task transfer)",   "exp4_card_wikidata_gpu",   "wikidata adapter, 1024 tok"),
    ("Cardinal-LoRA + extended budget",           "exp5_card_enriched_gpu",   "cardinal adapter, 1024 tok"),
]


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


def compute_metrics(df: pd.DataFrame, label: str) -> dict:
    if df is None or len(df) == 0:
        return {"label": label, "n": 0, "accuracy": None,
                "per_direction": {d: None for d in DIRECTIONS}}
    overall_acc = df["match"].mean()
    per_dir = {}
    for d in DIRECTIONS:
        sub = df[df["expected"] == d]
        per_dir[d] = sub["match"].mean() if len(sub) > 0 else None
    return {"label": label, "n": len(df), "accuracy": overall_acc,
            "per_direction": per_dir}


# ---------------------------------------------------------------------------
# TABLE
# ---------------------------------------------------------------------------
def print_summary_table(results_dir: str):
    print("\n" + "=" * 120)
    print("  CARDINAL DIRECTION — EXPERIMENT RESULTS (5 configs × 3 strategies)")
    print("=" * 120)
    header = f"{'Experiment':<45} {'Adapter':<30} {'CoT':>8} {'ToT':>8} {'GoT':>8} {'Best':>8}"
    print(header)
    print("-" * 120)

    for label, tag, note in EXPERIMENTS:
        accs = {}
        for strat in STRATEGIES:
            ckpt_path = os.path.join(results_dir, f"voletc_{tag}_{strat}_{SUFFIX}_ckpt.json")
            df = load_ckpt(ckpt_path)
            m  = compute_metrics(df, f"{label}/{strat}")
            accs[strat] = m["accuracy"]

        def fmt(v):
            return f"{v * 100:.1f}%" if v is not None else "  —  "

        best = max((v for v in accs.values() if v is not None), default=None)
        print(
            f"  {label:<43} {note:<30} "
            f"{fmt(accs['cot']):>8} {fmt(accs['tot']):>8} {fmt(accs['got']):>8} "
            f"{fmt(best):>8}"
        )
    print("=" * 120)


# ---------------------------------------------------------------------------
# PER-DIRECTION TABLE
# ---------------------------------------------------------------------------
def print_per_direction_table(results_dir: str):
    print("\n" + "=" * 120)
    print("  PER-DIRECTION BREAKDOWN (best strategy per experiment)")
    print("=" * 120)
    header = f"{'Experiment':<45} " + " ".join(f"{d:>12}" for d in DIRECTIONS)
    print(header)
    print("-" * 120)

    for label, tag, _ in EXPERIMENTS:
        best_df = None
        best_acc = -1.0
        for strat in STRATEGIES:
            ckpt_path = os.path.join(results_dir, f"voletc_{tag}_{strat}_{SUFFIX}_ckpt.json")
            df = load_ckpt(ckpt_path)
            if df is not None and df["match"].mean() > best_acc:
                best_acc = df["match"].mean()
                best_df = df
        m = compute_metrics(best_df, label)
        per_dir_str = " ".join(
            f"{m['per_direction'][d] * 100:>11.1f}%" if m['per_direction'][d] is not None else f"{'—':>12}"
            for d in DIRECTIONS
        )
        print(f"  {label:<43} {per_dir_str}")
    print("=" * 120)


# ---------------------------------------------------------------------------
# BAR CHART
# ---------------------------------------------------------------------------
def plot_bar_chart(results_dir: str, output_dir: str):
    labels = [e[0] for e in EXPERIMENTS]
    cot_accs, tot_accs, got_accs = [], [], []

    for _, tag, _ in EXPERIMENTS:
        for strat, acc_list in [("cot", cot_accs), ("tot", tot_accs), ("got", got_accs)]:
            ckpt_path = os.path.join(results_dir, f"voletc_{tag}_{strat}_{SUFFIX}_ckpt.json")
            df = load_ckpt(ckpt_path)
            m  = compute_metrics(df, tag)
            acc_list.append(m["accuracy"] * 100 if m["accuracy"] is not None else 0.0)

    x      = np.arange(len(labels))
    width  = 0.26
    fig, ax = plt.subplots(figsize=(14, 6))

    b1 = ax.bar(x - width, cot_accs, width, label="CoT", color="#0071e3", alpha=0.9)
    b2 = ax.bar(x,          tot_accs, width, label="ToT", color="#30d158", alpha=0.9)
    b3 = ax.bar(x + width,  got_accs, width, label="GoT", color="#ff9f0a", alpha=0.9)

    ax.set_xlabel("Experiment Configuration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cardinal Direction Reasoning — GPT-OSS-20B (5 configs × 3 strategies, n=80)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "cardinal_experiment_results.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n[CHART] Saved: {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir",  default="results")
    parser.add_argument("--no-chart",    action="store_true")
    args = parser.parse_args()

    print_summary_table(args.results_dir)
    print_per_direction_table(args.results_dir)

    if not args.no_chart:
        try:
            plot_bar_chart(args.results_dir, args.output_dir)
        except Exception as e:
            print(f"[WARN] Could not generate chart: {e}")


if __name__ == "__main__":
    main()
