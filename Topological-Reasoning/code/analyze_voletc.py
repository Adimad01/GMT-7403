"""
evaluate_voletc.py — Volet C: Comparative Analysis of Reasoning Strategies
===========================================================================
Produces side-by-side comparison of CoT, ToT, GoT strategies across:
  - Overall accuracy
  - Per-relation accuracy (contains, within, touches, crosses, disjoint, overlaps, equals)
  - Confusion matrices for each strategy
  - Strategy agreement analysis
  - Error pattern analysis

Usage:
  python evaluate_voletc.py --results-dir ./results
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score,
)
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
VALID_PREDICATES = sorted([
    "contains", "within", "touches", "crosses",
    "disjoint", "overlaps", "equals",
])

STRATEGY_ORDER = ["cot", "tot", "got"]
MODEL_TAGS     = ["base", "finetuned"]


RESULT_SUFFIX = "_neighborhood_details"
# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_voletc_logs(results_dir: str) -> pd.DataFrame:
    """
    Load all voletc_*_log.txt files into a single DataFrame with columns:
      index, expected, predicted, match, strategy, description, model_tag, filename
    """
    pattern = os.path.join(results_dir, "voletc_*_log.txt")
    files = glob.glob(pattern)

    if not files:
        print(f"No voletc log files found in {results_dir}")
        return pd.DataFrame()

    frames = []
    for fpath in files:
        fname = os.path.basename(fpath)
        # Parse model_tag and strategy from filename: voletc_{model_tag}_{strategy}_log.txt
        parts = fname.replace("voletc_", "").replace("_log.txt", "").split("_")
        if len(parts) >= 2:
            model_tag = parts[0]
            strategy = parts[1]
        else:
            model_tag = "unknown"
            strategy = "unknown"

        try:
            df = pd.read_csv(fpath, on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            if not {"expected", "predicted", "match"}.issubset(df.columns):
                print(f"  Skipping {fname}: missing required columns.")
                continue
            df["model_tag"] = model_tag
            df["strategy"] = strategy
            df["filename"] = fname
            frames.append(df)
            print(f"  Loaded {fname}: {len(df)} rows ({model_tag}/{strategy})")
        except Exception as e:
            print(f"  Error loading {fname}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["expected"]  = combined["expected"].astype(str).str.lower().str.strip()
    combined["predicted"] = combined["predicted"].astype(str).str.lower().str.strip()
    combined["match"]     = combined["match"].astype(str).str.lower() == "true"

    return combined


# ---------------------------------------------------------------------------
# ALSO LOAD VOLET A & B RESULTS FOR CROSS-VOLET COMPARISON
# ---------------------------------------------------------------------------
def load_voletab_logs(results_dir: str) -> pd.DataFrame:
    """Load Volet A & B log files for comparison."""
    patterns = [
        "gptoss_topological_log_*.txt",
        "gptoss_finetuned_topological_log_*.txt",
    ]
    frames = []
    for pat in patterns:
        for fpath in glob.glob(os.path.join(results_dir, pat)):
            fname = os.path.basename(fpath)
            try:
                df = pd.read_csv(fpath, on_bad_lines="skip")
                df.columns = df.columns.str.strip()
                if not {"expected", "predicted", "match"}.issubset(df.columns):
                    continue

                # Determine model tag and context type
                if "finetuned" in fname:
                    model_tag = "finetuned"
                else:
                    model_tag = "base"

                if "few_shot" in fname:
                    strategy = "few_shot"
                elif "zero_shot" in fname:
                    strategy = "zero_shot"
                else:
                    strategy = "unknown"

                df["model_tag"] = model_tag
                df["strategy"] = strategy
                df["filename"] = fname
                df["expected"]  = df["expected"].astype(str).str.lower().str.strip()
                df["predicted"] = df["predicted"].astype(str).str.lower().str.strip()
                df["match"]     = df["match"].astype(str).str.lower() == "true"
                frames.append(df)
                print(f"  Loaded (A/B) {fname}: {len(df)} rows")
            except Exception:
                pass

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# ANALYSIS FUNCTIONS
# ---------------------------------------------------------------------------
def overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall accuracy, F1, and invalid rate per strategy × model."""
    rows = []
    for (model, strat), grp in df.groupby(["model_tag", "strategy"]):
        total = len(grp)
        correct = grp["match"].sum()
        invalid = (grp["predicted"] == "invalid").sum()
        acc = (correct / total * 100) if total > 0 else 0
        inv_rate = (invalid / total * 100) if total > 0 else 0

        # F1 (macro, excluding 'invalid')
        valid_mask = grp["predicted"].isin(VALID_PREDICATES)
        if valid_mask.sum() > 0:
            f1 = f1_score(
                grp.loc[valid_mask, "expected"],
                grp.loc[valid_mask, "predicted"],
                average="macro", zero_division=0,
            ) * 100
        else:
            f1 = 0.0

        rows.append({
            "Model": model,
            "Strategy": strat,
            "Total": total,
            "Correct": correct,
            "Accuracy (%)": round(acc, 2),
            "Macro-F1 (%)": round(f1, 2),
            "Invalid (%)": round(inv_rate, 2),
        })

    return pd.DataFrame(rows).sort_values(["Model", "Strategy"])


def per_relation_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy broken down by expected relation × strategy."""
    rows = []
    for (model, strat, rel), grp in df.groupby(["model_tag", "strategy", "expected"]):
        total = len(grp)
        correct = grp["match"].sum()
        acc = (correct / total * 100) if total > 0 else 0
        rows.append({
            "Model": model,
            "Strategy": strat,
            "Relation": rel,
            "Total": total,
            "Correct": correct,
            "Accuracy (%)": round(acc, 2),
        })
    return pd.DataFrame(rows)


def generate_confusion_matrices(df: pd.DataFrame, output_dir: str):
    """Generate and save confusion matrix plots per strategy × model."""
    for (model, strat), grp in df.groupby(["model_tag", "strategy"]):
        y_true = grp["expected"]
        y_pred = grp["predicted"]
        labels = sorted(list(set(y_true) | set(y_pred)))

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"Confusion Matrix: {model} / {strat}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        plt.tight_layout()

        fname = f"voletc_cm_{model}_{strat}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")


def strategy_agreement_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, check which strategies agree on the same answer.
    Useful for understanding complementarity.
    """
    # Pivot: index=row index, columns=strategy, values=predicted
    pivot = df.pivot_table(
        index=["index", "model_tag"],
        columns="strategy",
        values="predicted",
        aggfunc="first",
    )

    strats = [c for c in pivot.columns if c in STRATEGY_ORDER]
    if len(strats) < 2:
        return pd.DataFrame()

    records = []
    for idx, row in pivot.iterrows():
        preds = {s: row.get(s, None) for s in strats if pd.notna(row.get(s, None))}
        unique = set(preds.values())
        records.append({
            "index": idx[0],
            "model_tag": idx[1],
            **preds,
            "all_agree": len(unique) == 1,
            "num_unique": len(unique),
        })

    agree_df = pd.DataFrame(records)
    return agree_df


def plot_comparison_bar(summary_df: pd.DataFrame, output_dir: str):
    """Bar chart comparing accuracy across strategies and models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for grouped bar chart
    plot_data = summary_df.copy()
    plot_data["Label"] = plot_data["Model"] + " / " + plot_data["Strategy"]
    plot_data = plot_data.sort_values("Accuracy (%)", ascending=True)

    colors = []
    for _, row in plot_data.iterrows():
        if row["Strategy"] == "cot":
            colors.append("#2196F3")
        elif row["Strategy"] == "tot":
            colors.append("#4CAF50")
        elif row["Strategy"] == "got":
            colors.append("#FF9800")
        elif "zero_shot" in row["Strategy"]:
            colors.append("#9E9E9E")
        elif "few_shot" in row["Strategy"]:
            colors.append("#607D8B")
        else:
            colors.append("#795548")

    bars = ax.barh(plot_data["Label"], plot_data["Accuracy (%)"], color=colors)

    # Add value labels
    for bar, val in zip(bars, plot_data["Accuracy (%)"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Topological Reasoning — Strategy Comparison (Volet C)")
    ax.set_xlim(0, 100)
    plt.tight_layout()

    fname = "voletc_comparison_bar.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_per_relation_heatmap(per_rel_df: pd.DataFrame, output_dir: str):
    """Heatmap of accuracy per relation per strategy."""
    for model in per_rel_df["Model"].unique():
        sub = per_rel_df[per_rel_df["Model"] == model]
        pivot = sub.pivot_table(
            index="Relation", columns="Strategy",
            values="Accuracy (%)", aggfunc="first",
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGn",
                    vmin=0, vmax=100, ax=ax)
        ax.set_title(f"Per-Relation Accuracy (%) — {model} model")
        plt.tight_layout()

        fname = f"voletc_per_relation_{model}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    """Parse arguments, load Volet C logs, and run the full comparative analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Volet C: Comparative analysis of reasoning strategies."
    )
    parser.add_argument("--results-dir", default="./results",
                        help="Directory containing voletc_*_log.txt files")
    parser.add_argument("--include-ab", action="store_true",
                        help="Also include Volet A/B results for cross-volet comparison")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"❌ Directory not found: {args.results_dir}")
        return

    print(f"\n{'='*70}")
    print(f"  VOLET C — COMPARATIVE ANALYSIS")
    print(f"{'='*70}\n")

    # Load Volet C results
    print("Loading Volet C logs...")
    df_c = load_voletc_logs(args.results_dir)

    if df_c.empty:
        print("\n  No Volet C results found. Run gptoss_voletc_eval.py first.")
        return

    # Optionally load Volet A/B results
    if args.include_ab:
        print("\nLoading Volet A/B logs...")
        df_ab = load_voletab_logs(args.results_dir)
        if not df_ab.empty:
            df_all = pd.concat([df_c, df_ab], ignore_index=True)
        else:
            df_all = df_c
    else:
        df_all = df_c

    # ---- 1. Overall Summary ----
    print(f"\n{'─'*70}")
    print("  1. OVERALL SUMMARY")
    print(f"{'─'*70}")
    summary = overall_summary(df_all)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(args.results_dir, "voletc_summary.csv"), index=False)

    # ---- 2. Per-Relation Accuracy ----
    print(f"\n{'─'*70}")
    print("  2. PER-RELATION ACCURACY")
    print(f"{'─'*70}")
    per_rel = per_relation_accuracy(df_all)
    # Show pivot view
    for model in per_rel["Model"].unique():
        sub = per_rel[per_rel["Model"] == model]
        pivot = sub.pivot_table(index="Relation", columns="Strategy",
                                values="Accuracy (%)", aggfunc="first")
        print(f"\n  [{model}]")
        print(pivot.to_string())
    per_rel.to_csv(os.path.join(args.results_dir, "voletc_per_relation.csv"), index=False)

    # ---- 3. Confusion Matrices ----
    print(f"\n{'─'*70}")
    print("  3. CONFUSION MATRICES")
    print(f"{'─'*70}")
    generate_confusion_matrices(df_all, args.results_dir)

    # ---- 4. Strategy Agreement ----
    print(f"\n{'─'*70}")
    print("  4. STRATEGY AGREEMENT ANALYSIS")
    print(f"{'─'*70}")
    agree_df = strategy_agreement_analysis(df_c)
    if not agree_df.empty and "all_agree" in agree_df.columns:
        for model in agree_df["model_tag"].unique():
            sub = agree_df[agree_df["model_tag"] == model]
            agree_pct = sub["all_agree"].mean() * 100
            print(f"  [{model}] All strategies agree: {agree_pct:.1f}% of rows")
        agree_df.to_csv(os.path.join(args.results_dir, "voletc_agreement.csv"), index=False)
    else:
        print("  Not enough strategies to compare agreement.")

    # ---- 5. Plots ----
    print(f"\n{'─'*70}")
    print("  5. GENERATING PLOTS")
    print(f"{'─'*70}")
    plot_comparison_bar(summary, args.results_dir)
    plot_per_relation_heatmap(per_rel, args.results_dir)

    # ---- 6. Classification Reports ----
    print(f"\n{'─'*70}")
    print("  6. CLASSIFICATION REPORTS")
    print(f"{'─'*70}")
    report_file = os.path.join(args.results_dir, "voletc_classification_reports.txt")
    with open(report_file, "w") as rf:
        for (model, strat), grp in df_all.groupby(["model_tag", "strategy"]):
            valid = grp[grp["predicted"].isin(VALID_PREDICATES)]
            if len(valid) == 0:
                continue
            header = f"\n{'='*50}\n{model} / {strat}\n{'='*50}\n"
            report = classification_report(
                valid["expected"], valid["predicted"],
                zero_division=0,
            )
            print(header + report)
            rf.write(header + report + "\n")
    print(f"\n  Full reports saved to: {report_file}")

    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE")
    print(f"  All outputs saved in: {args.results_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
