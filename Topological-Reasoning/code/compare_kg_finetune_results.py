"""
compare_kg_finetune_results.py
================================================================================
Phase 5 — Comparison and Analysis

Loads evaluation logs for all four model variants and produces:
  1. Overall accuracy table
  2. Per-predicate F1 table
  3. Per-predicate confusion matrix (text)
  4. Per-model confusion matrix saved as PNG (optional, requires matplotlib)

Auto-discovers files by a flexible path mapping so you can either point to
specific files with --log-* flags or let it search the default results/ dir.

Expected log files (all in --results-dir):
  • kg_eval_gptoss_base_no_kg_none.csv          (base model)
  • kg_eval_gptoss_data_only_none.csv            (data-only fine-tuned)
  • kg_eval_gptoss_osm_kg_finetuned_osm.csv      (OSM-KG fine-tuned)
  • kg_eval_gptoss_wikidata_kg_finetuned_wikidata.csv  (Wikidata-KG fine-tuned)

The script also accepts legacy logs from the existing eval pipeline:
  • gptoss_topological_log_zero_shot_improved.txt    (base, data-only baseline)
  • gptoss_finetuned_topological_log_zero_shot_improved.txt

Usage:
  python compare_kg_finetune_results.py --results-dir results

Or with explicit paths for each model:
  python compare_kg_finetune_results.py \
      --base-log         results/kg_eval_gptoss_base_no_kg_none.csv \
      --data-only-log    results/kg_eval_gptoss_data_only_none.csv \
      --osm-kg-log       results/kg_eval_gptoss_osm_kg_finetuned_osm.csv \
      --wikidata-kg-log  results/kg_eval_gptoss_wikidata_kg_finetuned_wikidata.csv
"""

import argparse
import os
import glob

import pandas as pd
import numpy as np

VALID_PREDICATES = ["contains", "within", "touches", "crosses", "disjoint", "overlaps"]


# ---------------------------------------------------------------------------
# Loaders — handle both CSV (new format) and legacy TXT (comma-separated)
# ---------------------------------------------------------------------------
def load_log(path: str) -> pd.DataFrame:
    """Load an evaluation log into a DataFrame with [expected, predicted] columns."""
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Normalise column names (legacy logs use same header)
    df.columns = [c.strip().lower() for c in df.columns]

    if "expected" not in df.columns or "predicted" not in df.columns:
        print(f"[WARN] Skipping {path} — missing 'expected' or 'predicted' columns.")
        return pd.DataFrame()

    df["expected"]  = df["expected"].astype(str).str.lower().str.strip()
    df["predicted"] = df["predicted"].astype(str).str.lower().str.strip()
    df["match"]     = df["expected"] == df["predicted"]
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def accuracy(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return df["match"].mean()


def per_predicate_f1(df: pd.DataFrame) -> dict[str, dict]:
    """Compute precision, recall, F1 for each valid predicate."""
    stats = {}
    for pred in VALID_PREDICATES:
        tp = ((df["expected"] == pred) & (df["predicted"] == pred)).sum()
        fp = ((df["expected"] != pred) & (df["predicted"] == pred)).sum()
        fn = ((df["expected"] == pred) & (df["predicted"] != pred)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = int((df["expected"] == pred).sum())

        stats[pred] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   support,
        }
    return stats


def confusion_matrix_text(df: pd.DataFrame, label: str) -> str:
    """Return a formatted confusion matrix string for the given model label."""
    preds_valid = [p for p in VALID_PREDICATES if p in df["predicted"].values or p in df["expected"].values]

    header = f"Confusion matrix — {label}\n"
    header += " " * 12 + "  ".join(f"{p[:8]:>8}" for p in VALID_PREDICATES) + "  (predicted)\n"
    rows = [header]

    for true_label in VALID_PREDICATES:
        row_data = []
        for pred_label in VALID_PREDICATES:
            count = ((df["expected"] == true_label) & (df["predicted"] == pred_label)).sum()
            row_data.append(f"{count:>8}")
        support = int((df["expected"] == true_label).sum())
        rows.append(f"{true_label[:11]:>11}  {'  '.join(row_data)}    n={support}")

    return "\n".join(rows)


def macro_f1(f1_stats: dict) -> float:
    scores = [v["f1"] for v in f1_stats.values()]
    return round(float(np.mean(scores)), 4) if scores else 0.0


# ---------------------------------------------------------------------------
# Optional confusion matrix PNG
# ---------------------------------------------------------------------------
def save_confusion_png(df: pd.DataFrame, label: str, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    matrix = np.zeros((len(VALID_PREDICATES), len(VALID_PREDICATES)), dtype=int)
    for i, true_lbl in enumerate(VALID_PREDICATES):
        for j, pred_lbl in enumerate(VALID_PREDICATES):
            matrix[i, j] = ((df["expected"] == true_lbl) & (df["predicted"] == pred_lbl)).sum()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=VALID_PREDICATES, yticklabels=VALID_PREDICATES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix PNG saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare KG fine-tuning results across all model variants")
    parser.add_argument("--results-dir",     default="results",
                        help="Directory containing all evaluation log files")
    parser.add_argument("--base-log",        default=None)
    parser.add_argument("--data-only-log",   default=None)
    parser.add_argument("--osm-kg-log",      default=None)
    parser.add_argument("--wikidata-kg-log", default=None)
    parser.add_argument("--save-png",        action="store_true",
                        help="Save per-model confusion matrices as PNG")
    args = parser.parse_args()

    results_dir = args.results_dir

    # Resolve log paths: explicit flags override auto-discovery
    def _auto(pattern: str) -> str | None:
        matches = glob.glob(os.path.join(results_dir, pattern))
        return matches[0] if matches else None

    log_map = {
        "Base GPT-OSS\n(no KG, no adapter)": (
            args.base_log
            or _auto("kg_eval_gptoss_base*none.csv")
            or _auto("gptoss_topological_log_zero_shot_improved.txt")
        ),
        "Data-Only Fine-tuned\n(no KG)": (
            args.data_only_log
            or _auto("kg_eval_gptoss_data_only*none.csv")
            or _auto("gptoss_finetuned_topological_log_zero_shot_improved.txt")
        ),
        "OSM-KG Fine-tuned\n(geometric KG)": (
            args.osm_kg_log
            or _auto("kg_eval_*osm_kg*osm.csv")
        ),
        "Wikidata-KG Fine-tuned\n(semantic KG)": (
            args.wikidata_kg_log
            or _auto("kg_eval_*wikidata_kg*wikidata.csv")
        ),
    }

    print("\n" + "=" * 80)
    print("KGs Instruction-Tuning — Topological Reasoning Comparison")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Per-model load + overall accuracy table
    # ------------------------------------------------------------------
    dfs: dict[str, pd.DataFrame] = {}
    print("\n[FILE DISCOVERY]")
    for label, path in log_map.items():
        short_label = label.replace("\n", " ")
        if path and os.path.exists(path):
            df = load_log(path)
            if not df.empty:
                dfs[label] = df
                print(f"  ✓ {short_label}: {path}  ({len(df)} rows)")
            else:
                print(f"  ✗ {short_label}: {path}  (empty or unreadable)")
        else:
            print(f"  — {short_label}: NOT FOUND (run the eval script first)")

    if not dfs:
        print("\n[ERROR] No valid log files found. Run eval_kg_instruction_finetuned.py first.")
        return

    # ------------------------------------------------------------------
    # Overall accuracy table
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("OVERALL ACCURACY")
    print("-" * 80)
    print(f"{'Model':<45} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print("-" * 80)
    for label, df in dfs.items():
        acc = accuracy(df)
        correct = df["match"].sum()
        total = len(df)
        print(f"{label.replace(chr(10), ' '):<45} {acc:>10.4f} {correct:>10} {total:>8}")
    print("-" * 80)

    # ------------------------------------------------------------------
    # Per-predicate F1 table
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PER-PREDICATE F1 SCORES")
    print("-" * 80)

    col_width = 13
    header = f"{'Predicate':<12}" + "".join(
        f"{lbl.replace(chr(10), ' ')[:col_width]:>{col_width}}" for lbl in dfs.keys()
    )
    print(header)
    print("-" * 80)

    f1_tables: dict[str, dict] = {lbl: per_predicate_f1(df) for lbl, df in dfs.items()}

    for pred in VALID_PREDICATES:
        row_str = f"{pred:<12}"
        for lbl in dfs.keys():
            f1_val = f1_tables[lbl].get(pred, {}).get("f1", float("nan"))
            row_str += f"{f1_val:>{col_width}.4f}"
        print(row_str)

    # Macro F1 row
    print("-" * 80)
    macro_row = f"{'macro-F1':<12}"
    for lbl, f1_stat in f1_tables.items():
        macro_row += f"{macro_f1(f1_stat):>{col_width}.4f}"
    print(macro_row)
    print("-" * 80)

    # ------------------------------------------------------------------
    # Per-predicate support
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PER-PREDICATE SUPPORT (test set size per class)")
    print("-" * 80)
    first_df = next(iter(dfs.values()))
    for pred in VALID_PREDICATES:
        support = int((first_df["expected"] == pred).sum())
        print(f"  {pred:<12}: {support}")

    # ------------------------------------------------------------------
    # Confusion matrices (text)
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("CONFUSION MATRICES (rows = true label, cols = predicted label)")
    print("-" * 80)
    for label, df in dfs.items():
        print()
        print(confusion_matrix_text(df, label.replace("\n", " ")))

    # ------------------------------------------------------------------
    # Key analysis questions
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESEARCH ANALYSIS")
    print("=" * 80)

    for label, df in dfs.items():
        short = label.replace("\n", " ")
        within_contains_err = (
            ((df["expected"] == "within") & (df["predicted"] == "contains")).sum()
            + ((df["expected"] == "contains") & (df["predicted"] == "within")).sum()
        )
        touches_overlaps_err = (
            ((df["expected"] == "touches") & (df["predicted"] == "overlaps")).sum()
            + ((df["expected"] == "overlaps") & (df["predicted"] == "touches")).sum()
        )
        invalid_count = (df["predicted"] == "invalid").sum()
        print(f"\n  {short}")
        print(f"    within ↔ contains confusions  : {within_contains_err}")
        print(f"    touches ↔ overlaps confusions  : {touches_overlaps_err}")
        print(f"    invalid predictions            : {invalid_count}")

    # ------------------------------------------------------------------
    # Optional PNG confusion matrices
    # ------------------------------------------------------------------
    if args.save_png:
        print("\n[PNG] Saving confusion matrix images ...")
        for label, df in dfs.items():
            safe_name = label.replace("\n", "_").replace(" ", "_").replace("(", "").replace(")", "")
            png_path = os.path.join(results_dir, f"cm_kg_finetune_{safe_name}.png")
            save_confusion_png(df, label.replace("\n", " "), png_path)

    print("\n[DONE] Comparison complete.")


if __name__ == "__main__":
    main()
