"""
analyze_gptoss.py — Base vs Fine-tuned GPT-OSS Evaluation
==========================================================
For each model (base, finetuned) × config (zero-shot, few-shot):
  - Saves a text confusion matrix (.txt)
  - Saves a visual confusion matrix (.png) for the best config only

Run from the code/ directory:
    python analyze_gptoss.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR  = RESULTS_DIR  # confusion matrices saved here

PREDICATES  = ["contains", "within", "touches", "crosses", "overlaps", "disjoint", "equals"]
ALL_LABELS  = PREDICATES + ["invalid"]

FILES = {
    "Base GPT-OSS": {
        "zero_shot": RESULTS_DIR / "gptoss_topological_log_zero_shot_improved.txt",
        "few_shot":  RESULTS_DIR / "gptoss_topological_log_few_shot_improved.txt",
    },
    "Fine-tuned GPT-OSS": {
        "zero_shot": RESULTS_DIR / "gptoss_finetuned_topological_log_zero_shot_improved.txt",
        "few_shot":  RESULTS_DIR / "gptoss_finetuned_topological_log_few_shot_improved.txt",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load(path: Path) -> pd.DataFrame:
    """Load a CSV result file and normalise expected/predicted columns."""
    df = pd.read_csv(path)
    df["expected"]  = df["expected"].astype(str).str.lower().str.strip()
    df["predicted"] = df["predicted"].astype(str).str.lower().str.strip()
    # anything not a known predicate is invalid
    df.loc[~df["predicted"].isin(PREDICATES), "predicted"] = "invalid"
    df["match"] = df["expected"] == df["predicted"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def metrics(df: pd.DataFrame) -> dict:
    """Compute overall accuracy, invalid rate, and per-predicate recall from a results DataFrame."""
    total        = len(df)
    correct      = df["match"].sum()
    invalid      = (df["predicted"] == "invalid").sum()
    accuracy     = correct / total * 100
    invalid_rate = invalid / total * 100

    per_pred = {}
    for pred in PREDICATES:
        mask    = df["expected"] == pred
        support = mask.sum()
        if support == 0:
            per_pred[pred] = {"support": 0, "correct": 0, "accuracy": float("nan")}
        else:
            correct_p = df.loc[mask, "match"].sum()
            per_pred[pred] = {
                "support":  int(support),
                "correct":  int(correct_p),
                "accuracy": correct_p / support * 100,
            }

    return {
        "accuracy":     accuracy,
        "invalid_rate": invalid_rate,
        "correct":      int(correct),
        "total":        total,
        "per_predicate": per_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    df: pd.DataFrame,
    model_name: str,
    config_name: str,
    accuracy: float,
) -> Path:
    """Render and save a visual confusion matrix PNG for the given model and config."""
    y_true = df["expected"].tolist()
    y_pred = df["predicted"].tolist()

    # Only include labels that actually appear
    present = sorted(set(y_true) | set(y_pred))
    # always show all 7 predicates first, then invalid if it appeared
    ordered_preds = [p for p in PREDICATES if p in present]
    if "invalid" in present:
        ordered_preds.append("invalid")

    cm = confusion_matrix(y_true, y_pred, labels=ordered_preds)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Annotate with counts; mask zeros for readability
    mask_zero = cm == 0
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="lightgrey",
        xticklabels=ordered_preds,
        yticklabels=ordered_preds,
        ax=ax,
        annot_kws={"size": 11},
        mask=mask_zero,
        cbar_kws={"shrink": 0.75},
    )
    # Print zeros in a lighter colour
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="lightgrey",
        xticklabels=ordered_preds,
        yticklabels=ordered_preds,
        ax=ax,
        annot_kws={"size": 11, "color": "#cccccc"},
        mask=~mask_zero,
        cbar=False,
        vmin=0,
        vmax=cm.max(),
    )

    config_label = "Few-shot" if config_name == "few_shot" else "Zero-shot"
    ax.set_title(
        f"{model_name}  ·  {config_label}  (best config)\n"
        f"Overall accuracy: {accuracy:.2f}%",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Predicted", fontsize=11, labelpad=10)
    ax.set_ylabel("Expected (Ground Truth)", fontsize=11, labelpad=10)
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", rotation=0,  labelsize=10)

    plt.tight_layout()

    safe = model_name.lower().replace(" ", "_").replace("-", "")
    out  = OUTPUT_DIR / f"cm_gptoss_{safe}_{config_name}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def save_text_confusion_matrix(
    df: pd.DataFrame,
    model_name: str,
    config_name: str,
    m: dict,
    is_best: bool,
) -> Path:
    """Write a formatted plain-text confusion matrix file for a model/config pair."""
    y_true = df["expected"].tolist()
    y_pred = df["predicted"].tolist()

    # Column order: all 7 predicates, then invalid only if it appears
    present = set(y_true) | set(y_pred)
    col_labels = [p for p in PREDICATES if p in present]
    if "invalid" in present:
        col_labels.append("invalid")
    row_labels = [p for p in PREDICATES if p in set(y_true)]

    cm = confusion_matrix(y_true, y_pred, labels=col_labels)
    # row_labels may be a subset of col_labels; slice only those rows
    row_idx = [col_labels.index(r) for r in row_labels]
    cm_rows = cm[row_idx, :]

    # ── layout geometry ──────────────────────────────────────────────────────
    ROW_LABEL_W = 22   # "touches     (n= 95)" width
    COL_W       = 9    # width of each numeric cell / column header
    SEP         = "═" * (ROW_LABEL_W + len(col_labels) * COL_W + 2)

    config_label = "Few-shot" if config_name == "few_shot" else "Zero-shot"
    invalid_n    = int((df["predicted"] == "invalid").sum())

    lines = []
    lines.append(SEP)
    lines.append(f"  Confusion Matrix — {model_name}  |  {config_label}"
                 + ("  ◀ best config" if is_best else ""))
    lines.append(SEP)
    lines.append(f"  Overall Accuracy : {m['accuracy']:.2f}%"
                 f"  ({m['correct']} / {m['total']})")
    lines.append(f"  Invalid Rate     : {m['invalid_rate']:.2f}%"
                 f"  ({invalid_n} / {m['total']})")
    lines.append("")

    # header row
    header = f"  {'Actual \\ Predicted':<{ROW_LABEL_W}}"
    for lbl in col_labels:
        header += f"{lbl:>{COL_W}}"
    lines.append(header)
    lines.append("  " + "─" * (ROW_LABEL_W + len(col_labels) * COL_W))

    # data rows
    for i, row_lbl in enumerate(row_labels):
        support  = m["per_predicate"].get(row_lbl, {}).get("support", 0)
        row_str  = f"  {row_lbl + f' (n={support})':<{ROW_LABEL_W}}"
        for val in cm_rows[i]:
            row_str += f"{int(val):>{COL_W}}"
        lines.append(row_str)

    # ── per-predicate recall ─────────────────────────────────────────────────
    lines.append("")
    lines.append("  Per-predicate Recall:")
    lines.append("  " + "─" * 46)
    for pred, v in m["per_predicate"].items():
        if v["support"] == 0:
            acc_str = "      —"
        else:
            acc_str = f"{v['accuracy']:>6.2f}%"
        lines.append(
            f"  {pred:<10}: {v['correct']:>4} / {v['support']:<4} = {acc_str}"
        )
    lines.append("  " + "─" * 46)
    lines.append(
        f"  {'OVERALL':<10}: {m['correct']:>4} / {m['total']:<4} ="
        f" {m['accuracy']:>6.2f}%"
    )
    lines.append(SEP)

    # ── write file ───────────────────────────────────────────────────────────
    safe = model_name.lower().replace(" ", "_").replace("-", "")
    out  = OUTPUT_DIR / f"cm_{safe}_{config_name}.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def print_section(title: str):
    """Print a prominent section header to stdout."""
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def print_comparison_table(model_name: str, zs: dict, fs: dict):
    """Print a two-row comparison table showing zero-shot vs few-shot metrics for a model."""
    best = "few_shot" if fs["accuracy"] >= zs["accuracy"] else "zero_shot"
    print(f"\n  Model : {model_name}")
    print(f"  {'Config':<14} {'Accuracy':>10}  {'Invalid %':>10}  {'Correct/Total':>14}")
    print(f"  {'-'*52}")
    star_z = " ◀ best" if best == "zero_shot" else ""
    star_f = " ◀ best" if best == "few_shot"  else ""
    print(f"  {'Zero-shot':<14} {zs['accuracy']:>9.2f}%  {zs['invalid_rate']:>9.2f}%  "
          f"{zs['correct']:>6}/{zs['total']:<6}{star_z}")
    print(f"  {'Few-shot':<14} {fs['accuracy']:>9.2f}%  {fs['invalid_rate']:>9.2f}%  "
          f"{fs['correct']:>6}/{fs['total']:<6}{star_f}")


def print_per_predicate_table(model_name: str, config_name: str, m: dict):
    """Print a per-predicate accuracy breakdown table for a model and config."""
    config_label = "Few-shot" if config_name == "few_shot" else "Zero-shot"
    print(f"\n  Per-predicate accuracy — {model_name} [{config_label}]")
    print(f"  {'Predicate':<12} {'Support':>8}  {'Correct':>8}  {'Accuracy':>10}")
    print(f"  {'-'*44}")
    for pred, v in m["per_predicate"].items():
        if v["support"] == 0:
            acc_str = "   —"
        else:
            acc_str = f"{v['accuracy']:>9.2f}%"
        print(f"  {pred:<12} {v['support']:>8}  {v['correct']:>8}  {acc_str}")
    print(f"  {'OVERALL':<12} {m['total']:>8}  {m['correct']:>8}  {m['accuracy']:>9.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Orchestrate loading, metric computation, and output generation for all model/config pairs."""
    print_section("GPT-OSS Topological Evaluation — Base vs Fine-tuned")

    # model_name → {"zero_shot": (df, m), "few_shot": (df, m)}
    all_data    = {}
    best_configs = {}  # model_name → config_name

    # ── Step 1: load all 4 configs, pick best per model ──────────────────────
    for model_name, paths in FILES.items():
        df_zs = load(paths["zero_shot"])
        df_fs = load(paths["few_shot"])
        m_zs  = metrics(df_zs)
        m_fs  = metrics(df_fs)

        all_data[model_name] = {
            "zero_shot": (df_zs, m_zs),
            "few_shot":  (df_fs, m_fs),
        }

        print_comparison_table(model_name, m_zs, m_fs)

        # Best = higher accuracy; tie-break = lower invalid rate
        if m_fs["accuracy"] > m_zs["accuracy"] or (
            m_fs["accuracy"] == m_zs["accuracy"]
            and m_fs["invalid_rate"] < m_zs["invalid_rate"]
        ):
            best_configs[model_name] = "few_shot"
        else:
            best_configs[model_name] = "zero_shot"

    # ── Step 2: text confusion matrix for ALL 4 configurations ───────────────
    print_section("Text Confusion Matrices — All Configurations")

    for model_name, configs in all_data.items():
        best_cfg = best_configs[model_name]
        for config_name, (df_cfg, m_cfg) in configs.items():
            is_best  = (config_name == best_cfg)
            out_path = save_text_confusion_matrix(
                df_cfg, model_name, config_name, m_cfg, is_best
            )
            marker = "  ◀ best" if is_best else ""
            print(f"  Saved → {out_path.name}{marker}")

    # ── Step 3: visual PNG for best config only ───────────────────────────────
    print_section("Visual Confusion Matrices — Best Config per Model")

    for model_name, best_cfg in best_configs.items():
        df_best, m_best = all_data[model_name][best_cfg]
        print_per_predicate_table(model_name, best_cfg, m_best)
        out_path = plot_confusion_matrix(df_best, model_name, best_cfg, m_best["accuracy"])
        print(f"\n  PNG saved → {out_path.name}")

    # ── Step 4: head-to-head summary ─────────────────────────────────────────
    print_section("Head-to-head Summary (Best Config per Model)")
    print(f"\n  {'Model':<22} {'Best Config':<12} {'Accuracy':>10}  {'Invalid %':>10}")
    print(f"  {'-'*58}")
    for model_name, best_cfg in best_configs.items():
        _, m = all_data[model_name][best_cfg]
        label = "Few-shot" if best_cfg == "few_shot" else "Zero-shot"
        print(f"  {model_name:<22} {label:<12} {m['accuracy']:>9.2f}%  {m['invalid_rate']:>9.2f}%")

    base_acc  = all_data["Base GPT-OSS"][best_configs["Base GPT-OSS"]][1]["accuracy"]
    ft_acc    = all_data["Fine-tuned GPT-OSS"][best_configs["Fine-tuned GPT-OSS"]][1]["accuracy"]
    delta     = ft_acc - base_acc
    direction = "▲" if delta >= 0 else "▼"
    print(f"\n  Fine-tuning impact: {direction} {abs(delta):.2f}% accuracy change\n")


if __name__ == "__main__":
    main()
