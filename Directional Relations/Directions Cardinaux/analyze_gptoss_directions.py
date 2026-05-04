"""
analyze_gptoss_directions.py — Cardinal Direction Classification Evaluation
============================================================================
Evaluates Base GPT-OSS and Fine-tuned GPT-OSS on the task of classifying
the cardinal/intercardinal direction of a spatial relationship.

Each question is answered 3 times at temperatures 0.25, 0.5, and 1.0.

Metrics reported:
  - Accuracy per temperature
  - Majority-vote accuracy (best answer across 3 temperatures)
  - Stability (% of questions where all 3 temperatures agree)
  - Per-direction accuracy
  - Confusion matrix (text)

Saves one txt results file per model:
  evaluation_results/results_base_gptoss.txt
  evaluation_results/results_finetuned_gptoss.txt

Run from Directions Cardinaux/:
    python analyze_gptoss_directions.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).parent / "evaluation_results"

VALID_DIRS = [
    "north", "north-east", "east", "south-east",
    "south", "south-west", "west", "north-west",
]

TEMPERATURES = [0.25, 0.5, 1.0]

MODELS = {
    "Base GPT-OSS": {
        "files": {
            "0-shot": EVAL_DIR / "experiment_results_gptoss20b_base_0shot.jsonl",
            "3-shot": EVAL_DIR / "experiment_results_gptoss20b_base_3shot.jsonl",
        },
        "out": EVAL_DIR / "results_base_gptoss.txt",
    },
    "Fine-tuned GPT-OSS": {
        "files": {
            "0-shot": EVAL_DIR / "experiment_results_gptoss20b_finetuned_0shot.jsonl",
            "3-shot": EVAL_DIR / "experiment_results_gptoss20b_finetuned_3shot.jsonl",
        },
        "out": EVAL_DIR / "results_finetuned_gptoss.txt",
    },
}

GT_FILE = EVAL_DIR / "answers_eval_subset.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_ground_truth() -> dict:
    """Loads ground-truth direction labels from the evaluation JSONL file."""
    gt = {}
    with open(GT_FILE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            gt[str(d["id"])] = d["absoluteAnswer"].lower().strip()
    return gt


def extract_direction(text: str) -> str | None:
    """Extracts a valid cardinal direction string from raw model output text."""
    text = text.lower().strip()
    # finetuned model uses "assistantfinal<answer>" pattern
    m = re.search(r"final\s*([a-z\-]+)", text)
    if m and m.group(1).strip() in VALID_DIRS:
        return m.group(1).strip()
    # scan from end of text for any valid direction word
    for w in reversed(re.findall(r"[a-z][a-z\-]*", text)):
        if w in VALID_DIRS:
            return w
    return None


def load_predictions(path: Path, gt: dict) -> dict:
    """
    Returns dict: question_id → list of (temperature, predicted_direction)
    """
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rid  = str(d["request"]["id"])
            temp = d["request"]["temperature"]
            text = d["response"]["content"][0]["text"]
            pred = extract_direction(text)
            groups[rid].append((temp, pred))
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(groups: dict, gt: dict) -> dict:
    """Computes per-temperature accuracy, majority-vote accuracy, stability, and confusion matrix."""
    by_temp   = {t: {"correct": 0, "total": 0, "invalid": 0} for t in TEMPERATURES}
    maj_correct = 0
    stable      = 0
    per_dir_true  = {d: {"correct": 0, "total": 0} for d in VALID_DIRS}
    per_dir_pred  = {d: 0 for d in VALID_DIRS}
    confusion     = {true: {pred: 0 for pred in VALID_DIRS + ["invalid"]}
                     for true in VALID_DIRS}

    for rid, preds in groups.items():
        if rid not in gt:
            continue
        true = gt[rid]

        # per-temperature
        for temp, pred in preds:
            by_temp[temp]["total"]  += 1
            if pred is None:
                by_temp[temp]["invalid"] += 1
                pred_for_cm = "invalid"
            else:
                pred_for_cm = pred
                if pred == true:
                    by_temp[temp]["correct"] += 1

            if true in VALID_DIRS:
                confusion[true][pred_for_cm] += 1

        # majority vote
        votes = [p for _, p in preds if p is not None]
        majority = Counter(votes).most_common(1)[0][0] if votes else None
        if majority == true:
            maj_correct += 1
        if len(set(votes)) == 1 and len(votes) == len(preds):
            stable += 1

        # per-direction (using T=0.25 as reference)
        t025_pred = next((p for t, p in preds if t == 0.25), None)
        if true in VALID_DIRS:
            per_dir_true[true]["total"] += 1
            if t025_pred == true:
                per_dir_true[true]["correct"] += 1

    n_q = len(gt)
    return {
        "by_temp":      by_temp,
        "maj_correct":  maj_correct,
        "stable":       stable,
        "n_q":          n_q,
        "per_dir":      per_dir_true,
        "confusion":    confusion,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_report(model_name: str, model_cfg: dict, gt: dict) -> str:
    """Builds a formatted text report comparing 0-shot and 3-shot configs for a model."""
    SEP  = "═" * 78
    DASH = "─" * 78

    config_results = {}
    for shot_label, path in model_cfg["files"].items():
        groups = load_predictions(path, gt)
        config_results[shot_label] = compute_metrics(groups, gt)

    # best config = highest majority-vote accuracy
    best_shot = max(config_results, key=lambda k: config_results[k]["maj_correct"])
    best_m    = config_results[best_shot]

    lines = [
        SEP,
        f"  Cardinal Direction Classification — {model_name}",
        f"  Task    : Classify the direction of a spatial relationship",
        f"  Classes : {', '.join(VALID_DIRS)}",
        f"  Each question answered 3× at temperatures: {', '.join(str(t) for t in TEMPERATURES)}",
        SEP,
    ]

    # ── accuracy table ────────────────────────────────────────────────────────
    lines += [
        "",
        DASH,
        f"  {'Config':<10}  {'T=0.25':>8}  {'T=0.50':>8}  {'T=1.00':>8}  "
        f"{'Majority':>9}  {'Stability':>10}",
        "  " + "─" * 60,
    ]

    for shot_label, m in config_results.items():
        n_q  = m["n_q"]
        t025 = m["by_temp"][0.25]
        t050 = m["by_temp"][0.5]
        t100 = m["by_temp"][1.0]
        is_best = " ◀" if shot_label == best_shot else ""
        lines.append(
            f"  {shot_label:<10}  "
            f"{t025['correct']/t025['total']*100:>7.2f}%  "
            f"{t050['correct']/t050['total']*100:>7.2f}%  "
            f"{t100['correct']/t100['total']*100:>7.2f}%  "
            f"{m['maj_correct']/n_q*100:>8.2f}%  "
            f"{m['stable']/n_q*100:>9.2f}%{is_best}"
        )

    lines += [
        "  " + "─" * 60,
        f"  Best config: {best_shot}  "
        f"(Majority-vote accuracy = {best_m['maj_correct']}/{best_m['n_q']} = "
        f"{best_m['maj_correct']/best_m['n_q']*100:.2f}%)",
    ]

    # ── per-direction accuracy (best config, T=0.25) ──────────────────────────
    lines += [
        "",
        DASH,
        f"  Per-Direction Accuracy — Best Config ({best_shot}, T=0.25)",
        f"  {'Direction':<14} {'Support':>8}  {'Correct':>8}  {'Accuracy':>10}",
        "  " + "─" * 46,
    ]

    per_dir = best_m["per_dir"]
    for d in VALID_DIRS:
        v = per_dir[d]
        if v["total"] == 0:
            acc_str = "      —"
        else:
            acc_str = f"{v['correct']/v['total']*100:>8.2f}%"
        lines.append(
            f"  {d:<14} {v['total']:>8}  {v['correct']:>8}  {acc_str}"
        )

    t025_total   = best_m["by_temp"][0.25]["total"]
    t025_correct = best_m["by_temp"][0.25]["correct"]
    lines.append(
        f"  {'OVERALL':<14} {t025_total:>8}  {t025_correct:>8}  "
        f"{t025_correct/t025_total*100:>8.2f}%"
    )

    # ── text confusion matrix (best config, all temperatures combined) ─────────
    cm = best_m["confusion"]
    col_labels = VALID_DIRS + ["invalid"]

    # abbreviate labels for compact display
    abbr = {
        "north": "N", "north-east": "NE", "east": "E", "south-east": "SE",
        "south": "S", "south-west": "SW", "west": "W", "north-west": "NW",
        "invalid": "INV",
    }
    col_abbrs = [abbr[c] for c in col_labels]

    COL_W   = 6
    ROW_LAB = 14

    lines += [
        "",
        DASH,
        f"  Confusion Matrix — Best Config ({best_shot}, all temperatures)",
        f"  Rows = Ground Truth  |  Columns = Predicted",
        "",
    ]

    # header
    header = f"  {'Actual \\ Pred':<{ROW_LAB}}"
    for ab in col_abbrs:
        header += f"{ab:>{COL_W}}"
    lines.append(header)
    lines.append("  " + "─" * (ROW_LAB + len(col_labels) * COL_W))

    for true_dir in VALID_DIRS:
        row_str = f"  {true_dir:<{ROW_LAB}}"
        for pred_dir in col_labels:
            val = cm[true_dir].get(pred_dir, 0)
            row_str += f"{val:>{COL_W}}"
        lines.append(row_str)

    # ── invalid / stability detail ────────────────────────────────────────────
    lines += [
        "",
        DASH,
        "  Invalid Predictions & Temperature Stability",
        "  " + "─" * 50,
    ]
    for shot_label, m in config_results.items():
        total_inv = sum(v["invalid"] for v in m["by_temp"].values())
        lines.append(
            f"  {shot_label:<10}  invalid={total_inv}  "
            f"stable={m['stable']}/{m['n_q']} ({m['stable']/m['n_q']*100:.1f}%)"
        )

    lines += ["", SEP]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Loads ground truth, builds reports for each model, and saves results to disk."""
    gt = load_ground_truth()
    print(f"Ground truth loaded: {len(gt)} questions\n")

    for model_name, cfg in MODELS.items():
        report = build_report(model_name, cfg, gt)
        cfg["out"].write_text(report, encoding="utf-8")
        print(f"Saved → {cfg['out'].name}")
        print(report)


if __name__ == "__main__":
    main()
