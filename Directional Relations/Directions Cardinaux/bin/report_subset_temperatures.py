import json
import argparse
from pathlib import Path
from datetime import datetime

# Optional visuals
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

VALID_DIRECTIONS = [
    "north", "south", "east", "west",
    "north-east", "north-west", "south-east", "south-west",
]


def normalize(answer: str) -> str:
    if answer is None:
        return ""
    a = answer.lower().strip()
    a = a.replace("northeast", "north-east").replace("north east", "north-east")
    a = a.replace("northwest", "north-west").replace("north west", "north-west")
    a = a.replace("southeast", "south-east").replace("south east", "south-east")
    a = a.replace("southwest", "south-west").replace("south west", "south-west")
    a = a.replace(".", "").replace(",", "").strip()
    return a


def load_subset_gt(path: Path):
    gt = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("id"))
            ans = normalize(obj.get("absoluteAnswer", ""))
            if qid and ans:
                gt[qid] = ans
    return gt


def load_results(path: Path):
    from collections import defaultdict
    results = defaultdict(lambda: {})  # id -> {temp: pred}
    temps = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            req = obj.get("request", {})
            qid = str(req.get("id"))
            temp = req.get("temperature")
            content = obj.get("response", {}).get("content", [])
            pred = None
            if content and isinstance(content, list) and content[0].get("text"):
                pred = normalize(content[0]["text"]) 
            if qid and (temp is not None) and pred is not None:
                t = float(temp)
                temps.add(t)
                results[qid][t] = pred
    return results, sorted(temps)


def confusion_matrix(y_true, y_pred, labels):
    # Minimal confusion matrix (avoid hard dep on sklearn)
    idx = {lab: i for i, lab in enumerate(labels)}
    size = len(labels)
    cm = [[0] * size for _ in range(size)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm


def plot_cm(y_true, y_pred, temperature, out_dir: Path, tag: str):
    if not (HAS_MPL and HAS_SNS):
        return None
    cm = confusion_matrix(y_true, y_pred, VALID_DIRECTIONS)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=VALID_DIRECTIONS, yticklabels=VALID_DIRECTIONS,
                cbar_kws={'label': 'Count'}, square=True, linewidths=0.5)
    plt.title(f'Confusion Matrix - Temp {temperature:.2f} ({tag})')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename = out_dir / f"confusion_matrix_{tag}_temp_{temperature:.2f}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return str(filename)


def classification_report(y_true, y_pred):
    # Minimal per-class accuracy summary (avoid hard dep on sklearn)
    from collections import defaultdict
    per_class = defaultdict(lambda: {"support": 0, "correct": 0})
    for t, p in zip(y_true, y_pred):
        per_class[t]["support"] += 1
        if t == p:
            per_class[t]["correct"] += 1
    lines = ["Label        Support  Accuracy"]
    for lab in VALID_DIRECTIONS:
        s = per_class[lab]["support"]
        c = per_class[lab]["correct"]
        acc = (c / s) if s else 0.0
        lines.append(f"{lab:<12} {s:<7} {acc:.4f}")
    return "\n".join(lines)


def build_report(gt, results, temperatures, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=" * 90)
    lines.append(f"SUBSET TEMPERATURE REPORT ({tag})")
    lines.append("=" * 90)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Subset IDs: {len(gt)} | Temperatures: {temperatures}")
    lines.append("")
    lines.append("METRICS BY TEMPERATURE")
    lines.append("-" * 90)
    lines.append(f"{'Temp':<10} {'Matched':<10} {'Accuracy':<10}")
    lines.append("-" * 90)

    for t in temperatures:
        y_true, y_pred = [], []
        matched = 0
        for qid, gt_ans in gt.items():
            pred = results.get(qid, {}).get(t)
            if pred is None:
                continue
            if gt_ans in VALID_DIRECTIONS and pred in VALID_DIRECTIONS:
                y_true.append(gt_ans)
                y_pred.append(pred)
                matched += 1
        acc = (sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)) if y_true else 0.0
        lines.append(f"{t:<10.2f} {matched:<10} {acc:<10.4f}")
    lines.append("-" * 90)

    # Detailed per-temp reports + confusion matrices
    for t in temperatures:
        y_true, y_pred = [], []
        for qid, gt_ans in gt.items():
            pred = results.get(qid, {}).get(t)
            if pred is None:
                continue
            if gt_ans in VALID_DIRECTIONS and pred in VALID_DIRECTIONS:
                y_true.append(gt_ans)
                y_pred.append(pred)
        lines.append("")
        lines.append(f"Temperature: {t}")
        lines.append("-" * 90)
        lines.append(classification_report(y_true, y_pred))
        cm_path = plot_cm(y_true, y_pred, t, out_dir, tag)
        if cm_path:
            lines.append(f"Saved confusion matrix: {cm_path}")

    lines.append("\n" + "=" * 90)
    lines.append("END OF REPORT")
    lines.append("=" * 90)

    report_path = out_dir / f"evaluation_report_{tag}.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="Create text report for subset results across temperatures")
    parser.add_argument("--results", default="experiment_results_gptoss_fewshot.jsonl", help="Results JSONL path")
    parser.add_argument("--subset", default="evaluation_results/answers_eval_subset.jsonl", help="Subset answers JSONL path")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--tag", default="fewshot_subset", help="Tag for filenames")
    parser.add_argument("--expected-temps", nargs="*", type=float, default=None, help="Optional list of temperatures to include in report")
    args = parser.parse_args()

    results_path = Path(args.results)
    subset_path = Path(args.subset)
    out_dir = Path(args.output_dir)

    if not results_path.exists():
        print(f"❌ Results not found: {results_path}")
        return
    if not subset_path.exists():
        print(f"❌ Subset answers not found: {subset_path}")
        return

    gt = load_subset_gt(subset_path)
    results, temps = load_results(results_path)
    if args.expected_temps:
        allowed = set(args.expected_temps)
        temps = sorted([t for t in temps if t in allowed])
    print(f"Loaded GT: {len(gt)} | Results IDs: {len(results)} | Temps: {temps}")

    report_path = build_report(gt, results, temps, out_dir, args.tag)
    print(f"✅ Saved report: {report_path}")


if __name__ == "__main__":
    main()
