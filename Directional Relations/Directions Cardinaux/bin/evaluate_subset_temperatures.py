import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple


VALID_DIRECTIONS = {
    "north", "south", "east", "west",
    "north-east", "north-west", "south-east", "south-west",
}


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


def load_subset_gt(path: Path) -> Dict[str, str]:
    gt: Dict[str, str] = {}
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


def load_results_by_temp(path: Path) -> Tuple[Dict[str, Dict[float, str]], List[float]]:
    results: Dict[str, Dict[float, str]] = {}
    temps_set = set()
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
                temps_set.add(float(temp))
                results.setdefault(qid, {})[float(temp)] = pred
    return results, sorted(temps_set)


def evaluate(gt: Dict[str, str], results: Dict[str, Dict[float, str]], temps: List[float]):
    # Try to use sklearn for precision/recall/f1 if available
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        use_sklearn = True
    except Exception:
        use_sklearn = False

    summary = []
    for t in temps:
        y_true: List[str] = []
        y_pred: List[str] = []
        matched_ids = 0
        for qid, gt_ans in gt.items():
            preds_for_id = results.get(qid)
            if not preds_for_id:
                continue
            if t not in preds_for_id:
                continue
            pred_ans = preds_for_id[t]
            # Only consider valid normalized labels
            if gt_ans in VALID_DIRECTIONS and pred_ans in VALID_DIRECTIONS:
                y_true.append(gt_ans)
                y_pred.append(pred_ans)
                matched_ids += 1

        metrics = {
            "temperature": t,
            "matched_ids": matched_ids,
            "accuracy": None,
            "precision_weighted": None,
            "recall_weighted": None,
            "f1_weighted": None,
        }
        if y_true:
            # Always compute accuracy
            if use_sklearn:
                metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
                metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
                metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            else:
                correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
                metrics["accuracy"] = correct / len(y_true)

        summary.append(metrics)
    return summary


def save_outputs(summary: List[Dict], out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    json_path = out_dir / f"eval_summary_{tag}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"temperatures": summary}, f, ensure_ascii=False, indent=2)
    # CSV
    csv_path = out_dir / f"eval_summary_{tag}.csv"
    headers = ["temperature", "matched_ids", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for m in summary:
            row = [m.get(h) for h in headers]
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")
    return str(json_path), str(csv_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate subset IDs across temperatures")
    parser.add_argument("--results", default="experiment_results_gptoss_fewshot.jsonl", help="Results JSONL path")
    parser.add_argument("--subset", default="evaluation_results/answers_eval_subset.jsonl", help="Subset answers JSONL path")
    parser.add_argument("--output-dir", default="evaluation_results", help="Directory to write summaries")
    parser.add_argument("--tag", default="subset_temps", help="Tag suffix for output filenames")
    parser.add_argument("--expected-temps", nargs="*", type=float, default=None, help="Optional list of temperatures to evaluate (filters results)")
    args = parser.parse_args()

    results_path = Path(args.results)
    subset_path = Path(args.subset)
    out_dir = Path(args.output_dir)

    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return
    if not subset_path.exists():
        print(f"❌ Subset answers file not found: {subset_path}")
        return

    gt = load_subset_gt(subset_path)
    results, temps = load_results_by_temp(results_path)
    if args.expected_temps:
        allowed = set(args.expected_temps)
        temps = sorted([t for t in temps if t in allowed])
    print(f"Loaded GT IDs: {len(gt)} | Result IDs: {len(results)} | Temps: {temps}")

    summary = evaluate(gt, results, temps)
    json_path, csv_path = save_outputs(summary, out_dir, args.tag)
    print(f"✅ Saved summary JSON: {json_path}")
    print(f"✅ Saved summary CSV:  {csv_path}")
    # Print a concise table to console
    for m in summary:
        print(f"Temp {m['temperature']}: matched={m['matched_ids']}, accuracy={m['accuracy']}")


if __name__ == "__main__":
    main()
