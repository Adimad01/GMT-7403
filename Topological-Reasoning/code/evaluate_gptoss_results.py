#!/usr/bin/env python3
"""Evaluate GPT-OSS predictions for zero-shot and few-shot runs.

Usage:
  python evaluate_gptoss_results.py \
    --pred-zero "..\results\gptoss_preds_30_zero.jsonl" \
    --pred-few "..\results\gptoss_preds_30_few.jsonl" \
    --out-dir "..\results\eval"

The script reads JSONL prediction files produced by `run_gptoss_inference.py`,
canonicalizes both the gold (`spatial_relation`) and predicted values, computes
accuracy and per-class precision/recall/f1, and writes a JSON report and a
detailed CSV for each input file.
"""

import os
import argparse
import json
import csv
from collections import defaultdict

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
except Exception:
    classification_report = None
    confusion_matrix = None
    accuracy_score = None


SYN_MAP = {
    "contains": ["contains", "is home to", "contains/has", "has", "includes", "include", "home to"],
    "within": ["is within", "within", "inside", "in"],
    "touches": ["touches", "adjacent", "borders", "adjacent to", "bordered"],
    "crosses": ["crosses", "straddles", "spans", "crossed"],
    "disjoint": ["disjoint", "between", "separate", "separated"],
    "overlaps": ["overlaps", "overlap", "extends into", "partly"],
}


def canonicalize(text: str):
    if text is None:
        return None
    txt = str(text).lower()
    # bracketed style: extract first bracket
    if "[" in txt and "]" in txt:
        start = txt.find("[") + 1
        end = txt.find("]", start)
        if end > start:
            inside = txt[start:end]
            for candidate in [p.strip() for p in inside.split("/") if p.strip()]:
                for canon, syns in SYN_MAP.items():
                    if candidate == canon or candidate in syns:
                        return canon
            return inside.split("/")[0].strip()

    # fallback: search for synonyms
    for canon, syns in SYN_MAP.items():
        for s in syns:
            if s in txt:
                return canon
    # as last resort, return original token if it matches a canonical key
    for canon in SYN_MAP.keys():
        if canon in txt:
            return canon
    return None


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # skip malformed
                continue
    return rows


def evaluate_rows(rows):
    y_true = []
    y_pred = []
    details = []
    for r in rows:
        gold_raw = r.get("spatial_relation") or r.get("relation_predicate") or r.get("vernacular_relation")
        pred_raw = r.get("predicate") or r.get("generated")
        gold = canonicalize(gold_raw)
        pred = canonicalize(pred_raw)
        correct = (gold is not None and pred is not None and gold == pred)
        details.append({
            "gold_raw": gold_raw,
            "pred_raw": pred_raw,
            "gold": gold,
            "pred": pred,
            "correct": correct,
            **{k: v for k, v in r.items() if k not in ("spatial_relation", "predicate", "generated", "relation_predicate")} 
        })
        y_true.append(gold)
        y_pred.append(pred)
    return y_true, y_pred, details


def safe_accuracy(y_true, y_pred):
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if not pairs:
        return 0.0
    correct = sum(1 for t, p in pairs if t == p)
    return correct / len(pairs)


def run_one(path, out_dir):
    rows = read_jsonl(path)
    y_true, y_pred, details = evaluate_rows(rows)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    csv_path = os.path.join(out_dir, base + "_details.csv")
    json_path = os.path.join(out_dir, base + "_report.json")

    # write details CSV
    fieldnames = ["gold_raw", "pred_raw", "gold", "pred", "correct"] + [k for k in (rows[0].keys() if rows else []) if k not in ("spatial_relation","predicate","generated","relation_predicate")]
    with open(csv_path, "w", newline='', encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for d in details:
            writer.writerow(d)

    report = {}
    report["n_examples"] = len(rows)
    report["n_scored"] = sum(1 for t, p in zip(y_true, y_pred) if t is not None and p is not None)
    report["accuracy"] = safe_accuracy(y_true, y_pred)

    # classification report if sklearn available
    if classification_report is not None:
        # replace None with 'None' label for sklearn
        y_true_f = [t if t is not None else "__NONE__" for t in y_true]
        y_pred_f = [p if p is not None else "__NONE__" for p in y_pred]
        report["classification_report"] = classification_report(y_true_f, y_pred_f, zero_division=0, output_dict=True)
        report["confusion_matrix"] = confusion_matrix(y_true_f, y_pred_f).tolist()
    else:
        # simple per-class counts
        counts = defaultdict(lambda: {"tp": 0, "pred_count": 0, "true_count": 0})
        for t, p in zip(y_true, y_pred):
            if t is not None:
                counts[t]["true_count"] += 1
            if p is not None:
                counts[p]["pred_count"] += 1
            if t is not None and p is not None and t == p:
                counts[t]["tp"] += 1
        summary = {}
        for k, v in counts.items():
            tp = v["tp"]
            pc = v["pred_count"]
            tc = v["true_count"]
            prec = tp / pc if pc else 0.0
            rec = tp / tc if tc else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            summary[k] = {"precision": prec, "recall": rec, "f1": f1, "support": tc}
        report["per_class"] = summary

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2, ensure_ascii=False)

    # Write a human-readable text report
    txt_path = os.path.join(out_dir, base + "_report.txt")
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(f"File: {path}\n")
        tf.write(f"Examples: {report['n_examples']}\n")
        tf.write(f"Scored (gold+pred present): {report['n_scored']}\n")
        tf.write(f"Accuracy: {report['accuracy']:.4f}\n\n")
        if classification_report is not None:
            y_true_f = [t if t is not None else "__NONE__" for t in y_true]
            y_pred_f = [p if p is not None else "__NONE__" for p in y_pred]
            tf.write("Classification report:\n")
            try:
                from sklearn.metrics import classification_report as sk_cr
                tf.write(sk_cr(y_true_f, y_pred_f, zero_division=0))
            except Exception:
                tf.write(json.dumps(report.get("classification_report", {}), indent=2, ensure_ascii=False))
            tf.write("\n")
        else:
            tf.write("Per-class summary:\n")
            for k, v in report.get("per_class", {}).items():
                tf.write(f"  {k}: precision={v['precision']:.3f}, recall={v['recall']:.3f}, f1={v['f1']:.3f}, support={v['support']}\n")

    # Confusion matrix CSV and optional PNG
    try:
        # prepare labels
        y_true_f = [t if t is not None else "__NONE__" for t in y_true]
        y_pred_f = [p if p is not None else "__NONE__" for p in y_pred]
        if confusion_matrix is not None:
            labels = sorted(list(set(y_true_f) | set(y_pred_f)))
            cm = confusion_matrix(y_true_f, y_pred_f, labels=labels)
            # save CSV
            cm_csv = os.path.join(out_dir, base + "_confusion.csv")
            with open(cm_csv, "w", newline='', encoding="utf-8") as cf:
                w = csv.writer(cf)
                w.writerow([""] + labels)
                for lab, row in zip(labels, cm.tolist()):
                    w.writerow([lab] + row)
            # save PNG if matplotlib available
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel='Predicted', ylabel='True', title='Confusion Matrix')
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                thresh = cm.max() / 2.0 if cm.max() else 0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
                fig.tight_layout()
                png_path = os.path.join(out_dir, base + "_confusion.png")
                fig.savefig(png_path)
                plt.close(fig)
            except Exception:
                pass
        else:
            # fallback: build a simple confusion matrix using known canonical labels
            labels = list(SYN_MAP.keys())
            idx = {l: i for i, l in enumerate(labels)}
            cm = [[0 for _ in labels] for _ in labels]
            for t, p in zip(y_true, y_pred):
                if t in idx and p in idx:
                    cm[idx[t]][idx[p]] += 1
            cm_csv = os.path.join(out_dir, base + "_confusion.csv")
            with open(cm_csv, "w", newline='', encoding="utf-8") as cf:
                w = csv.writer(cf)
                w.writerow([""] + labels)
                for lab, row in zip(labels, cm):
                    w.writerow([lab] + row)
    except Exception:
        pass

    return report, csv_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-zero", default=r"..\results\gptoss_preds_30_zero.jsonl")
    parser.add_argument("--pred-few", default=r"..\results\gptoss_preds_30_few.jsonl")
    parser.add_argument("--out-dir", default=r"..\results\eval")
    args = parser.parse_args()

    results = {}
    if os.path.exists(args.pred_zero):
        r0 = run_one(args.pred_zero, args.out_dir)
        results["zero"] = r0[0]
    else:
        print("Warning: pred-zero file not found:", args.pred_zero)

    if os.path.exists(args.pred_few):
        r1 = run_one(args.pred_few, args.out_dir)
        results["few"] = r1[0]
    else:
        print("Warning: pred-few file not found:", args.pred_few)

    # combined summary
    summary_path = os.path.join(args.out_dir, "combined_report.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(results, sf, indent=2, ensure_ascii=False)

    print("Evaluation completed. Reports written to:")
    print(" -", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
