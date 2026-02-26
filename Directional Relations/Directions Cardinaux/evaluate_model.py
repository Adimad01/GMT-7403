
"""
Task 4 Model Evaluation Script (Fixed)
Evaluates directional reasoning for first 1400 IDs
Computes: precision, recall, F1, accuracy, and confusion matrix
"""

import json
from collections import defaultdict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, 
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ================= CONFIGURATION =================
GROUND_TRUTH = ".\\data\\answers.jsonl"
MODEL_RESULTS = "experiment_results_gptoss.jsonl"
OUTPUT_DIR = "evaluation_results"
MAX_IDS = 1400
EVAL_RATIO = 0.30  # evaluate on 30% of available data
RANDOM_SEED = 42   # reproducible split
VALID_DIRECTIONS = ["north", "south", "east", "west", "north-east", "north-west", "south-east", "south-west"]
# =================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_ground_truth(max_ids=MAX_IDS):
    """Load ground truth answers from answers.jsonl"""
    gt = {}
    with open(GROUND_TRUTH, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            q_id = int(obj['id'])
            if q_id > max_ids:
                continue
            answer = obj.get('absoluteAnswer', '').lower().strip()
            gt[str(q_id)] = answer
    
    print(f"✅ Loaded {len(gt)} ground truth answers (IDs 1-{max_ids})")
    return gt

def load_model_results(max_ids=MAX_IDS):
    """Load model predictions from experiment_resultsgptoss.jsonl"""
    results = defaultdict(lambda: {})  # {id: {temperature: answer}}
    temps = set()
    
    with open(MODEL_RESULTS, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            q_id = int(obj['request']['id'])
            if q_id > max_ids:
                continue
            
            temp = float(obj['request']['temperature'])
            # Extract answer from response content
            answer_text = obj['response']['content'][0]['text'].lower().strip()
            # Clean punctuation
            answer_text = answer_text.replace('.', '').replace(',', '').strip()
            
            results[str(q_id)][temp] = answer_text
            temps.add(temp)
    
    print(f"✅ Loaded {len(results)} model results (IDs 1-{max_ids})")
    print(f"   Temperatures found: {sorted(temps)}")
    return results, sorted(temps)

def normalize_answer(answer):
    """Normalize answer variations"""
    answer = answer.lower().strip()
    answer = answer.replace("northeast", "north-east")
    answer = answer.replace("northwest", "north-west")
    answer = answer.replace("southeast", "south-east")
    answer = answer.replace("southwest", "south-west")
    return answer

def is_valid_direction(answer):
    """Check if answer is a valid direction"""
    return normalize_answer(answer) in VALID_DIRECTIONS

def compute_eval_split_ids(gt, eval_ratio=EVAL_RATIO, random_state=RANDOM_SEED):
    """Compute stratified 30/70 split IDs based on ground truth labels.
    Returns eval_ids (30%) and finetune_ids (70%).
    """
    ids = []
    labels = []
    for q_id_str, ans in gt.items():
        ids.append(int(q_id_str))
        labels.append(normalize_answer(ans))

    df = pd.DataFrame({"id": ids, "label": labels})

    # Perform stratified split; if stratify fails (e.g., single-class), fall back to random split
    try:
        eval_df, finetune_df = train_test_split(
            df,
            test_size=1.0 - eval_ratio,
            stratify=df["label"],
            random_state=random_state,
        )
    except Exception:
        eval_df, finetune_df = train_test_split(
            df,
            test_size=1.0 - eval_ratio,
            random_state=random_state,
        )

    eval_ids = sorted(eval_df["id"].tolist())
    finetune_ids = sorted(finetune_df["id"].tolist())

    # Persist split metadata
    split_meta = {
        "eval_ratio": eval_ratio,
        "random_state": random_state,
        "eval_count": len(eval_ids),
        "finetune_count": len(finetune_ids),
        "max_ids": MAX_IDS,
    }
    with open(f"{OUTPUT_DIR}/id_splits.json", "w", encoding="utf-8") as f:
        json.dump({"eval_ids": eval_ids, "finetune_ids": finetune_ids, "meta": split_meta}, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved split IDs: {OUTPUT_DIR}/id_splits.json")

    # Also persist finetune subset of ground truth for future use
    # Read source JSONL once and filter lines by finetune_ids
    finetune_set = set(finetune_ids)
    src_path = GROUND_TRUTH
    dst_path = f"{OUTPUT_DIR}/answers_finetune_subset.jsonl"
    with open(src_path, "r", encoding="utf-8") as src, open(dst_path, "w", encoding="utf-8") as dst:
        for line in src:
            obj = json.loads(line)
            q_id = int(obj.get("id", -1))
            if q_id in finetune_set and q_id <= MAX_IDS:
                dst.write(line)
    print(f"✅ Saved finetune subset: {dst_path}")

    # Persist eval subset of ground truth
    eval_set = set(eval_ids)
    eval_dst_path = f"{OUTPUT_DIR}/answers_eval_subset.jsonl"
    with open(src_path, "r", encoding="utf-8") as src, open(eval_dst_path, "w", encoding="utf-8") as dst:
        for line in src:
            obj = json.loads(line)
            q_id = int(obj.get("id", -1))
            if q_id in eval_set and q_id <= MAX_IDS:
                dst.write(line)
    print(f"✅ Saved eval subset: {eval_dst_path}")

    return eval_ids, finetune_ids

def evaluate_by_temperature(gt, results, temperatures, max_ids=MAX_IDS, eval_ids=None):
    """Evaluate model performance at each temperature"""
    metrics_by_temp = {}
    
    for temp in temperatures:
        y_true = []
        y_pred = []
        matched_count = 0
        
        # Determine which IDs to evaluate
        ids_to_eval = eval_ids if eval_ids is not None else list(range(1, max_ids + 1))
        for q_id in ids_to_eval:
            q_id_str = str(q_id)
            if q_id_str not in gt or q_id_str not in results:
                continue
            
            if temp not in results[q_id_str]:
                continue
            
            gt_answer = normalize_answer(gt[q_id_str])
            pred_answer = normalize_answer(results[q_id_str][temp])
            
            # Only include valid directions
            if is_valid_direction(gt_answer) and is_valid_direction(pred_answer):
                y_true.append(gt_answer)
                y_pred.append(pred_answer)
                matched_count += 1
        
        if len(y_true) == 0:
            print(f"⚠️  No valid predictions for temperature {temp}")
            continue
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics_by_temp[temp] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'matched_samples': matched_count,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        print(f"  Temperature {temp:.2f}: {matched_count} samples matched")
    
    return metrics_by_temp

def plot_confusion_matrix(y_true, y_pred, temperature):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=VALID_DIRECTIONS)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=VALID_DIRECTIONS, yticklabels=VALID_DIRECTIONS,
                cbar_kws={'label': 'Count'}, square=True, linewidths=0.5)
    plt.title(f'Confusion Matrix - Temperature {temperature:.2f} (First {MAX_IDS} IDs)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    filename = f"{OUTPUT_DIR}/confusion_matrix_temp_{temperature:.2f}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

def plot_metrics_comparison(metrics_by_temp):
    """Plot metrics across temperatures"""
    temps = sorted(metrics_by_temp.keys())
    
    if len(temps) < 2:
        print("⚠️  Not enough temperatures for comparison")
        return
    
    accuracies = [metrics_by_temp[t]['accuracy'] for t in temps]
    precisions = [metrics_by_temp[t]['precision'] for t in temps]
    recalls = [metrics_by_temp[t]['recall'] for t in temps]
    f1s = [metrics_by_temp[t]['f1'] for t in temps]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(temps, accuracies, marker='o', linewidth=2.5, label='Accuracy', markersize=10)
    ax.plot(temps, precisions, marker='s', linewidth=2.5, label='Precision', markersize=10)
    ax.plot(temps, recalls, marker='^', linewidth=2.5, label='Recall', markersize=10)
    ax.plot(temps, f1s, marker='D', linewidth=2.5, label='F1-Score', markersize=10)
    
    ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Performance Metrics Across Temperatures (First {MAX_IDS} IDs)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(temps)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/metrics_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

def generate_report(gt, results, metrics_by_temp, temperatures, eval_ids=None):
    """Generate comprehensive text report"""
    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("TASK 4: DIRECTIONAL REASONING EVALUATION REPORT")
    report_lines.append("=" * 90)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Report sample details for eval subset
    if eval_ids is not None:
        report_lines.append(
            f"Sample: Eval subset ({int(EVAL_RATIO*100)}%), Instances: {len(eval_ids)} of first {MAX_IDS} IDs"
        )
    else:
        report_lines.append(f"Sample Size: First {MAX_IDS} question IDs")
    report_lines.append("")
    
    # Summary metrics table
    report_lines.append("PERFORMANCE METRICS BY TEMPERATURE")
    report_lines.append("-" * 90)
    report_lines.append(f"{'Temp':<10} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Samples':<10}")
    report_lines.append("-" * 90)
    
    for temp in temperatures:
        if temp not in metrics_by_temp:
            continue
        m = metrics_by_temp[temp]
        report_lines.append(
            f"{temp:<10.2f} {m['accuracy']:<15.4f} {m['precision']:<15.4f} "
            f"{m['recall']:<15.4f} {m['f1']:<15.4f} {m['matched_samples']:<10}"
        )
    
    report_lines.append("-" * 90)
    
    # Best performers
    if temperatures:
        best_temp_acc = max(temperatures, key=lambda t: metrics_by_temp[t]['accuracy'] if t in metrics_by_temp else 0)
        best_temp_f1 = max(temperatures, key=lambda t: metrics_by_temp[t]['f1'] if t in metrics_by_temp else 0)
        
        report_lines.append("")
        report_lines.append("BEST PERFORMERS")
        report_lines.append("-" * 90)
        report_lines.append(f"Best Accuracy:  Temperature {best_temp_acc:.2f} ({metrics_by_temp[best_temp_acc]['accuracy']:.4f})")
        report_lines.append(f"Best F1-Score:  Temperature {best_temp_f1:.2f} ({metrics_by_temp[best_temp_f1]['f1']:.4f})")
        report_lines.append("")
    
    # Detailed classification reports
    report_lines.append("DETAILED CLASSIFICATION REPORTS BY TEMPERATURE")
    report_lines.append("=" * 90)
    
    for temp in temperatures:
        if temp not in metrics_by_temp:
            continue
        m = metrics_by_temp[temp]
        report_lines.append(f"\n\nTemperature: {temp}")
        report_lines.append("-" * 90)
        
        clf_report = classification_report(
            m['y_true'], m['y_pred'],
            labels=VALID_DIRECTIONS,
            zero_division=0
        )
        report_lines.append(clf_report)
    
    report_lines.append("\n" + "=" * 90)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 90)
    
    # Write to file
    report_text = "\n".join(report_lines)
    filename = f"{OUTPUT_DIR}/evaluation_report.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✅ Saved: {filename}")
    
    # Print to console
    print("\n" + report_text)

def main():
    print("\n" + "=" * 90)
    print("TASK 4: MODEL EVALUATION (First 1400 IDs)")
    print("=" * 90 + "\n")
    
    # Load data
    print("📂 Loading data...")
    gt = load_ground_truth(MAX_IDS)
    results, temperatures = load_model_results(MAX_IDS)
    
    if not gt or not results or not temperatures:
        print("❌ Failed to load data")
        return
    
    # Split IDs: 30% for evaluation, 70% reserved for future finetuning
    print("\n🔀 Creating 30/70 split (stratified)...")
    eval_ids, finetune_ids = compute_eval_split_ids(gt, EVAL_RATIO, RANDOM_SEED)
    print(f"   Eval IDs: {len(eval_ids)} | Finetune IDs: {len(finetune_ids)}")

    # Evaluate only on 30% eval IDs
    print("\n📊 Evaluating model performance on eval subset (30%)...")
    metrics_by_temp = evaluate_by_temperature(gt, results, temperatures, MAX_IDS, eval_ids=eval_ids)
    
    if not metrics_by_temp:
        print("❌ No valid metrics computed")
        return
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    for temp in sorted(metrics_by_temp.keys()):
        m = metrics_by_temp[temp]
        plot_confusion_matrix(m['y_true'], m['y_pred'], temp)
    
    plot_metrics_comparison(metrics_by_temp)
    
    # Generate report
    print("\n📝 Generating report...")
    generate_report(gt, results, metrics_by_temp, temperatures, eval_ids=eval_ids)
    
    print("\n" + "=" * 90)
    print("✅ EVALUATION COMPLETE")
    print("=" * 90)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}/")
    print(f"   - confusion_matrix_temp_*.png")
    print(f"   - metrics_comparison.png")
    print(f"   - evaluation_report.txt\n")

if __name__ == "__main__":
    main()
