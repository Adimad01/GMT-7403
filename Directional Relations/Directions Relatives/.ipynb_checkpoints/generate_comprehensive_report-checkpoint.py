#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Extracts metrics from JSON result files (both summary-based and raw-list-based),
generates a text leaderboard report, and produces high-impact visualizations.
"""

import json
import os
import glob
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
RESULTS_DIR = "./evaluation_results"
OUTPUT_REPORT = os.path.join(RESULTS_DIR, "FINAL_COMPREHENSIVE_MODELS_REPORT.txt")
# =================================================

def load_json_results(filepath: str) -> dict:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return {}

def extract_stats_from_data(data: dict) -> dict:
    """Attempts multiple methods to find or calculate accuracy stats."""
    # Method 1: Look for pre-calculated 'global_maps'
    if "global_maps" in data:
        for key in ["merged_global_30", "merged_global_70", list(data["global_maps"].keys())[0]]:
            if key in data["global_maps"]:
                stats = data["global_maps"][key]
                return {
                    "total": stats.get("total_questions", stats.get("total", 0)),
                    "correct": stats.get("correct", 0),
                    "accuracy": stats.get("accuracy", 0.0) * 100 if stats.get("accuracy", 0.0) <= 1.0 else stats.get("accuracy", 0.0),
                    "errors": stats.get("errors", 0)
                }

    # Method 2: Calculate manually from a raw list of results
    results_list = None
    for key, value in data.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and "correct" in value[0]:
            results_list = value
            break
            
    if results_list:
        total = len(results_list)
        correct = sum(1 for item in results_list if item.get("correct") is True)
        # Some formats might log errors explicitly, otherwise default to 0
        errors = sum(1 for item in results_list if item.get("error") is not None) 
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "errors": errors
        }

    return None

def parse_metadata_from_filename(filename: str):
    """Derives model, setup, and mode directly from the filename."""
    lname = filename.lower()
    
    # Model
    if "qwen" in lname:
        model = "Qwen-14B"
    elif "gptoss" in lname or "gpt_oss" in lname:
        model = "GPT-OSS-20B"
    else:
        model = "Unknown Model"
        
    # Setup
    if "lora" in lname or "finetuned" in lname or "ft" in lname or "improved" in lname:
        setup = "Finetuned"
    else:
        setup = "Base Model"
        
    # Mode
    if "fewshot" in lname or "3shot" in lname or "k3" in lname:
        mode = "Few-Shot"
    else:
        mode = "Zero-Shot"
        
    return model, setup, mode

def analyze_results(results_dir: str):
    """Finds all json evaluation files and extracts key metrics."""
    search_pattern = os.path.join(results_dir, "*.json")
    result_files = glob.glob(search_pattern)
    
    if not result_files:
        print(f"⚠️ No JSON files found matching: {search_pattern}")
        return []

    summaries = []
    
    for filepath in sorted(result_files):
        filename = os.path.basename(filepath)
        data = load_json_results(filepath)
        
        if not data:
            continue
            
        stats = extract_stats_from_data(data)
        if not stats:
            print(f"⚠️ Skipping {filename}: Could not find or calculate stats.")
            continue
            
        model_family, setup, mode = parse_metadata_from_filename(filename)

        summary = {
            "filename": filename,
            "model_family": model_family,
            "setup": setup,
            "mode": mode,
            "accuracy": stats["accuracy"],
            "correct": stats["correct"],
            "total": stats["total"],
            "errors": stats["errors"]
        }
        summaries.append(summary)
        print(f"✅ Loaded metrics from: {filename} (Acc: {stats['accuracy']:.1f}%)")
        
    return summaries

def write_report(summaries: list, output_path: str):
    if not summaries:
        print("No valid data to write to report.")
        return

    # Sort summaries by Accuracy (Highest first)
    summaries = sorted(summaries, key=lambda x: x['accuracy'], reverse=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 110 + "\n")
        f.write("SPATIAL REASONING COMPREHENSIVE EVALUATION\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 110 + "\n\n")
        
        f.write("LEADERBOARD:\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Model Family':<15} | {'Configuration':<15} | {'Mode':<18} | {'Acc (%)':<8} | {'Score'} | {'Format Errors'}\n")
        f.write("-" * 110 + "\n")
        
        for s in summaries:
            acc_str = f"{s['accuracy']:.2f}%"
            score_str = f"{s['correct']}/{s['total']}"
            f.write(f"{s['model_family']:<15} | {s['setup']:<15} | {s['mode']:<18} | {acc_str:<8} | {score_str:<7} | {s['errors']}\n")
            
        f.write("-" * 110 + "\n\n")
        
        f.write("DETAILED SOURCE MAPPING:\n")
        f.write("=" * 110 + "\n")
        for s in summaries:
            f.write(f"File:      {s['filename']}\n")
            f.write(f"Config:    {s['model_family']} | {s['setup']} | {s['mode']}\n")
            f.write(f"Accuracy:  {s['accuracy']:.2f}% ({s['correct']}/{s['total']})\n")
            f.write(f"Errors:    {s['errors']} invalid formats\n")
            f.write("-" * 40 + "\n")
            
    print(f"\n✅ Text report generated: {output_path}")

def generate_visualizations(summaries: list, output_dir: str):
    if not summaries:
        return
    
    sns.set_theme(style="whitegrid")
    df = pd.DataFrame(summaries)
    df['Config_Mode'] = df['setup'] + " | " + df['mode']
    
    # 1. Grouped Bar Chart
    plt.figure(figsize=(12, 6))
    bar_plot = sns.barplot(data=df, x='model_family', y='accuracy', hue='Config_Mode', palette='viridis')
    plt.title("Spatial Reasoning Accuracy by Model and Configuration", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.xlabel("Model Family", fontsize=12, fontweight='bold')
    plt.ylim(0, 105) 
    
    for p in bar_plot.patches:
        if p.get_height() > 0:
            bar_plot.annotate(f"{p.get_height():.1f}", 
                              (p.get_x() + p.get_width() / 2., p.get_height()), 
                              ha='center', va='bottom', xytext=(0, 4), textcoords='offset points', fontsize=9)

    plt.legend(title="Configuration", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "CHART_1_Grouped_Bar_Comparison.png"), dpi=300)
    plt.close()

    # 2. Performance Heatmap
    try:
        pivot_df = df.pivot_table(index='model_family', columns='Config_Mode', values='accuracy', aggfunc='max')
        plt.figure(figsize=(10, 4))
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=.5, cbar_kws={'label': 'Accuracy (%)'})
        plt.title("Model Configuration Accuracy Matrix", fontsize=14, fontweight='bold')
        plt.ylabel("Model Family", fontweight='bold')
        plt.xlabel("Setup & Prompting Strategy", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "CHART_2_Accuracy_Heatmap.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"⚠️ Could not generate Heatmap: {e}")

    # 3. Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='errors', y='accuracy', hue='model_family', style='setup', s=150, palette='Set1')
    
    for i in range(df.shape[0]):
        plt.text(df['errors'].iloc[i] + 0.1, df['accuracy'].iloc[i], df['mode'].iloc[i], fontsize=8, alpha=0.7)

    plt.title("Logical Accuracy vs. Formatting Errors", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.xlabel("Total Format Errors", fontsize=12, fontweight='bold')
    
    max_errors = max(df['errors'].max() + 2, 5)
    plt.xlim(max_errors, -0.5)
    plt.ylim(0, 105)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "CHART_3_Accuracy_vs_Errors.png"), dpi=300)
    plt.close()

    print("📊 Visualizations generated and saved to the results directory.")

def main():
    print("=" * 50)
    print("🚀 INITIALIZING COMPREHENSIVE REPORT BUILDER")
    print("=" * 50)
    
    summaries = analyze_results(RESULTS_DIR)
    
    if summaries:
        write_report(summaries, OUTPUT_REPORT)
        generate_visualizations(summaries, RESULTS_DIR)
        print("\n🎉 Process complete. Check your evaluation folder for the report and graphs.")
    else:
        print("No summaries could be generated. Please ensure JSON files are present.")

if __name__ == "__main__":
    main()