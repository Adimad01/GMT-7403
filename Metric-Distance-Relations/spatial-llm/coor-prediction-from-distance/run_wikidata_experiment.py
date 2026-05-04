"""
run_wikidata_experiment.py — Execution Script for Wikidata Coordinate Experiment
"""

import os
import json
import pandas as pd
import argparse
from tqdm import tqdm

from geographic_kg_wikidata import DynamicWikidataMetricGraph
from reasoning_strategies_wikidata import get_strategy, STRATEGY_MAP

def run_experiment(strategy_name: str, sample_size: int = 10):
    print(f"🚀 Loading Base Checkpoint to ensure we evaluate overlapping pairs...")
    
    # Load the exact pairs that the Base model evaluated
    base_ckpt_path = "results/voletc_base_cot_ckpt.json"
    if not os.path.exists(base_ckpt_path):
        print(f"❌ Error: Could not find {base_ckpt_path}. Ensure you run this from the root directory.")
        return

    with open(base_ckpt_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
        
    target_pairs = []
    for res in base_data.get("results", [])[:sample_size]:
        target_pairs.append({
            'a_name': res['a_name'],
            'b_name': res['b_name'],
            'distance': res['true_distance']
        })
        
    df = pd.DataFrame(target_pairs)
    print(f"🎯 Selected exactly {len(df)} overlapping city pairs from the BASE experiment.\n")
    
    kg = DynamicWikidataMetricGraph(cache_path="results/wikidata_metric_cache.json")
    strategies_to_run = list(STRATEGY_MAP.keys()) if strategy_name == "all" else [strategy_name]
    os.makedirs("results", exist_ok=True)
    
    for strat in strategies_to_run:
        print(f"\n{'='*50}\nEvaluating Strategy: {strat.upper()}\n{'='*50}")
        strategy_obj = get_strategy(strat, kg)
        
        results = []
        log_file = f"results/metric_eval_wikidata_{strat}.txt"
        
        with open(log_file, "w", encoding="utf-8") as lf:
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                city_a, city_b, true_dist = row['a_name'], row['b_name'], row['distance']
                
                def logger(msg):
                    lf.write(msg + "\n")
                    lf.flush()
                
                logger(f"\n--- Pair {idx+1} / {len(df)} ---")
                logger(f"Estimating: {city_a} to {city_b}")
                logger(f"True Distance: {true_dist:.1f} km")
                
                evidence = kg.build_evidence_block(city_a, city_b)
                pred_dist, trace = strategy_obj.reason(city_a, city_b, evidence, log_fn=logger)
                
                if pred_dist is not None:
                    error = abs(pred_dist - true_dist)
                    logger(f"✅ Prediction: {pred_dist} km | Error: {error:.1f} km")
                else:
                    error = None
                    logger(f"❌ Failed to extract valid numeric prediction.")
                    
                results.append({"city_a": city_a, "city_b": city_b, "true_dist": true_dist, "pred_dist": pred_dist, "error": error})
        
        valid_errors = [r['error'] for r in results if r['error'] is not None]
        if valid_errors:
            mae = sum(valid_errors) / len(valid_errors)
            print(f"📊 Strategy {strat.upper()} Completed! MAE: {mae:.1f} km ({len(valid_errors)}/{len(df)} valid)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGY_MAP.keys()) + ["all"], default="all")
    args = parser.parse_args()
    
    run_experiment(args.strategy, sample_size=10)