"""
run_coord_experiment.py — Zero-Shot Coordinate Prediction (Direct Comparison)
===========================================================================
Reads the first 10 VALID cities from your old eval_outputs_real/.jsonl files.
Cross-references worldcities100k.pkl to find the true City Name.
Skips any city where the old Base model failed (NaN / None).
"""

import os
import json
import math
import argparse
import pandas as pd
from tqdm import tqdm

from geographic_kg_wikidata import DynamicWikidataCoordGraph
from reasoning_strategies_wikidata import get_strategy, STRATEGY_MAP

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def run_experiment(strategy_name: str, sample_size: int = 10):
    dataset_path = "eval_outputs_real/eval_cot_outputs.jsonl"
    pkl_path = "worldcities100k.pkl"
    
    print(f"🚀 Loading original city dataset from {pkl_path}...")
    if not os.path.exists(pkl_path):
        print(f"❌ Error: Could not find {pkl_path}")
        return
        
    # Build Reverse-Lookup Dictionary
    df = pd.read_pickle(pkl_path)
    coord_to_city = {}
    for _, row in df.iterrows():
        lat = round(float(row['Latitude']), 4)
        lon = round(float(row['Longitude']), 4)
        coord_to_city[(lat, lon)] = row['AccentCity']
        
    print(f"🚀 Extracting {sample_size} valid cities from {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Could not find {dataset_path}")
        return
        
    cities_to_test = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            
            gold_coords = record.get("gold_norm", [])
            if not gold_coords or len(gold_coords) != 2: continue
            
            true_lat = float(gold_coords[0])
            true_lon = float(gold_coords[1])
            
            lookup_key = (round(true_lat, 4), round(true_lon, 4))
            city_name = coord_to_city.get(lookup_key)
            
            if not city_name:
                continue 
            
            old_error = record.get("error_km")
            
            # 🔥 THE FIX: Skip cities where the old model failed (N/A)
            if old_error is None:
                continue
            
            cities_to_test.append({
                "name": city_name,
                "true_lat": true_lat,
                "true_lon": true_lon,
                "old_err": old_error
            })
            
            if len(cities_to_test) >= sample_size:
                break
                
    print(f"🎯 Selected {len(cities_to_test)} valid cities for 1:1 comparison.\n")
    for c in cities_to_test:
        print(f"   - {c['name']} (Old Error: {c['old_err']:.1f} km)")
    
    kg = DynamicWikidataCoordGraph(cache_path="results/wikidata_coord_cache.json")
    strategies_to_run = list(STRATEGY_MAP.keys()) if strategy_name == "all" else [strategy_name]
    os.makedirs("results", exist_ok=True)
    
    for strat in strategies_to_run:
        print(f"\n{'='*70}\nEvaluating Strategy: {strat.upper()}\n{'='*70}")
        strategy_obj = get_strategy(strat, kg)
        
        results = []
        log_file = f"results/coord_eval_wikidata_{strat}.txt"
        
        with open(log_file, "w", encoding="utf-8") as lf:
            for idx, city in tqdm(enumerate(cities_to_test), total=len(cities_to_test)):
                target = city["name"]
                true_lat, true_lon = city["true_lat"], city["true_lon"]
                old_err = city["old_err"]
                
                def logger(msg):
                    lf.write(msg + "\n"); lf.flush()
                
                logger(f"\n--- City {idx+1} / {len(cities_to_test)} ---")
                logger(f"Predicting coordinates for: {target}")
                logger(f"True Coordinates: Lat {true_lat:.4f}, Lon {true_lon:.4f}")
                
                # Fetch semantic evidence
                evidence = kg.build_evidence_block(target)
                
                # Predict new coordinates
                pred_coords, trace = strategy_obj.reason(target, evidence, log_fn=logger)
                
                old_err_disp = f"{old_err:.1f} km"
                
                if pred_coords:
                    new_err = haversine(true_lat, true_lon, pred_coords[0], pred_coords[1])
                    logger(f"✅ LLM Prediction: Lat {pred_coords[0]}, Lon {pred_coords[1]}")
                    logger(f"📊 COMPARISON -> Old Base Error: {old_err_disp}  |  New Wiki Error: {new_err:.1f} km")
                else:
                    new_err = None
                    logger(f"❌ Failed to extract valid coordinates.")
                    logger(f"📊 COMPARISON -> Old Base Error: {old_err_disp}  |  New Wiki Error: N/A (Failed)")
                    
                results.append({"city": target, "old_err": old_err, "new_err": new_err})
        
        valid_new = [r['new_err'] for r in results if r['new_err'] is not None]
        valid_old = [r['old_err'] for r in results if r['old_err'] is not None]
        
        if valid_new:
            mae_new = sum(valid_new) / len(valid_new)
            mae_old = sum(valid_old) / len(valid_old) if valid_old else float('nan')
            print(f"📊 {strat.upper()} Complete! (10 Cities)")
            print(f"   Old Base MAE: {mae_old:.1f} km")
            print(f"   New Wiki MAE: {mae_new:.1f} km")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGY_MAP.keys()) + ["all"], default="all")
    args = parser.parse_args()
    
    run_experiment(args.strategy, sample_size=10)