import os
import re
import json
import pandas as pd

STRATEGIES = ["cot", "tot", "got"]

def parse_text_log(filepath):
    """Parses OSM or Wikidata text logs to extract results."""
    results = {}
    if not os.path.exists(filepath):
        return results

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    pairs = re.split(r'---\s+Pair\s+\d+\s+/\s+\d+\s+---', content)[1:]
    for pair_text in pairs:
        est_match = re.search(r'Estimating:\s+(.*?)\s+to\s+(.*?)\n', pair_text)
        if not est_match: continue
        
        city_a, city_b = est_match.group(1).strip(), est_match.group(2).strip()
        pair_key = frozenset([city_a, city_b])

        pred_match = re.search(r'✅ Prediction:\s+([\d\.]+)\s+km\s+\|\s+Error:\s+([\d\.]+)\s+km', pair_text)
        if pred_match:
            results[pair_key] = float(pred_match.group(2)) # Store the Error
        else:
            results[pair_key] = None
            
    return results

def load_base_json(filepath):
    """Loads Base JSON checkpoints."""
    results = {}
    if not os.path.exists(filepath): return results

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for res in data.get("results", []):
        pair_key = frozenset([res.get("a_name", "").strip(), res.get("b_name", "").strip()])
        results[pair_key] = res.get("error")
        
    return results

def main():
    summary_data = []

    print("=" * 105)
    print(" 10-INSTANCE COMPARISON: BASE (Static) vs OSM (Coords) vs WIKIDATA (Coords + Semantics)")
    print("=" * 105)

    for strat in STRATEGIES:
        osm_file = f"metric_eval_osm_{strat}.txt"
        wiki_file = f"metric_eval_wikidata_{strat}.txt"
        base_file = f"voletc_base_{strat}_ckpt.json"

        # Load all three sets of results
        osm_results = parse_text_log(osm_file)
        wiki_results = parse_text_log(wiki_file)
        base_results = load_base_json(base_file)

        valid_osm_errs, valid_wiki_errs, valid_base_errs = [], [], []

        # Iterate using the keys found in the OSM run as the baseline "Target 10"
        for pair_key in osm_results.keys():
            osm_err = osm_results.get(pair_key)
            wiki_err = wiki_results.get(pair_key)
            base_err = base_results.get(pair_key)

            if osm_err is not None: valid_osm_errs.append(osm_err)
            if wiki_err is not None: valid_wiki_errs.append(wiki_err)
            if base_err is not None: valid_base_errs.append(base_err)

        osm_mae = sum(valid_osm_errs) / len(valid_osm_errs) if valid_osm_errs else float('nan')
        wiki_mae = sum(valid_wiki_errs) / len(valid_wiki_errs) if valid_wiki_errs else float('nan')
        base_mae = sum(valid_base_errs) / len(valid_base_errs) if valid_base_errs else float('nan')

        summary_data.append({
            "Strategy": strat.upper(),
            "Target": len(osm_results),
            "Base MAE": f"{base_mae:.1f}",
            "OSM MAE": f"{osm_mae:.1f}",
            "Wiki MAE": f"{wiki_mae:.1f}",
            "Base Valid": f"{len(valid_base_errs)}/10",
            "OSM Valid": f"{len(valid_osm_errs)}/10",
            "Wiki Valid": f"{len(valid_wiki_errs)}/10"
        })

    df_summary = pd.DataFrame(summary_data)
    
    print(df_summary.to_string(index=False))
    print("=" * 105)
    
    df_summary.to_csv("comparison_3way_metric.csv", index=False)
    print("\n✅ Saved detailed 3-way comparison to 'comparison_3way_metric.csv'.")

if __name__ == "__main__":
    main()