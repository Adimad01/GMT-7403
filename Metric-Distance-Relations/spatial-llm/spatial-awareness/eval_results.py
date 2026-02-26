import argparse
import pandas as pd
import json
import numpy as np
import re
import os
import glob
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None
try:
    import folium
except Exception:
    folium = None

# --- Configuration ---
RESULTS_DIR = r"C:\Users\imadl\OneDrive\Documents\Session Autmn 2025\IFT-7026\Geospatial Knowledge\spatial-llm\spatial-awareness\outputs_gpt_oss"
RESULTS_FILE = None  # Will be set dynamically, or specify directly
CITIES_DB = "cities.pkl"

def load_reference_data(cities_pkl=CITIES_DB):
    """
    Loads valid cities to act as a lookup dictionary for extraction.
    Returns a dict: {'city name': (lat, lon)}
    """
    df = pd.read_pickle(CITIES_DB)
    df.to_csv("cities.csv")  # Save for inspection/debugging
    # Create a lookup map. 
    # Key: Lowercase city name (e.g., "lubbock")
    # Value: Tuple of (latitude, longitude)
    city_map = {}
    for _, row in df.iterrows():
        # Clean name: "Lubbock, Texas" -> "lubbock"
        clean_name = row['name'].split(',')[0].strip().lower()
        city_map[clean_name] = (float(row['lat']), float(row['lng']))
    return city_map

def extract_first_city(text, input_city_name, city_map):
    """
    Scans the generated text and returns the FIRST city found 
    that is present in city_map and is NOT the input city.
    """
    # Normalize text
    text = text.lower()
    input_clean = input_city_name.split(',')[0].strip().lower()
    
    # We search for known cities in the text.
    # To be efficient and prioritize early mentions, we can iterate through words
    # or use a regex. Given the 'conversational' nature, looking for known names is safest.
    
    # Optimization: Find all start indices of all known cities (Naive approach)
    # For a large DB, Aho-Corasick algorithm is better, but this works for thousands of cities.
    found_candidates = []
    
    for city, coords in city_map.items():
        if city == input_clean:
            continue
            
        # Find position of city in text
        # We add spaces to avoid matching substrings (e.g. "York" in "New York")
        idx = text.find(f" {city} ") 
        if idx != -1:
            found_candidates.append((idx, city, coords))
            
    # Sort by index (find the one mentioned earliest)
    found_candidates.sort(key=lambda x: x[0])
    
    if found_candidates:
        return found_candidates[0][2], found_candidates[0][1]  # (lat,lon), city_name
    return None, None

def process_evaluation(results_path=None, cities_pkl=CITIES_DB, out_dir=None):
    # 1. Load Data
    # Find the results file
    # If caller supplied a path, use it. Otherwise fall back to constants.
    if results_path:
        # if it's a directory, pick first JSON
        if os.path.isdir(results_path):
            json_files = glob.glob(os.path.join(results_path, "*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON files in {results_path}")
            results_path = json_files[0]
        # else assume it's a file path
    elif RESULTS_FILE and os.path.exists(RESULTS_FILE):
        results_path = RESULTS_FILE
    else:
        # Auto-find first .json file in RESULTS_DIR
        json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json files found in {RESULTS_DIR}")
        results_path = json_files[0]
    print(f"[*] Using results file: {results_path}")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
        # Handle if data is wrapped in a dict or list
        if isinstance(data, dict) and 'results' in data:
            results_list = data['results']
        else:
            results_list = data

    city_map = load_reference_data(cities_pkl)
    
    eval_rows = []

    print(f"[*] Evaluating {len(results_list)} cities...")

    for entry in results_list:
        input_name = entry['name']
        input_coords = (float(entry['lat']), float(entry['lng']))
        p_type = entry.get('p_type')
        p_length = entry.get('p_length')
        state = entry.get('state')
        
        # The paper generates 50 samples per prompt [cite: 132]
        outputs = entry['output'] 
        
        for i, gen_text in enumerate(outputs):
            # 2. Extract Location
            target_coords, target_name = extract_first_city(gen_text, input_name, city_map)
            dist_km = np.nan
            status = "FAILED_PARSE"
            if target_coords:
                dist_km = geodesic(input_coords, target_coords).kilometers
                status = "SUCCESS"
            
            eval_rows.append({
                'input_city': input_name,
                'p_type': p_type,
                'p_length': p_length,
                'state': state,
                'sample_idx': i,
                'distance_km': dist_km,
                'status': status,
                'matched_city': target_name,
                'generated_text': gen_text
            })

    # 4. Analyze Results
    df_res = pd.DataFrame(eval_rows)

    # Drop failed parses for statistics
    df_valid = df_res.dropna(subset=['distance_km'])

    print("\n--- Results Summary ---")
    print(f"Total Samples: {len(df_res)}")
    print(f"Successful Parses: {len(df_valid)} ({len(df_valid)/len(df_res):.1%})")

    print("\n--- Mean Distance by Prompt Type ---")
    summary = df_valid.groupby('p_type')['distance_km'].agg(['mean', 'median', 'std', 'count'])
    print(summary)

    # Save raw and valid tables
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        df_res.to_csv(os.path.join(out_dir, 'eval_results_all.csv'), index=False)
        df_valid.to_csv(os.path.join(out_dir, 'eval_results_valid.csv'), index=False)

        # Export failed parses for manual review
        failed = df_res[df_res['status'] != 'SUCCESS']
        failed[['input_city', 'sample_idx', 'generated_text']].to_csv(os.path.join(out_dir, 'failed_parses.csv'), index=False)

    return df_valid, df_res


def summary_and_plots(df_valid, df_res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Basic aggregated tables
    agg_by_plength = df_valid.groupby('p_length')['distance_km'].agg(['count', 'mean', 'median', 'std']).reset_index()
    agg_by_ptype = df_valid.groupby('p_type')['distance_km'].agg(['count', 'mean', 'median', 'std']).reset_index()
    agg_by_state = df_valid.groupby('state')['distance_km'].agg(['count', 'mean', 'median', 'std']).reset_index()
    agg_by_plength.to_csv(os.path.join(out_dir, f'agg_by_p_length_{timestamp}.csv'), index=False)
    agg_by_ptype.to_csv(os.path.join(out_dir, f'agg_by_p_type_{timestamp}.csv'), index=False)
    agg_by_state.to_csv(os.path.join(out_dir, f'agg_by_state_{timestamp}.csv'), index=False)

    # CDF plot per p_length
    plt.figure(figsize=(8,5))
    for name, grp in df_valid.groupby('p_length'):
        data = np.sort(grp['distance_km'].values)
        y = np.arange(1, len(data)+1)/len(data)
        plt.step(data, y, where='post', label=str(name))
    plt.xlabel('Distance (km)')
    plt.ylabel('ECDF')
    plt.legend()
    plt.title('ECDF of distances by prompt length')
    plt.grid(True)
    plt.tight_layout()
    cdf_path = os.path.join(out_dir, f'ecdf_by_p_length_{timestamp}.png')
    plt.savefig(cdf_path)
    plt.close()

    # Histogram (log-scaled bins)
    plt.figure(figsize=(8,5))
    sns.histplot(df_valid['distance_km'].replace(0, 0.001), bins=50, log_scale=(False, True))
    plt.xlabel('Distance (km)')
    plt.title('Histogram (log scale on y) of distances')
    plt.tight_layout()
    hist_path = os.path.join(out_dir, f'hist_dist_{timestamp}.png')
    plt.savefig(hist_path)
    plt.close()

    # Boxplot by prompt length
    plt.figure(figsize=(8,5))
    sns.boxplot(x='p_length', y='distance_km', data=df_valid)
    plt.yscale('symlog', linthresh=1)
    plt.xlabel('Prompt length')
    plt.ylabel('Distance (km)')
    plt.title('Distance by prompt length')
    plt.tight_layout()
    box_path = os.path.join(out_dir, f'box_by_p_length_{timestamp}.png')
    plt.savefig(box_path)
    plt.close()

    # Scatter of distance vs sample index colored by p_length
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='sample_idx', y='distance_km', hue='p_length', data=df_valid, alpha=0.6)
    plt.yscale('symlog', linthresh=1)
    plt.xlabel('Sample index')
    plt.ylabel('Distance (km)')
    plt.title('Distance vs sample index')
    plt.tight_layout()
    scatter_path = os.path.join(out_dir, f'scatter_idx_dist_{timestamp}.png')
    plt.savefig(scatter_path)
    plt.close()

    # Folium map of a sample of successful predictions (if available)
    if folium is not None:
        try:
            m = folium.Map(location=[39, -95], zoom_start=4)
            sample = df_res[df_res['status']=='SUCCESS'].head(200)
            for _, r in sample.iterrows():
                try:
                    src = None
                    tgt = None
                    # attempt to recover coords via cities DB name in matched_city
                    # matched_city is like 'new york'; we need lat/lon from load_reference_data
                    # reload city map
                    city_map = load_reference_data()
                    if isinstance(r['matched_city'], str):
                        tgt = city_map.get(r['matched_city'])
                    src_coords = None
                    # find input city coords via dataframe? We didn't store coords in df_res, so skip if not available
                    # add marker for target if found
                    if tgt:
                        folium.CircleMarker(location=[tgt[0], tgt[1]], radius=3, color='red', fill=True).add_to(m)
                except Exception:
                    continue
            map_path = os.path.join(out_dir, f'map_sample_{timestamp}.html')
            m.save(map_path)
        except Exception:
            pass

    # Statistical tests: compare 3-shot vs zero-shot (if present)
    stats_report = {}
    try:
        group3 = df_valid[df_valid['p_length']=='3-shot']['distance_km'].dropna()
        group0 = df_valid[df_valid['p_length']=='zero-shot']['distance_km'].dropna()
        stats_report['3_shot_count'] = int(len(group3))
        stats_report['zero_shot_count'] = int(len(group0))
        if mannwhitneyu is not None and len(group3)>0 and len(group0)>0:
            ures = mannwhitneyu(group3, group0, alternative='two-sided')
            stats_report['mannwhitneyu_stat'] = float(ures.statistic)
            stats_report['mannwhitneyu_p'] = float(ures.pvalue)
        # bootstrap median difference
        def bootstrap_diff(a,b, n_boot=2000):
            rng = np.random.RandomState(0)
            diffs = []
            for _ in range(n_boot):
                sa = rng.choice(a, size=len(a), replace=True)
                sb = rng.choice(b, size=len(b), replace=True)
                diffs.append(np.median(sa)-np.median(sb))
            return np.percentile(diffs, [2.5,97.5]), np.median(diffs)
        if len(group3)>0 and len(group0)>0:
            ci, med_diff = bootstrap_diff(group3.values, group0.values)
            stats_report['median_diff_boot_median'] = float(med_diff)
            stats_report['median_diff_boot_ci_lo'] = float(ci[0])
            stats_report['median_diff_boot_ci_hi'] = float(ci[1])
    except Exception:
        pass

    # Save stats report
    with open(os.path.join(out_dir, f'stats_report_{timestamp}.json'), 'w', encoding='utf-8') as fh:
        json.dump(stats_report, fh, indent=2)

    # Export top failures (largest distances)
    top_outliers = df_valid.sort_values('distance_km', ascending=False).head(200)
    top_outliers.to_csv(os.path.join(out_dir, f'top_outliers_{timestamp}.csv'), index=False)

    return {
        'cdf': cdf_path,
        'hist': hist_path,
        'box': box_path,
        'scatter': scatter_path,
        'map': map_path if folium is not None else None,
        'stats': os.path.join(out_dir, f'stats_report_{timestamp}.json'),
        'outliers': os.path.join(out_dir, f'top_outliers_{timestamp}.csv')
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate spatial-awareness outputs and produce plots/reports')
    parser.add_argument('--results-file', type=str, default=None, help='Path to a single results JSON file')
    parser.add_argument('--results-dir', type=str, default=None, help='Directory to search for results (overrides default)')
    parser.add_argument('--cities-pkl', type=str, default=CITIES_DB, help='Path to cities pickle used for lookup')
    parser.add_argument('--out-dir', type=str, default='eval_outputs', help='Directory to write evaluation outputs')
    args = parser.parse_args()

    # determine results path
    if args.results_file:
        results_path = args.results_file
    else:
        if args.results_dir:
            # pick first .json in that dir
            files = glob.glob(os.path.join(args.results_dir, '*.json'))
            if not files:
                raise FileNotFoundError(f'No JSON files in {args.results_dir}')
            results_path = files[0]
        else:
            # fallback to previous constant
            json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
            if not json_files:
                raise FileNotFoundError(f"No .json files found in {RESULTS_DIR}")
            results_path = json_files[0]

    print(f"[*] Using results file: {results_path}")
    df_valid, df_res = process_evaluation(results_path=results_path, cities_pkl=args.cities_pkl, out_dir=args.out_dir)
    outputs = summary_and_plots(df_valid, df_res, args.out_dir)
    print('\nGenerated artifacts:')
    for k,v in outputs.items():
        print(f" - {k}: {v}")