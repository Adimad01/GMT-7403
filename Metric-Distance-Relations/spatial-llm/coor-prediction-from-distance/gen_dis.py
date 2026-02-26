import argparse
import json
import pandas as pd
import requests
import os
import shutil
from tqdm import tqdm

# Configuration
MODEL = "gpt-oss"
BASE_URL = "http://ollama.apps.crdig.ulaval.ca"
API_ENDPOINT = f"{BASE_URL}/api/generate"

def query_ollama(prompt):
    payload = {"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1, "max_tokens": 400}}
    try:
        response = requests.post(API_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get('response', "").strip()
    except Exception:
        return ""

def get_few_shot_prefix(cities, current_idx, p_type):
    # Sample 3 rows that aren't the current one from the FULL pool to ensure variety
    samples = cities.drop(current_idx).sample(3)
    prefix = "Here are some examples:\n\n"

    for _, row in samples.iterrows():
        if p_type == 'p1':
            prefix += f"Question: Distance between {row['a_name']} and {row['b_name']}? Answer: {row['distance']}\n"
        elif p_type == 'p2':
            a_coords = f"({row['a_lat']:.4f}, {row['a_lon']:.4f})"
            b_coords = f"({row['b_lat']:.4f}, {row['b_lon']:.4f})"
            prefix += f"Question: Distance between {row['a_name']} {a_coords} and {row['b_name']} {b_coords}? Answer: {row['distance']}\n"
        elif p_type == 'p3':
            prefix += f"Question: Distance between {row['a_name']} and {row['b_name']}? Answer: The distance is approximately {row['distance']} km. This is calculated using spatial geometry.\n"
        elif p_type == 'p4':
            a_coords = f"({row['a_lat']:.4f}, {row['a_lon']:.4f})"
            b_coords = f"({row['b_lat']:.4f}, {row['b_lon']:.4f})"
            prefix += f"Question: Distance between {row['a_name']} {a_coords} and {row['b_name']} {b_coords}? Answer: Using the Haversine formula on these coordinates, the distance is {row['distance']} km.\n"

    return prefix + "\nNow fulfill the following:\n"

def gen_dis(cities, p_type, shots, output_file):
    results = []
    processed_keys = set()

    # --- CHECKPOINT LOADING ---
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                results = data
                # Identify which (A, B) pairs are already in this specific output file
                processed_keys = {(r['a_name'], r['b_name']) for r in results}
                print(f"--> Resuming {p_type}_{shots}: Found {len(processed_keys)} existing entries.")
        except Exception as e:
            print(f"--> Warning: Could not load existing file {output_file}. Starting fresh. Error: {e}")

    # Start loop from the DataFrame (already sliced to 1000 in __main__)
    for i, each in tqdm(cities.iterrows(), total=cities.shape[0], desc=f"Processing {p_type} ({shots})"):
        # Skip pairs already saved in the JSON
        if (each["a_name"], each["b_name"]) in processed_keys:
            continue

        # Coordinates only needed in p2/p4; guard for datasets without lat/lon
        if p_type in ('p2', 'p4'):
            a_c = f"({each['a_lat']:.4f}, {each['a_lon']:.4f})"
            b_c = f"({each['b_lat']:.4f}, {each['b_lon']:.4f})"

        # 1. Base Prompt Construction
        if p_type == 'p1':
            core_prompt = f"Question: What is the distance in kilometers between {each['a_name']} and {each['b_name']}? Answer exactly with the number only.\nAnswer:"
        elif p_type == 'p2':
            core_prompt = f"Question: What is the distance in kilometers between {each['a_name']} {a_c} and {each['b_name']} {b_c}? Answer exactly with the number only.\nAnswer:"
        elif p_type == 'p3':
            core_prompt = f"Question: What is the distance in kilometers between {each['a_name']} and {each['b_name']}? Explain your reasoning and how the distance is found.\nAnswer:"
        elif p_type == 'p4':
            core_prompt = f"Question: What is the distance in kilometers between {each['a_name']} {a_c} and {each['b_name']} {b_c}? Explain your reasoning and how the distance is found using these coordinates.\nAnswer:"

        # 2. Few-Shot Logic
        final_prompt = core_prompt
        if shots == '3-shot':
            prefix = get_few_shot_prefix(cities, i, p_type)
            final_prompt = prefix + core_prompt

        generated_text = query_ollama(final_prompt)

        # 3. Store and Atomic Save
        res = each.to_dict()
        res.update({'output': generated_text, 'prompt': final_prompt, 'p_type': p_type, 'shots': shots})
        results.append(res)
        
        # Atomic save
        temp_file = output_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(results, f, indent=4)
        shutil.move(temp_file, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_type', choices=['p1', 'p2', 'p3', 'p4'], required=True)
    parser.add_argument('--shots', choices=['zero-shot', '3-shot'], required=True)
    parser.add_argument('--dataset', default='cities_with_coords.pkl', help='Pickle dataset to use (e.g., cities_opt.pkl)')
    parser.add_argument('--subset-percent', type=float, default=1.0, help='Fraction of rows to sample (e.g., 0.3 for 30%%)')
    parser.add_argument('--subset-ids-output', default=None, help='Path to save sampled IDs (list of {a_name,b_name})')
    parser.add_argument('--output-tag', default=None, help='Suffix tag to append to output filename (e.g., 30percentage)')
    parser.add_argument('--max-rows', type=int, default=1000, help='Cap rows for bounded runs (default 1000)')
    args = parser.parse_args()

    # Load dataset
    cities_df = pd.read_pickle(args.dataset)

    # First, limit to the base experiment scope (e.g. first 1000 rows), THEN sample
    if args.max_rows:
        cities_df = cities_df.head(args.max_rows)

    # Optional subset sampling
    if args.subset_percent and 0.0 < args.subset_percent < 1.0:
        cities_df = cities_df.sample(frac=args.subset_percent, random_state=42).copy()
        
    cities_subset = cities_df
    
    # Persist sampled IDs if requested
    if args.subset_ids_output:
        try:
            ids = [{"a_name": r["a_name"], "b_name": r["b_name"]} for _, r in cities_subset.iterrows()]
        except KeyError:
            # Fallback to index if expected keys are missing
            ids = [{"index": int(idx)} for idx in cities_subset.index]
        with open(args.subset_ids_output, 'w', encoding='utf-8') as f:
            json.dump(ids, f, indent=2)

    print(f"Experiment rows: {len(cities_subset)} selected (dataset={args.dataset}, subset={args.subset_percent})")

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    suffix = f'_{args.output_tag}' if args.output_tag else ''
    out_path = f'outputs/gen_dis_{args.p_type}_{args.shots}{suffix}.json'

    # Pass the subset to the generation function
    gen_dis(cities_subset, args.p_type, args.shots, out_path)