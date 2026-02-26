import requests
import json
import os
import time
from tqdm import tqdm
from datetime import datetime

def query_ollama(api_client, payload, max_retries=5):
    """
    Sends a request to Ollama with retry logic and error handling.
    """
    url = f"{api_client['base_url']}/api/generate"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                print(f"    [API Error] Failed after {max_retries} attempts: {e}")
                return None
            time.sleep(1 * (attempt + 1))
    return None

def gen_sen(
        api_client,
        cities,
        p_type='near',
        p_length='zero-shot',
        state=True,
        checkpoint_file=None
):

    results = []
    # Track completed (CityName, VariationIndex) pairs to allow granular resuming
    done_variations = set()

    # --- Load Checkpoint ---
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"[*] Loading checkpoint from {checkpoint_file}...")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', [])
                
                # Rebuild the "done" set from the loaded results
                for r in results:
                    if 'name' in r and 'prompt_iteration' in r:
                        done_variations.add((r['name'], r['prompt_iteration']))
                
            print(f"[*] Resuming with {len(results)} entries ({len(done_variations)} variations completed).")
        except json.JSONDecodeError:
            print("[!] Checkpoint file corrupted. Starting over.")

    # --- Load Templates ---
    try:
        template = [
            open('templates/near-zero-shot.txt').read(),
            open('templates/and-zero-shot.txt').read(),
            open('templates/far-zero-shot.txt').read(),
            open('templates/close-zero-shot.txt').read(),
            open('templates/near-3-shot.txt').read(),
            open('templates/and-3-shot.txt').read(),
            open('templates/far-3-shot.txt').read(),
            open('templates/close-3-shot.txt').read(),
        ]
    except FileNotFoundError:
        print("[!] Error: Template files not found in 'templates/' directory.")
        return []

    print(f"[*] Starting processing for {len(cities)} cities...")
    
    # Iterate through cities
    for i, (idx, each) in enumerate(cities.iterrows()):
        city_name = each["name"]
        
        # Check if this city is fully done (all 10 variations)
        # If so, skip silently or log briefly
        is_city_fully_done = all((city_name, v) in done_variations for v in range(10))
        if is_city_fully_done:
            continue

        tqdm.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing City {i+1}/{len(cities)}: {city_name}")

        # Loop 10 times for prompt variations
        for prompt_idx in range(10):
            
            # --- RESUME CHECK: Skip if this specific variation is already saved ---
            if (city_name, prompt_idx) in done_variations:
                # print(f"  > Variation {prompt_idx + 1}/10 already done. Skipping.")
                continue

            print(f"  > Variation {prompt_idx + 1}/10...", end='', flush=True)

            res = each.to_dict()
            prompt = ""
            
            # --- Prompt Construction Logic ---
            if p_length == 'zero-shot':
                c_name_fmt = each["name"] if state else each["name"].split(",")[0]
                if p_type == 'near': prompt = template[0].format(city=c_name_fmt)
                elif p_type == 'and': prompt = template[1].format(city=c_name_fmt)
                elif p_type == 'far': prompt = template[2].format(city=c_name_fmt)
                elif p_type == 'close': prompt = template[3].format(city=c_name_fmt)
                    
            elif p_length == '3-shot':
                remaining = cities.loc[cities.index != idx]
                def fmt_dict(d): return {k: v.split(",")[0] for k, v in d.items()} if not state else d

                if p_type == 'near':
                    c = remaining.sample(3)
                    args = {'city_a': c.iloc[0]["name"], 'city_b': c.iloc[0].near_city, 'city_c': c.iloc[1]["name"], 'city_d': c.iloc[1].near_city, 'city_e': c.iloc[2]["name"], 'city_f': c.iloc[2].near_city, 'city': each["name"]}
                    prompt = template[4].format(**fmt_dict(args))
                elif p_type == 'and':
                    c = remaining.sample(6)
                    args = {'city_a': c.iloc[0]["name"], 'city_b': c.iloc[1]["name"], 'city_c': c.iloc[2]["name"], 'city_d': c.iloc[3]["name"], 'city_e': c.iloc[4]["name"], 'city_f': c.iloc[5]["name"], 'city': each["name"]}
                    prompt = template[5].format(**fmt_dict(args))
                elif p_type == 'far':
                    c = remaining.sample(3)
                    args = {'city_a': c.iloc[0]["name"], 'city_b': c.iloc[0].far_city, 'city_c': c.iloc[1]["name"], 'city_d': c.iloc[1].far_city, 'city_e': c.iloc[2]["name"], 'city_f': c.iloc[2].far_city, 'city': each["name"]}
                    prompt = template[6].format(**fmt_dict(args))
                elif p_type == 'close':
                    c = remaining.sample(3)
                    args = {'city_a': c.iloc[0]["name"], 'city_b': c.iloc[0].near_city, 'city_c': c.iloc[1]["name"], 'city_d': c.iloc[1].near_city, 'city_e': c.iloc[2]["name"], 'city_f': c.iloc[2].near_city, 'city': each["name"]}
                    prompt = template[7].format(**fmt_dict(args))

            # --- API Generation Loop (50 samples) ---
            outputs = []
            payload = {
                'model': api_client['model_name'],
                'prompt': prompt,
                'stream': False,
                'temperature': 0.9,
                'top_k': 100,
                'num_predict': 10,
            }

            for sample_i in range(50):
                response_json = query_ollama(api_client, payload)
                if response_json and 'response' in response_json:
                    outputs.append(response_json['response'])
                else:
                    outputs.append("")
                # Visual feedback
                if (sample_i + 1) % 10 == 0:
                    print(f" {(sample_i + 1)//10 * 20}%", end='', flush=True)

            print(" Done.") 

            res.update({
                'output': outputs,
                'prompt': prompt,
                'p_type': p_type,
                'p_length': p_length,
                'state': 'state' if state else 'no-state',
                'prompt_iteration': prompt_idx
            })
            
            # --- UPDATE STATE ---
            results.append(res)
            done_variations.add((city_name, prompt_idx))
            
            # --- SAVE CHECKPOINT IMMEDIATELY ---
            if checkpoint_file:
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'results': results,
                        # We don't strictly need 'processed_indices' anymore since we use done_variations
                        # but we keep it for backward compatibility if needed.
                        'processed_indices': [] 
                    }, f)
                    f.flush()
                    os.fsync(f.fileno())

    return results