import argparse
import json
import pandas as pd
import os
import requests
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# --- Configuration ---
REMOTE_URL = "http://ollama.apps.crdig.ulaval.ca"
LOCAL_URL = "http://localhost:11434"
MODEL_NAME = "gpt-oss"
REQUEST_TIMEOUT = 120  # Increased timeout to 120 seconds for slow servers

# --- Helper Function: API Query ---
def query_ollama(api_client, payload, max_retries=5):
    url = f"{api_client['base_url']}/api/generate"
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            wait_time = 5 * (attempt + 1)
            tqdm.write(f"    ⏱️  Timeout (attempt {attempt+1}/{max_retries}). Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                print(f"    [API Error] Failed after {max_retries} attempts: Request timeout")
                return None
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                print(f"    [API Error] Failed after {max_retries} attempts: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None

# --- Core Logic ---
def gen_sen(api_client, cities, p_type, p_length, state, checkpoint_file):
    results = []
    done_cities = set()

    # Load Checkpoint
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', [])
                for r in results:
                    if 'name' in r:
                        done_cities.add(r['name'])
            print(f"[*] Resuming: {len(done_cities)} cities already done.")
        except json.JSONDecodeError:
            print("[!] Checkpoint corrupted. Starting fresh.")

    # Load Templates
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
        print("[!] Error: Templates not found.")
        return []

    print(f"[*] Processing {len(cities)} cities...")
    
    for i, (idx, each) in enumerate(cities.iterrows()):
        city_name = each["name"]
        
        if city_name in done_cities:
            continue

        tqdm.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing City {i+1}/{len(cities)}: {city_name}")

        res = each.to_dict()
        prompt = ""
        
        # Prompt Logic
        if p_length == 'zero-shot':
            c_name_fmt = each["name"] if state else each["name"].split(",")[0]
            if p_type == 'near': prompt = template[0].format(city=c_name_fmt)
            elif p_type == 'and': prompt = template[1].format(city=c_name_fmt)
            elif p_type == 'far': prompt = template[2].format(city=c_name_fmt)
            elif p_type == 'close': prompt = template[3].format(city=c_name_fmt)
                
        elif p_length == '3-shot':
            # Note: For 3-shot, we need to sample from the full dataframe if possible, 
            # but here we sample from the passed 'cities' subset to keep it self-contained.
            remaining = cities.loc[cities.index != idx]
            
            # Safety check for small datasets (like 10 cities)
            if len(remaining) < 3:
                print("[!] Warning: Not enough data for 3-shot sampling. Duplicating samples.")
                remaining = pd.concat([remaining] * 3) # Hack to make it work for tiny sets
                
            def fmt_dict(d): return {k: v.split(",")[0] for k, v in d.items()} if not state else d

            if p_type == 'near':
                c = remaining.sample(3)
                args = {'city_a': c.iloc[0]["name"], 'city_b': c.iloc[0].near_city, 'city_c': c.iloc[1]["name"], 'city_d': c.iloc[1].near_city, 'city_e': c.iloc[2]["name"], 'city_f': c.iloc[2].near_city, 'city': each["name"]}
                prompt = template[4].format(**fmt_dict(args))
            elif p_type == 'and':
                c = remaining.sample(6, replace=True) # replace=True allows sampling if len < 6
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

        # Generation Loop
        outputs = []
        payload = {
            'model': api_client['model_name'],
            'prompt': prompt,
            'stream': False,
            'temperature': 0.9,
            'top_k': 100,
            'num_predict': 10,
        }

        print(f"  > Generating 2 samples...", end='', flush=True)
        for sample_i in range(2):
            response_json = query_ollama(api_client, payload)
            if response_json and 'response' in response_json:
                outputs.append(response_json['response'])
            else:
                outputs.append("")
            if (sample_i + 1) % 10 == 0:
                print(f" {(sample_i + 1)//10 * 20}%", end='', flush=True)
        print(" Done.") 

        res.update({
            'output': outputs,
            'prompt': prompt,
            'p_type': p_type,
            'p_length': p_length,
            'state': 'state' if state else 'no-state',
        })
        
        results.append(res)
        done_cities.add(city_name)
        
        if checkpoint_file:
            with open(checkpoint_file, 'w') as f:
                json.dump({'results': results}, f)
                f.flush()
                os.fsync(f.fileno())

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', choices=['true', 'false'], default='false')
    parser.add_argument('--p_type', required=True, choices=['near', 'far', 'close', 'and'])
    parser.add_argument('--p_length', required=True, choices=['zero-shot', '3-shot'])
    parser.add_argument('--state', required=True, choices=['true', 'false'])
    # NEW ARGUMENT FOR CUSTOM DATA FILE
    parser.add_argument('--data', dest='data_file', default='cities.pkl', help='Path to pickle data file')
    # NEW ARGUMENT FOR OUTPUT DIR
    parser.add_argument('--out_dir', dest='output_dir', default='outputs_gpt_oss', help='Directory to save outputs')
    
    args = parser.parse_args()
    
    use_local = True if args.local == 'true' else False
    state = True if args.state == 'true' else False
    state_rec = 'state' if state else 'no-state'
    base_url = LOCAL_URL if use_local else REMOTE_URL

    if not os.path.exists(args.data_file):
        print(f"[!] Error: Data file '{args.data_file}' not found.")
        exit(1)
    
    cities = pd.read_pickle(args.data_file)
    
    # Checkpoint setup
    checkpoint_dir = 'checkpoints'
    Path(checkpoint_dir).mkdir(exist_ok=True)
    checkpoint_filename = f'checkpoint-{args.p_length}-{args.p_type}-{state_rec}.json'
    checkpoint_file = os.path.join(checkpoint_dir, checkpoint_filename)

    # Output setup
    output_dir = args.output_dir
    Path(output_dir).mkdir(exist_ok=True)
    output_filename = f'gen_sen-{args.p_length}-{args.p_type}-{state_rec}.json'
    final_output_path = os.path.join(output_dir, output_filename)

    api_client = {'base_url': base_url, 'model_name': MODEL_NAME}

    try:
        result = gen_sen(api_client, cities, args.p_type, args.p_length, state, checkpoint_file)
        
        with open(final_output_path, 'w+') as f:
            json.dump(result, f, indent=4)
        
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            
        print(f"[*] SUCCESS. Results saved to: {final_output_path}")

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
        exit(0)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        exit(1)