#!/usr/bin/env python3
"""Evaluate a model (Ollama) on relative spatial questions using CoT/ToT/GoT.
Features: 
- Evaluates on the 30% test split.
- Fetches map graphs from the 70% Global KG dictionary (Zero Data Leakage).
- Resilient JSON loading.
- Robust checkpointing & tqdm progress bars.
"""

import argparse
import json
import os
import time
from typing import List, Dict, Any, Tuple
import urllib.request
import urllib.error
from tqdm import tqdm

from reasoning_prompt_eval import (
    serialize_graph_lines,
    build_cot_prompt,
    build_tot_prompt,
    build_got_prompt,
    extract_json_like,
    extract_answer_from_cot,
    normalize_answer,
)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def load_global_kg(path: str) -> Dict[str, Any]:
    """Highly resilient loader for the Global Knowledge Graph dictionary."""
    kg = {}
    try:
        # First, try loading it as a standard single JSON object
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If it fails (e.g., extra data / appended lines), parse line-by-line safely
        print(f"\n[Info] {path} has extra data. Running recovery parser...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        if isinstance(data, dict):
                            kg.update(data)
                    except json.JSONDecodeError:
                        continue
    return kg

def load_checkpoint(out_file: str) -> Tuple[set, int, int]:
    """Reads existing outputs to restore progress and metrics."""
    processed_questions = set()
    correct_count = 0
    total_count = 0
    
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    processed_questions.add(item['question'])
                    total_count += 1
                    if item.get('correct') is True:
                        correct_count += 1
                except Exception:
                    continue
    return processed_questions, correct_count, total_count

def call_ollama(prompt: str, base_url: str, model: str, max_retries: int = 3) -> str:
    """Calls Ollama with stream=False and hard limits to prevent hanging."""
    url = base_url.rstrip('/') + '/api/generate'
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': 0.0,
            'top_p': 0.9,
            'num_predict': 1024  # High limit for ToT/GoT JSON generation
        }
    }
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=180.0) as resp:
                raw = resp.read().decode('utf-8')
                j = json.loads(raw)
                return j.get('response', '')
        except urllib.error.URLError as e:
            print(f"\n[Warning] Connection error (Attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(3)
        except Exception as e:
            print(f"\n[Warning] Unexpected error: {e}")
            time.sleep(3)
            
    return ""

def evaluate(mode: str, test_data: List[Dict[str, Any]], global_kg: Dict[str, Any], out_dir: str, base_url: str, model_name: str, dry_run: bool):
    """Runs evaluation for a single reasoning mode and writes results and logs to disk."""
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'eval_{mode.lower()}_outputs.jsonl')
    log_file = os.path.join(out_dir, f'eval_{mode.lower()}_reasoning_log.txt') # <-- Added log file path
    
    # Checkpoint Initialization
    processed_qs, correct_count, total_count = load_checkpoint(out_file)
    
    # Filter remaining data
    remaining_data = [d for d in test_data if d.get('question') not in processed_qs]
    
    print(f"\nMode: {mode}")
    print(f"Already Processed: {total_count} | Remaining: {len(remaining_data)}")
    
    if len(remaining_data) == 0:
        print("All items completed for this mode!")
        return

    # Open BOTH files in append mode for live saving
    with open(out_file, 'a', encoding='utf-8') as outf, \
         open(log_file, 'a', encoding='utf-8') as logf:  
        
        for item in tqdm(remaining_data, desc=f"{mode} Processing", unit="items"):
            q = item.get('question','')
            gold = item.get('answer', '')
            origin = item.get('origin_file', '')
            
            # Fetch the specific map topology for this question using the origin_file
            triples = global_kg.get(origin, [])
            
            # Skip if no question or no triples found in the global dictionary
            if not q or not triples:
                continue
                
            graph_text = serialize_graph_lines(triples, max_lines=150)
            
            if mode == 'CoT':
                prompt = build_cot_prompt(q, graph_text)
            elif mode == 'ToT':
                prompt = build_tot_prompt(q, graph_text)
            else:
                prompt = build_got_prompt(q, graph_text)

            if dry_run:
                raw = f"Answer: {gold}" if mode == 'CoT' else json.dumps({'final_answer': gold})
            else:
                raw = call_ollama(prompt, base_url, model_name)

            # Parse Output
            if mode == 'CoT':
                ans = extract_answer_from_cot(raw)
            else:
                parsed = extract_json_like(raw)
                ans = None
                if isinstance(parsed, dict):
                    ans = parsed.get('final_answer') or parsed.get('answer')

            norm_pred = normalize_answer(ans if ans else raw)
            norm_gold = normalize_answer(gold)

            # Strict comparison, allows for substring matches if model babbles
            correct = False
            if norm_gold and norm_pred:
                if norm_pred == norm_gold or norm_gold in norm_pred:
                    correct = True

            total_count += 1
            if correct:
                correct_count += 1

            # 1. Save Live JSON Checkpoint
            result_payload = {
                'origin_file': origin, 
                'question': q, 
                'gold': gold, 
                'raw': raw, 
                'pred': ans, 
                'pred_norm': norm_pred, 
                'correct': correct
            }
            outf.write(json.dumps(result_payload, ensure_ascii=False) + '\n')
            outf.flush()

            # 2. Save Live Full Reasoning Log
            log_block = (
                f"==================================================\n"
                f"Mode:        {mode}\n"
                f"Map Origin:  {origin}\n"
                f"Question:    {q}\n"
                f"Expected:    {gold}\n"
                f"--- LLM RAW REASONING & OUTPUT ---\n"
                f"{raw}\n"
                f"==================================================\n\n"
            )
            logf.write(log_block)
            logf.flush()

            # Optional: Print live accuracy underneath the tqdm bar
            live_acc = correct_count / total_count if total_count else 0.0
            tqdm.write(f"Live accuracy ({total_count} evaluated): {live_acc:.3f}")

    # Final Metrics Save
    acc = correct_count / total_count if total_count else 0.0
    with open(os.path.join(out_dir, f'metrics_{mode.lower()}.json'), 'w', encoding='utf-8') as mf:
        json.dump({'mode': mode, 'accuracy': acc, 'total': total_count, 'correct': correct_count}, mf, indent=2)

    print(f"\nMode {mode} Completed: accuracy={acc:.3f} ({correct_count}/{total_count})")
    
def main():
    """Parses CLI arguments, loads data, and runs the selected evaluation modes."""
    parser = argparse.ArgumentParser(description='Evaluate CoT/ToT/GoT using Ollama on Extracted KG')
    
    parser.add_argument('--test-data', default='SpatialEvalLLM/merged_global_30.jsonl', help='Path to 30% test dataset')
    parser.add_argument('--kg-data', default='global_kg_from_70.json', help='Path to global_kg_from_70.json')
    parser.add_argument('--out-dir', required=True, help='Directory to write per-mode outputs')
    parser.add_argument('--mode', choices=['CoT','ToT','GoT','all'], default='all')
    parser.add_argument('--use-ollama', action='store_true')
    parser.add_argument('--base-url', default='http://ollama.apps.crdig.ulaval.ca')
    parser.add_argument('--model-name', default='gpt-oss')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--sample-size', type=int, default=0, help='Process only first N items (0=all)')
    args = parser.parse_args()

    print("Loading test data and global Knowledge Graph...")
    test_data = load_jsonl(args.test_data)
    global_kg = load_global_kg(args.kg_data)
    
    if args.sample_size and args.sample_size > 0:
        test_data = test_data[:args.sample_size]

    modes = ['CoT','ToT','GoT'] if args.mode == 'all' else [args.mode]
    for m in modes:
        evaluate(m, test_data, global_kg, args.out_dir, args.base_url, args.model_name, args.dry_run)

if __name__ == '__main__':
    main()