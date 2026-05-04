import argparse
import json
import os
import glob
import logging
import re
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch
from gptoss_predict_coor import load_model, build_prompt

SUBSET_IDS_PATH = "outputs/subset_ids_gptoss_30pct.json"

def main():
    """Load the holdout city subset, run inference, and save per-row checkpoint results."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_type', choices=['0', '1'], required=True)
    parser.add_argument('--p_length', choices=['zero-shot', '3-shot'], required=True)
    parser.add_argument('--adapter', default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    # 1. Load Data
    full_df = pd.read_pickle('worldcities100k.pkl')
    os.makedirs('outputs', exist_ok=True)

    if os.path.exists(SUBSET_IDS_PATH):
        with open(SUBSET_IDS_PATH, 'r') as f: ids = json.load(f)
        cities_30 = full_df[full_df.index.isin(ids)].copy()
    else:
        cities_30 = full_df.sample(frac=0.30, random_state=42).copy()
        with open(SUBSET_IDS_PATH, 'w') as f: json.dump(cities_30.index.tolist(), f)
    
    cities_70 = full_df[~full_df.index.isin(cities_30.index)]

    # 2. Find Existing Checkpoint File dynamically
    model_tag = 'gptoss-lora' if args.adapter else 'gptoss-base'
    search_pattern = f"outputs/results_{model_tag}_{args.p_length}_type{args.p_type}_*.json"
    existing_files = glob.glob(search_pattern)

    results = []
    processed_names = set()
    
    if existing_files:
        existing_files.sort()
        out_file = existing_files[-1] 
        with open(out_file, 'r') as f:
            content = json.load(f)
            if isinstance(content, dict) and 'data' in content:
                results = content['data']
            else:
                results = content
        processed_names = {r['AccentCity'] for r in results}
        log.info(f"Resuming from {out_file}: {len(processed_names)} cities already completed.")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_file = f"outputs/results_{model_tag}_{args.p_length}_type{args.p_type}_{timestamp}.json"
        log.info(f"No existing file found. Creating new: {out_file}")

    # 3. Filter Remaining Cities
    cities_to_run = cities_30[~cities_30['AccentCity'].isin(processed_names)]
    if len(cities_to_run) == 0:
        log.info(f"Experiment {args.p_type}/{args.p_length} already fully completed. Exiting.")
        return

    # 4. Model Inference
    model, tokenizer = load_model(args.adapter, log)
    
    def save_checkpoint():
        """Write the current results list with metadata to the output JSON file."""
        payload = {
            "metadata": {
                "p_type": args.p_type, 
                "p_length": args.p_length, 
                "adapter": args.adapter,
                "completed_cities": len(results)
            },
            "data": results
        }
        with open(out_file, 'w') as f: 
            json.dump(payload, f, indent=2)

    for idx, (i, row) in enumerate(tqdm(cities_to_run.iterrows(), total=len(cities_to_run))):
        sys_msg, user_msg = build_prompt(row.AccentCity, args.p_type, args.p_length, cities_70)
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False, num_beams=3)
        
        raw_pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # REGEX SAFETY NET: Find the coordinates even if the model yaps
        match = re.search(r'(-?\d{1,3}\.\d+),\s*(-?\d{1,3}\.\d+)', raw_pred)
        if match:
            clean_pred = f"{match.group(1)}, {match.group(2)}"
        else:
            clean_pred = raw_pred # Fallback to raw text if no numbers found
        
        res_entry = row.to_dict()
        res_entry.update({
            'output': clean_pred,           # Clean format: "Lat, Long"
            'raw_output': raw_pred,         # Store what the model actually said for debugging
            'p_type': args.p_type, 
            'p_length': args.p_length
        })
        results.append(res_entry)

        # SAVE AFTER EVERY SINGLE ROW
        save_checkpoint()

    log.info(f"Finished. Saved final results to {out_file}")

if __name__ == "__main__":
    main()