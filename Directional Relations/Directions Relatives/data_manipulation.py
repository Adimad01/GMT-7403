import os
import json
import random
from pathlib import Path

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_GLOBAL_DIR = os.path.join(BASE_DIR, "SpatialEvalLLM", "map_global")
MERGED_OUTPUT_DIR = os.path.join(BASE_DIR, "SpatialEvalLLM", "merged_data")

def process_files():
    if not os.path.exists(MERGED_OUTPUT_DIR):
        os.makedirs(MERGED_OUTPUT_DIR)

    jsonl_files = sorted(Path(MAP_GLOBAL_DIR).glob("*.jsonl"))
    
    all_30_data = []
    all_70_data = []

    print(f"Found {len(jsonl_files)} files in {MAP_GLOBAL_DIR}")

    for filepath in jsonl_files:
        print(f"Processing {filepath.name}...")
        
        file_data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    item['origin_file'] = filepath.name
                    file_data.append(item)
        
        # Shuffle specifically for this file
        # Using a fixed seed for reproducibility if needed, but random is generally fine for splitting
        random.seed(42) 
        random.shuffle(file_data)
        
        split_idx = int(len(file_data) * 0.3)
        data_30 = file_data[:split_idx]
        data_70 = file_data[split_idx:]
        
        all_30_data.extend(data_30)
        all_70_data.extend(data_70)

    # Save merged 30%
    merged_30_path = os.path.join(MERGED_OUTPUT_DIR, "merged_global_30.jsonl")
    with open(merged_30_path, 'w', encoding='utf-8') as f:
        for item in all_30_data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(all_30_data)} items to {merged_30_path}")

    # Save merged 70%
    merged_70_path = os.path.join(MERGED_OUTPUT_DIR, "merged_global_70.jsonl")
    with open(merged_70_path, 'w', encoding='utf-8') as f:
        for item in all_70_data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(all_70_data)} items to {merged_70_path}")

if __name__ == "__main__":
    process_files()
