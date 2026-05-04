import json
import os
import glob

# ─── Configuration ────────────────────────────────────────────────────────────
BACKUP_DIR = "outputs/backup_30percent"
OUTPUTS_DIR = "outputs"
TARGET_IDS_FILE = "100_target_ids.json"

# The exact file you specified to pull the 100 IDs from
SOURCE_FILE = os.path.join(BACKUP_DIR, "gen_dis_qwen14b_p1_3-shot_qwen14b_30percentage.json")
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Check if the source file actually exists
    if not os.path.exists(SOURCE_FILE):
        print(f"❌ Error: Cannot find {SOURCE_FILE}")
        print("Make sure you are running this script from the 'coor-prediction-from-distance' folder.")
        return

    # 2. Extract the 100 IDs
    print(f"📥 Reading source file: {os.path.basename(SOURCE_FILE)}")
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    if len(source_data) < 100:
        print(f"⚠️ Warning: Source file only has {len(source_data)} rows. Grabbing all of them.")
        subset_100 = source_data
    else:
        subset_100 = source_data[:100]

    # Format the IDs and save them
    target_ids = [{"a_name": d["a_name"], "b_name": d["b_name"]} for d in subset_100]
    
    with open(TARGET_IDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(target_ids, f, indent=4)

    target_set = {(item['a_name'], item['b_name']) for item in target_ids}
    print(f"✅ Saved {len(target_set)} target ID pairs to {TARGET_IDS_FILE}\n")

    # 3. Process all backup files and move the filtered versions to outputs/
    backup_files = glob.glob(os.path.join(BACKUP_DIR, "gen_dis*.json"))
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("-" * 70)
    for file_path in backup_files:
        filename = os.path.basename(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Keep only the rows that match our 100 targets
        filtered_data = [d for d in data if (d.get('a_name'), d.get('b_name')) in target_set]
        
        # Save to the main outputs folder
        out_path = os.path.join(OUTPUTS_DIR, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4)
            
        print(f"🔄 Filtered {filename[:40]}... | {len(data):>4} -> {len(filtered_data):>3} rows")

    print("-" * 70)
    print("🎯 Success! The 'outputs/' folder now contains properly aligned 100-row files.")
    print(f"You can now run GPT-OSS using '--input-ids {TARGET_IDS_FILE}'")

if __name__ == "__main__":
    main()