import json
import os

FOLDER = "results/"
# 1. Your OSM reference file (the one with the IDs you want to isolate)
# Change this if your 21-sample file has a different name!
REFERENCE_FILE = os.path.join(FOLDER, "voletc_dynamic_osm_neighborhood_details_spatial_relation_16_sample_cot_neighborhood_details_spatial_relation_16_sample_ckpt.json")

# 2. The old static KG files you want to filter
TARGET_FILES = {
    "CoT (Static)": os.path.join(FOLDER, "voletc_static_kg_neighborhood_details_spatial_relation_cot_neighborhood_details_spatial_relation_ckpt.json"),
    "ToT (Static)": os.path.join(FOLDER, "voletc_static_kg_neighborhood_details_spatial_relation_tot_neighborhood_details_spatial_relation_ckpt.json"),
    "GoT (Static)": os.path.join(FOLDER, "voletc_static_kg_neighborhood_details_spatial_relation_got_neighborhood_details_spatial_relation_ckpt.json")
}

def main():
    """Load OSM reference IDs, filter static KG result files, and save accuracy-annotated subsets."""
    if not os.path.exists(REFERENCE_FILE):
        print(f"❌ Could not find reference file: {REFERENCE_FILE}")
        return

    # Load the IDs from the OSM run
    with open(REFERENCE_FILE, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
    
    # Extract the indices (row IDs)
    target_ids = set(ref_data.get("processed_indices", []))
    print(f"🎯 Extracted {len(target_ids)} target IDs from the OSM experiment.\n")
    print("-" * 50)

    # Process each old file
    for label, filename in TARGET_FILES.items():
        if not os.path.exists(filename):
            print(f"⚠️ Missing {label} file: {filename}")
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        all_results = data.get("results", [])
        
        # Keep only the rows whose index is in our target_ids
        filtered_results = [r for r in all_results if r.get("index") in target_ids]
        
        # Calculate the new accuracy for just this subset
        if filtered_results:
            correct = sum(1 for r in filtered_results if r.get("match") is True)
            acc = (correct / len(filtered_results)) * 100
        else:
            acc = 0.0
            
        # Save the filtered data to a new JSON file
        out_filename = os.path.join(FOLDER, f"filtered_subset_static_{label[:3].lower()}.json")
        
        with open(out_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "subset_size": len(filtered_results),
                "accuracy": acc,
                "results": filtered_results
            }, f, indent=2, ensure_ascii=False)
            
        print(f"✅ {label}:")
        print(f"   Saved to : {out_filename}")
        print(f"   Accuracy : {acc:.2f}% (Matched {len(filtered_results)} rows)")
        print("-" * 50)

if __name__ == "__main__":
    main()