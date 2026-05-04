import pandas as pd
import json

# Load the full dataset
full_df = pd.read_pickle('worldcities100k.pkl') 

# Load the 30% evaluation IDs
with open('outputs/subset_ids_gptoss_30pct.json', 'r') as f: 
    holdout_ids = set(json.load(f))

# Isolate the 70% training data for the Knowledge Graph
kg_df = full_df[~full_df.index.isin(holdout_ids)].copy()
print(kg_df.columns)
print(f"Knowledge Graph base: {len(kg_df)} cities.")