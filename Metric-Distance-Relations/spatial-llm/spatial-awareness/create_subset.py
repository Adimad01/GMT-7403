import pandas as pd
import os

# Configuration
INPUT_FILE = 'cities.pkl'
OUTPUT_FILE = 'cities_10.pkl'
SUBSET_SIZE = 10

if not os.path.exists(INPUT_FILE):
    print(f"[!] Error: {INPUT_FILE} not found.")
    exit(1)

# Load and slice
df = pd.read_pickle(INPUT_FILE)
df_subset = df.head(SUBSET_SIZE)

# Save
df_subset.to_pickle(OUTPUT_FILE)
print(f"[*] Success! Created '{OUTPUT_FILE}' with {len(df_subset)} cities.")
print(f"[*] The first city is: {df_subset.iloc[0]['name']}")