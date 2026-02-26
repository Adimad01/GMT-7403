import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Get directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load original dataset
file_path = os.path.join(BASE_DIR, 'cities.pkl')
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    exit(1)

print(f"Loading {file_path}...")
df = pd.read_pickle(file_path)
print(f"Original dataset size: {len(df)}")

# Split 30% and 70%
# Use train_test_split. test_size=0.7 means 70% for test (df_70), and 30% for train (df_30).
df_30, df_70 = train_test_split(df, test_size=0.7, random_state=42, shuffle=True)

print(f"30% split size: {len(df_30)}")
print(f"70% split size: {len(df_70)}")

# Save to new pickle files
path_30 = os.path.join(BASE_DIR, 'cities_30.pkl')
path_70 = os.path.join(BASE_DIR, 'cities_70.pkl')

df_30.to_pickle(path_30)
df_70.to_pickle(path_70)

print(f"Saved 30% split to {path_30}")
print(f"Saved 70% split to {path_70}")