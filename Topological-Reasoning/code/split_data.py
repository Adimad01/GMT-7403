import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Get directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check relative path to dataset
# Based on ls output:
# c:\Users\imadl\OneDrive\Documents\Session Autmn 2025\IFT-7026\Topological-Reasoning\code
# c:\Users\imadl\OneDrive\Documents\Session Autmn 2025\IFT-7026\Topological-Reasoning\dataset\triplet_update_v3.csv
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'dataset'))
FILE_PATH = os.path.join(DATASET_DIR, 'triplet_update_v3.csv')

if not os.path.exists(FILE_PATH):
    print(f"Error: {FILE_PATH} not found.")
    exit(1)

print(f"Loading {FILE_PATH}...")
df = pd.read_csv(FILE_PATH)
print(f"Original dataset size: {len(df)}")

# Drop rows with missing values in important columns
df = df.dropna(subset=['Sentence', 'spatial_relation'])
print(f"Dataset size after dropna: {len(df)}")


# Split 30% and 70%
# Use train_test_split. test_size=0.7 means 70% for test (df_70), and 30% for train (df_30).
# Stratify by 'spatial_relation' to maintain distribution
try:
    df_30, df_70 = train_test_split(df, test_size=0.7, stratify=df['spatial_relation'], random_state=42, shuffle=True)
except ValueError as e:
    print(f"Warning: Stratification failed ({e}). Splitting without stratify.")
    df_30, df_70 = train_test_split(df, test_size=0.7, random_state=42, shuffle=True)

print(f"30% split size: {len(df_30)}")
print(f"70% split size: {len(df_70)}")

# Save to new CSV files in the SAME directory as the original dataset
path_30 = os.path.join(DATASET_DIR, 'triplet_update_v3_30.csv')
path_70 = os.path.join(DATASET_DIR, 'triplet_update_v3_70.csv')

df_30.to_csv(path_30, index=False)
df_70.to_csv(path_70, index=False)

print(f"Saved 30% split to {path_30}")
print(f"Saved 70% split to {path_70}")
