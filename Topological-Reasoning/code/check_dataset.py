import os
import pandas as pd
p = r"c:\Users\imadl\OneDrive\Documents\Session Autmn 2025\IFT-7026\Topological-Reasoning\dataset\triplet_update_v3_70.csv"
print('checking path:', p)
print('exists:', os.path.exists(p))
if os.path.exists(p):
    df = pd.read_csv(p, nrows=5)
    print('rows sample:', len(df))
    print('cols:', list(df.columns))
else:
    print('file not found')
