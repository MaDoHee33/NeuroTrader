
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.brain.feature_eng import add_features

def check_data():
    path = "data/processed/XAUUSD_M5_processed.parquet"
    print(f"Checking {path}...")
    
    if not os.path.exists(path):
        print("File not found!")
        return

    df = pd.read_parquet(path)
    print(f"Raw shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("Head:\n", df.head())
    
    print("\n--- Applying Features ---")
    try:
        df_feat = add_features(df.copy())
        print(f"Shape after features: {df_feat.shape}")
        
        print("\n--- Dropping NaNs ---")
        df_clean = df_feat.dropna()
        print(f"Shape after dropna: {df_clean.shape}")
        
        if len(df_clean) == 0:
            print("❌ All data dropped! Inspecting NaNs...")
            null_counts = df_feat.isnull().sum()
            print(null_counts[null_counts > 0])
            
    except Exception as e:
        print(f"Error adding features: {e}")

if __name__ == "__main__":
    check_data()
