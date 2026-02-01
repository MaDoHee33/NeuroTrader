
import pandas as pd
import sys

try:
    df = pd.read_parquet("data/processed/XAUUSD_M5_processed.parquet")
    print("Columns:", df.columns.tolist())
    print("Head:", df.head(1).to_dict())
    
    # Check lowercase conversion
    df.columns = df.columns.str.lower()
    print("Columns (Lower):", df.columns.tolist())
    
    if 'close' in df.columns:
        print("SUCCESS: 'close' found.")
    else:
        print("FAILURE: 'close' NOT found.")
        
except Exception as e:
    print(f"Error: {e}")
