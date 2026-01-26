
import pandas as pd
import glob

DATA_DIR = "data/nautilus_store/data/bar/XAUUSD.SIM-5-MINUTE-LAST-EXTERNAL"
files = glob.glob(DATA_DIR + "/*.parquet")

if not files:
    print("No files found!")
else:
    f = files[0]
    print(f"Reading {f}")
    df = pd.read_parquet(f)
    print("\n--- DTYPES ---")
    print(df.dtypes)
    print("\n--- HEAD ---")
    print(df.head())
    print("\n--- RAW VALUES (First Row) ---")
    print(df.iloc[0].to_dict())
