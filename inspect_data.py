import pandas as pd
import pyarrow.parquet as pq

file_path = "data/nautilus_catalog/data/bar/XAUUSD.SIM-15-MINUTE-LAST-EXTERNAL/part-0.parquet"
try:
    df = pd.read_parquet(file_path)
    print("Read successful with Pandas")
    print(df.dtypes)
    print(df.head())
    
    parquet_file = pq.ParquetFile(file_path)
    print("\nParquet Schema:")
    print(parquet_file.schema)
except Exception as e:
    print(f"Error: {e}")
