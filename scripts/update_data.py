
import asyncio
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.body.mt5_driver import MT5Driver

async def update_data():
    print("🚀 Starting Data Update (via MT5)...")
    
    driver = MT5Driver()
    if not await driver.initialize():
        print("❌ Failed to initialize MT5 Driver.")
        return

    # Configuration
    symbol = "XAUUSD" # Adjust suffix if needed e.g. XAUUSDm
    timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"] 
    
    # Destination
    catalog_path = ROOT_DIR / "data" / "nautilus_catalog"
    catalog_path.mkdir(parents=True, exist_ok=True)

    try:
        for tf in timeframes:
            print(f"🔄 Fetching {symbol} {tf}...")
            
            # Fetch last 10000 candles or date range
            # For update, ideally we fetch range from last known, but here we fetch fresh recent history
            df = await driver.fetch_history(symbol=symbol, timeframe=tf, count=5000)
            
            if df is not None and not df.empty:
                # Format for Nautilus (Optional, but good practice to have raw parquet)
                # Nautilus expects: date, open, high, low, close, volume (standard)
                
                # Save as Parquet (Raw)
                filename = f"{symbol}.SIM-{tf}-LAST-EXTERNAL.parquet" # Naming convention mimicking Nautilus
                # Actually Nautilus catalog ingestion requires specific specific writing via Catalog API
                # But for now, let's save as CSV/Parquet in raw folder so user has it
                
                output_path = catalog_path / filename
                # If using Parquet Catalog, we should technically use Catalog.write_data, 
                # but simple file save works for "Raw Data Update" request.
                
                # Ensure column names are lower case
                df.columns = [c.lower() for c in df.columns]
                
                # Nautilus Compatibility Fix:
                # Rename 'time' -> 'timestamp'
                if 'time' in df.columns:
                    df.rename(columns={'time': 'timestamp'}, inplace=True)
                
                # Convert 'timestamp' to int64 (nanoseconds) if it's datetime
                if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = df['timestamp'].astype('int64') # This gives nanoseconds for dt64[ns]
                
                # Ensure 'volume' exists
                if 'tick_volume' in df.columns and 'volume' not in df.columns:
                     df.rename(columns={'tick_volume': 'volume'}, inplace=True)

                # Save
                df.to_parquet(output_path)
                print(f"✅ Saved {len(df)} rows to {output_path}")
            else:
                print(f"⚠️ No data provided for {tf}")

    finally:
        driver.shutdown()
        print("🏁 Update Complete.")

if __name__ == "__main__":
    asyncio.run(update_data())
