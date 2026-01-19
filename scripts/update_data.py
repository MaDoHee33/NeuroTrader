
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
    symbols = ["XAUUSD", "BTCUSD"] 
    timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"] 
    
    # Destination
    catalog_path = ROOT_DIR / "data" / "nautilus_catalog"
    catalog_path.mkdir(parents=True, exist_ok=True)
    
    # Safety Check: Do not save Mock data over real data!
    if getattr(driver, 'is_mock', False):
         print("⚠️  driver is in MOCK MODE. Skipping save to protect real data.")
         print("    (Please run this script on the machine with real MT5 terminal running)")
         driver.shutdown()
         return

    try:
        for symbol in symbols:
            print(f"\n🪙 Processing {symbol}...")
            for tf in timeframes:
                print(f"  🔄 Fetching {tf}...", end=" ", flush=True)
                
                # Fetch recent history (adjust count as needed, e.g. 10000)
                df = await driver.fetch_history(symbol=symbol, timeframe=tf, count=10000)
                
                if df is not None and not df.empty:
                    # Format for Nautilus (Optional, but good practice to have raw parquet)
                    # Name format: SYMBOL.SIM-TF-LAST-EXTERNAL.parquet
                    # e.g. XAUUSD.SIM-15-MINUTE-LAST-EXTERNAL.parquet (standardizing names)
                    
                    # Convert 'M15' -> '15-MINUTE'
                    tf_str = tf.upper()
                    if tf_str.startswith('M') and not tf_str.startswith('MN'):
                         dur = tf_str[1:]
                         unit = 'MINUTE'
                    elif tf_str.startswith('H'):
                         dur = tf_str[1:]
                         unit = 'HOUR'
                    elif tf_str.startswith('D'):
                         dur = '1'
                         unit = 'DAY'
                    elif tf_str.startswith('W'):
                         dur = '1' # roughly
                         unit = 'WEEK' # Nautilus might map differently, sticking to basic for now
                    elif tf_str.startswith('MN'):
                         dur = '1'
                         unit = 'MONTH'
                    else:
                         dur = '1'
                         unit = 'UNKNOWN'
                    
                    filename = f"{symbol}.SIM-{dur}-{unit}-LAST-EXTERNAL.parquet" 
                    output_path = catalog_path / filename

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
