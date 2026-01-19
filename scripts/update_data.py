
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
        for base_symbol in symbols:
            print(f"\n🪙 Processing {base_symbol}...")
            
            # Logic to find the correct broker symbol (e.g. XAUUSD vs XAUUSDm)
            # We try base, then base+'m', then base+'c' (common suffixes)
            found_symbol = None
            suffixes_to_try = ['', 'm', 'c', 'pro', '.a']
            
            # Quick check to see which one works (using M1 as probe)
            # However, MT5 copy_rates_from needs 'timeframe' which is passed in loop
            # We will just iterate suffixes inside the logic or pre-check?
            
            # Better approach: Try to fetch for the first timeframe, if fail, try next suffix.
            # Once found, use that suffix for all other timeframes of this symbol.
            
            active_suffix = ""
            
            # Probe check
            for suffix in suffixes_to_try:
                probe_sym = f"{base_symbol}{suffix}"
                # Just check if symbol exists in MT5 usually via symbol_info() but wait... 
                # driver doesn't expose symbol_info directly efficiently for us without async overhead?
                # We can just try fetch M1 count=1
                probe = await driver.fetch_history(symbol=probe_sym, timeframe="M1", count=1)
                if probe is not None and not probe.empty:
                    found_symbol = probe_sym
                    active_suffix = suffix
                    print(f"   ✅ Detected Broker Symbol: {found_symbol}")
                    break
            
            if not found_symbol:
                print(f"   ❌ Could not find valid symbol for {base_symbol} (Tried: {suffixes_to_try})")
                print("      Please ensure the symbol is in Market Watch!")
                continue

            for tf in timeframes:
                # 1. Determine Output Path
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
                        dur = '1'
                        unit = 'WEEK'
                elif tf_str.startswith('MN'):
                        dur = '1'
                        unit = 'MONTH'
                else:
                        dur = '1'
                        unit = 'UNKNOWN'
                
                filename = f"{base_symbol}.SIM-{dur}-{unit}-LAST-EXTERNAL.parquet" 
                output_path = catalog_path / filename
                
                # 2. Check for Existing Data (Incremental Update)
                existing_df = None
                last_time = None
                
                if output_path.exists():
                    try:
                        existing_df = pd.read_parquet(output_path)
                        if not existing_df.empty and 'timestamp' in existing_df.columns:
                            # nautilus timestamp is usually int64 nanoseconds
                            # We need to convert to datetime for MT5
                            last_ts_ns = existing_df['timestamp'].iloc[-1]
                            last_time = pd.to_datetime(last_ts_ns, unit='ns')
                            print(f"  📂 Found existing {filename}, last update: {last_time}")
                    except Exception as e:
                        print(f"  ⚠️ Error reading existing file: {e}. Starting fresh.")
                
                # 3. Fetch Data
                df_new = None
                if last_time:
                    # Incremental: Fetch from last_time to Now
                    print(f"  🔄 Updating {found_symbol} {tf} from {last_time}...", end=" ", flush=True)
                    # Add small buffer? No, let's overlap slightly or trust range
                    df_new = await driver.fetch_history_range(symbol=found_symbol, timeframe=tf, date_from=last_time, date_to=datetime.now())
                else:
                    # Fresh: Fetch max (e.g. 50000 bars)
                    print(f"  🆕 Fetching fresh {found_symbol} {tf} (50,000 bars)...", end=" ", flush=True)
                    df_new = await driver.fetch_history(symbol=found_symbol, timeframe=tf, count=50000)

                # 4. Merge and Save
                if df_new is not None and not df_new.empty:
                    # Normalize New Data
                    df_new.columns = [c.lower() for c in df_new.columns]
                    if 'time' in df_new.columns:
                        df_new.rename(columns={'time': 'timestamp'}, inplace=True)
                    if pd.api.types.is_datetime64_any_dtype(df_new['timestamp']):
                        df_new['timestamp'] = df_new['timestamp'].astype('int64')
                    if 'tick_volume' in df_new.columns and 'volume' not in df_new.columns:
                        df_new.rename(columns={'tick_volume': 'volume'}, inplace=True)
                    
                    # Merge if existing
                    if existing_df is not None and not existing_df.empty:
                        # Concat
                        combined_df = pd.concat([existing_df, df_new])
                        # Deduplicate by timestamp
                        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                        combined_df = combined_df.sort_values(by='timestamp')
                        final_df = combined_df
                        print(f"✅ Merged. Total rows: {len(final_df)} (+{len(df_new)} new)")
                    else:
                        final_df = df_new
                        print(f"✅ Saved. Total rows: {len(final_df)}")
                    
                    # Save
                    final_df.to_parquet(output_path)
                else:
                    print(f"⚠️ No new data for {tf} (Already up-to-date or failed)")

    finally:
        driver.shutdown()
        print("🏁 Update Complete.")

if __name__ == "__main__":
    asyncio.run(update_data())
