import asyncio
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.body.mt5_driver import MT5Driver
    from src.brain.features import add_features
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Ensure you are running this from the project root or scripts folder.")
    sys.exit(1)

async def main():
    print("🚀 Starting XAUUSDm Data Fetch & Processing...")
    
    # Initialize Driver
    driver = MT5Driver()
    if not await driver.initialize():
        print("❌ Failed to initialize MT5 Driver.")
        return

    try:
        # 1. Identify Symbol
        target_symbol = "XAUUSDm"
        found_symbol = None
        
        # Check XAUUSDm
        print(f"🔎 Checking for {target_symbol}...")
        probe = await driver.fetch_history(symbol=target_symbol, timeframe="M1", count=1)
        if probe is not None and not probe.empty:
            print(f"✅ Found {target_symbol}.")
            found_symbol = target_symbol
        else:
            print(f"⚠️  {target_symbol} not found. Checking 'XAUUSD'...")
            probe = await driver.fetch_history(symbol="XAUUSD", timeframe="M1", count=1)
            if probe is not None and not probe.empty:
                print("✅ Found 'XAUUSD'. Using it as fallback.")
                found_symbol = "XAUUSD"
            else:
                print("❌ No XAUUSD variant found. Please ensure Market Watch has the symbol.")
                return

        # 2. Prepare Directories
        raw_dir = ROOT_DIR / "data" / "raw"
        processed_dir = ROOT_DIR / "data" / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Output Directories:\n   - {raw_dir}\n   - {processed_dir}")

        # 3. Fetch & Process
        # Ensure symbol is selected
        import MetaTrader5 as mt5
        if not mt5.symbol_select(found_symbol, True):
            print(f"⚠️  Failed to select {found_symbol} in Market Watch.")
            
        # We fetch multiple timeframes using count to avoid range issues
        timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]
        count_bars = 50000 # Reduced to 50k for safety
        
        for tf in timeframes:
            print(f"\n🔄 Processing {found_symbol} [{tf}]...")
            
            # Fetch
            df = await driver.fetch_history(
                symbol=found_symbol, 
                timeframe=tf, 
                count=count_bars
            )
            
            if df is not None and not df.empty:
                rows = len(df)
                print(f"   📥 Fetched {rows} rows (from {df['time'].iloc[0]} to {df['time'].iloc[-1]})")
                
                # Save Raw
                raw_path = raw_dir / f"{found_symbol}_{tf}_raw.parquet"
                df.to_parquet(raw_path)
                print(f"   💾 Saved RAW: {raw_path.name}")
                
                # Feature Engineering
                print("   ⚙️  Applying Feature Engineering...")
                try:
                    # Normalize columns for feature_eng (it expects 'close', 'high', 'low', 'volume')
                    # MT5Driver returns lower case: 'time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'
                    # We usually map 'tick_volume' -> 'volume'
                    df_eng = df.copy()
                    if 'tick_volume' in df_eng.columns and 'volume' not in df_eng.columns:
                        df_eng.rename(columns={'tick_volume': 'volume'}, inplace=True)
                    
                    # Rename time to timestamp if needed or set as index? 
                    # feature_eng doesn't explicitly require index but 'add_features' returns df.
                    # existing 'verify_features.py' set index. let's keep it simple.
                    
                    df_processed = add_features(df_eng)
                    
                    # Save Processed
                    proc_path = processed_dir / f"{found_symbol}_{tf}_processed.parquet"
                    df_processed.to_parquet(proc_path)
                    print(f"   💾 Saved PROCESSED: {proc_path.name}")
                    print(f"      Useable features: {len(df_processed.columns)}")
                    
                except Exception as e:
                    print(f"   ❌ Feature Engineering Error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"   ⚠️  No data returned for {tf}")

    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
    finally:
        driver.shutdown()
        print("\n✅ Task Finished.")

if __name__ == "__main__":
    asyncio.run(main())
