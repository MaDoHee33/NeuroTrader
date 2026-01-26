import asyncio
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import MetaTrader5 as mt5

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

try:
    from src.body.mt5_driver import MT5Driver
    from src.brain.feature_eng import add_features
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

async def main():
    print("🚀 Starting Multi-Asset Big Data Fetch...")
    
    driver = MT5Driver()
    if not await driver.initialize():
        print("❌ Failed to initialize MT5 Driver.")
        return

    # Target Assets (Base names)
    targets = ["XAUUSD", "BTCUSD", "DXY"] 
    # Attempt variants like "m", "c", "pro" etc.
    
    # Processed Dir
    processed_dir = ROOT_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Timeframes to fetch
    timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]
    
    # Max Count logic (approximate max for MT5 is usually limited by terminal settings)
    # We will try a safe number.
    MAX_COUNT = 50_000 

    try:
        for base in targets:
            print(f"\n🪙 Finding real symbol for {base}...")
            found_symbol = None
            
            # Common suffixes
            suffixes = ['m', '', 'c', 'pro', '.a', '.s']
            
            for s in suffixes:
                candidate = f"{base}{s}"
                if mt5.symbol_select(candidate, True):
                    # Check if data exists
                    ticks = mt5.copy_rates_from_pos(candidate, mt5.TIMEFRAME_M1, 0, 1)
                    if ticks is not None and len(ticks) > 0:
                        found_symbol = candidate
                        print(f"   ✅ Found active symbol: {found_symbol}")
                        break
            
            if not found_symbol:
                print(f"   ⚠️  Could not find valid symbol for {base}. Skipping.")
                continue
                
            # Fetch All Timeframes
            for tf_str in timeframes:
                print(f"   🔄 Fetching {found_symbol} [{tf_str}]...", end=" ")
                
                # Fetch
                df = await driver.fetch_history(symbol=found_symbol, timeframe=tf_str, count=MAX_COUNT)
                
                if df is not None and not df.empty:
                    print(f"✅ {len(df):,} rows")
                    
                    # Feature Engineering (Hybrid Price Action)
                    try:
                        # Normalize cols
                        if 'tick_volume' in df.columns and 'volume' not in df.columns:
                            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
                        
                        # Add Features
                        df_eng = add_features(df)
                        
                        # Save
                        filename = f"{base}_{tf_str}_processed.parquet" # Use base name for consistency
                        save_path = processed_dir / filename
                        df_eng.to_parquet(save_path)
                        # print(f"      Saved to {filename}")
                        
                    except Exception as e:
                        print(f"      ❌ Feature Eng Error: {e}")
                else:
                    print("❌ No Data")

    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
    finally:
        driver.shutdown()
        print("\n✅ Multi-Asset Fetch Complete.")

if __name__ == "__main__":
    asyncio.run(main())
