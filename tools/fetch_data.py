import asyncio
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.body.mt5_driver import MT5Driver
from src.utils.logger import get_logger

async def fetch_job():
    logger = get_logger("DataFetcher")
    
    # 1. Initialize Driver
    driver = MT5Driver()
    if not await driver.initialize():
        print("Failed to init driver.")
        return

    # 2. Config
    symbols = ["BTCUSDm", "XAUUSDm"]
    # All standard timeframes
    timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]
    
    # Date Range: From 2010 (or earliest) to Now
    date_from = datetime(2010, 1, 1)
    date_to = datetime.now()
    
    print(f"🚀 Starting Massive Data Extraction...")
    print(f"🎯 Targets: {symbols}")
    print(f"📅 Range: {date_from.date()} -> {date_to.date()}")
    print(f"⏱️  Timeframes: {timeframes}")

    for symbol in symbols:
        print(f"\n🪙  Processing Symbol: {symbol}")
        
        for tf in timeframes:
            print(f"   ⏳ Fetching {tf}...", end="\r")
            
            # 3. Fetch Range
            df = await driver.fetch_history_range(symbol, tf, date_from, date_to)
            
            if df is not None and not df.empty:
                # 4. Save
                out_dir = Path("data/raw")
                out_dir.mkdir(parents=True, exist_ok=True)
                
                out_file = out_dir / f"{symbol}_{tf}.csv"
                df.to_csv(out_file, index=False)
                
                # Stats
                start_date = df['time'].iloc[0]
                rows = len(df)
                print(f"   ✅ {tf}: Saved {rows:,} rows ({start_date} -> {df['time'].iloc[-1]})")
            else:
                print(f"   ⚠️  {tf}: No data. (Check if symbol exists or has history)")

    driver.shutdown()
    print("\n✨ Extraction Complete.")

if __name__ == "__main__":
    asyncio.run(fetch_job())
