import asyncio
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import subprocess

# Add src to path
# Path: skills/neuro_trader/scripts/update_data.py -> Root is ../../../
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.body.mt5_driver import MT5Driver
from src.utils.logger import get_logger

class DataPipeline:
    def __init__(self):
        self.logger = get_logger("DataPipeline")
        self.raw_data_dir = ROOT_DIR / "data" / "raw"
        self.processed_data_dir = ROOT_DIR / "data" / "processed"
        self.driver = MT5Driver()
        
        # Config (Sync with fetch_data.py for now)
        self.symbols = ["BTCUSDm", "XAUUSDm"]
        self.timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]
        self.min_start_date = datetime(2010, 1, 1)

    async def initialize(self):
        return await self.driver.initialize()

    def get_last_timestamp(self, file_path):
        """Reads the last timestamp from a CSV file."""
        if not os.path.exists(file_path):
            return None
            
        try:
            # optimize: read only last few lines or use pandas with chunksize if file is huge
            # for now, simple read since files aren't gigabytes yet
            df = pd.read_csv(file_path)
            if df.empty or 'time' not in df.columns:
                return None
            
            # Ensure time is datetime
            last_time_str = df['time'].iloc[-1]
            return pd.to_datetime(last_time_str)
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return None

    async def update_data(self):
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting Data Pipeline Check...")
        
        for symbol in self.symbols:
            for tf in self.timeframes:
                file_name = f"{symbol}_{tf}.csv"
                file_path = self.raw_data_dir / file_name
                
                # 1. Check Last Timestamp
                last_date = self.get_last_timestamp(file_path)
                
                now = datetime.now()
                fetch_from = self.min_start_date
                
                if last_date:
                    # Fetch from last_date. 
                    # Note: Existing last row might be incomplete or we want overlap to be safe.
                    # We'll fetch from last_date and deduplicate later.
                    fetch_from = last_date
                    print(f"   🔄 {symbol} {tf}: Found existing data. Updating from {fetch_from}...")
                else:
                    print(f"   🆕 {symbol} {tf}: No data. Fetching from scratch ({fetch_from})...")

                # 2. Fetch New Data
                # If fetch_from is excessively close to now (e.g. < 1 min), skip?
                if (now - fetch_from).total_seconds() < 60:
                     print(f"   ✅ {symbol} {tf}: Up to date.")
                     continue

                new_df = await self.driver.fetch_history_range(symbol, tf, fetch_from, now)
                
                if new_df is not None and not new_df.empty:
                    # 3. Merge Phase
                    if last_date and os.path.exists(file_path):
                        # Append mode logic
                        existing_df = pd.read_csv(file_path)
                        existing_df['time'] = pd.to_datetime(existing_df['time'])
                        
                        # Concatenate
                        combined_df = pd.concat([existing_df, new_df])
                        
                        # Drop Duplicates based on time
                        # Keep='last' effectively updates the candle if it changed
                        combined_df.drop_duplicates(subset=['time'], keep='last', inplace=True)
                        combined_df.sort_values('time', inplace=True)
                        
                        combined_df.to_csv(file_path, index=False)
                        added_rows = len(combined_df) - len(existing_df)
                        print(f"   📥 {symbol} {tf}: Updated. Added {added_rows} new rows.")
                    else:
                        # New file
                        new_df.to_csv(file_path, index=False)
                        print(f"   💾 {symbol} {tf}: Created new file ({len(new_df)} rows).")
                else:
                    # No new data found or error
                    pass  # warning handled in driver

    def run_processing(self):
        print("\n⚙️  Triggering Feature Engineering...")
        # Run process_data.py for Level 1 and Level 2
        # subprocess allows running it as a separate clean process
        
        tools_dir = ROOT_DIR / "tools"
        process_script = tools_dir / "process_data.py"
        
        if not process_script.exists():
            print("❌ process_data.py not found!")
            return

        for lvl in [1, 2]:
            print(f"   🔨 Processing Level {lvl}...")
            try:
                subprocess.run(
                    [sys.executable, str(process_script), "--level", str(lvl)],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"❌ Error processing Level {lvl}: {e}")

    async def run(self):
        # 1. Connect MT5
        if not await self.initialize():
            print("❌ MT5 Initialization Failed.")
            return

        try:
            # 2. Update Raw Data
            await self.update_data()
            
            # 3. Process Data
            self.run_processing()
            
            print("\n✅ Data Pipeline Completed Successfully.")
            
        finally:
            self.driver.shutdown()

if __name__ == "__main__":
    pipeline = DataPipeline()
    asyncio.run(pipeline.run())
