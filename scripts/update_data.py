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

# Nautilus Imports
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.identifiers import InstrumentId, Venue, Symbol
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

async def update_data():
    print("🚀 Starting Data Update (via MT5) -> Nautilus Catalog...")
    
    driver = MT5Driver()
    if not await driver.initialize():
        print("❌ Failed to initialize MT5 Driver.")
        return

    # Configuration
    symbols = ["XAUUSD", "BTCUSD"] 
    timeframes = ["M15", "M5", "M1", "H1", "H4", "D1"] 
    
    # Destination (Catalog Root)
    catalog_path = ROOT_DIR / "data" / "nautilus_catalog"
    catalog_path.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(catalog_path)
    
    # Safety Check
    if getattr(driver, 'is_mock', False):
         print("⚠️  driver is in MOCK MODE. Skipping save to protect real data.")
         driver.shutdown()
         return

    try:
        for base_symbol in symbols:
            print(f"\n🪙 Processing {base_symbol}...")
            
            # 1. Symbol Discovery
            found_symbol = None
            suffixes_to_try = ['', 'm', 'c', 'pro', '.a', '.s']
            
            for suffix in suffixes_to_try:
                probe_sym = f"{base_symbol}{suffix}"
                probe = await driver.fetch_history(symbol=probe_sym, timeframe="M1", count=1)
                if probe is not None and not probe.empty:
                    found_symbol = probe_sym
                    print(f"   ✅ Detected Broker Symbol: {found_symbol}")
                    break
            
            if not found_symbol:
                print(f"   ❌ Could not find valid symbol for {base_symbol}")
                continue

            # 2. Iterate Timeframes
            for tf in timeframes:
                 # Map string to Nautilus Aggregation
                 tf_str = tf.upper()
                 if tf_str.startswith('M') and not tf_str.startswith('MN'):
                     count = int(tf_str[1:])
                     agg = BarAggregation.MINUTE
                 elif tf_str.startswith('H'):
                     count = int(tf_str[1:])
                     agg = BarAggregation.HOUR
                 elif tf_str.startswith('D'):
                     count = 1
                     agg = BarAggregation.DAY
                 else:
                     print(f"Skipping unknown TF: {tf}")
                     continue
                 
                 # Create BarType
                 # Note: Venue is SIM, Instrument is XAUUSD (base)
                 instrument_id = InstrumentId(Symbol(base_symbol), Venue("SIM"))
                 bar_spec = BarSpecification(count, agg, PriceType.LAST)
                 bar_type = BarType(instrument_id, bar_spec)
                 
                 print(f"  🔄 Fetching {found_symbol} {tf} (Latests)...", end=" ", flush=True)
                 
                 # Fetch 5000 bars
                 df_new = await driver.fetch_history(symbol=found_symbol, timeframe=tf, count=5000)
                 
                 if df_new is not None and not df_new.empty:
                     bars = []
                     for row in df_new.itertuples():
                         ts = int(pd.Timestamp(row.time).value) # ns
                         
                         vol = getattr(row, 'tick_volume', 0)
                         if hasattr(row, 'real_volume') and row.real_volume > 0:
                             vol = row.real_volume
                             
                         bar = Bar(
                            bar_type=bar_type,
                            open=Price.from_str(str(row.open)),
                            high=Price.from_str(str(row.high)),
                            low=Price.from_str(str(row.low)),
                            close=Price.from_str(str(row.close)),
                            volume=Quantity.from_str(str(vol)),
                            ts_event=ts,
                            ts_init=ts
                         )
                         bars.append(bar)
                     
                     catalog.write_data(bars)
                     print(f"✅ Extracted {len(bars)} bars -> Catalog.")
                 else:
                     print("⚠️ No data.")

    finally:
        driver.shutdown()
        print("🏁 Update Complete.")

if __name__ == "__main__":
    asyncio.run(update_data())
