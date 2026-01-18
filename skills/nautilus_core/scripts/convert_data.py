
import argparse
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from decimal import Decimal

# Nautilus Imports
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

# Local Import (Need to add to path)
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))
from skills.nautilus_core.custom_data import SentimentData

def convert_to_nautilus(processed_file, catalog_path):
    print(f"🚀 Converting {processed_file} to Nautilus Parquet...")
    
    # Setup Catalog
    catalog_path = Path(catalog_path)
    if catalog_path.exists():
        shutil.rmtree(catalog_path)
    catalog_path.mkdir(parents=True)
    
    catalog = ParquetDataCatalog(str(catalog_path))
    
    # Load DF
    df = pd.read_parquet(processed_file)
    df = df.reset_index(drop=True)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    # Define Instrument
    instrument_id = InstrumentId.from_str("XAUUSD.SIM")
    
    # 1. Convert BARS (OHLCV)
    # -----------------------
    print("📊 Converting Price Bars...")
    # Use from_str for version compatibility
    # Format: INSTRUMENT-AGGREGATION-TYPE-MARKET
    bar_type = BarType.from_str(f"{instrument_id}-15-MINUTE-LAST-EXTERNAL")
    
    bars = []
    for _, row in df.iterrows():
        # Convert Timestamp to ns
        ts = int(row['time'].timestamp() * 1e9)
        
        # Create Bar
        # Note: Prices must be capable of being Decimal, Nautilus handles float to decimal conversion usually, 
        # but passing strings or Decimals is safer for precision.
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(str(row['open'])),
            high=Price.from_str(str(row['high'])),
            low=Price.from_str(str(row['low'])),
            close=Price.from_str(str(row['close'])),
            volume=Quantity.from_str(str(row.get('tick_volume', 0))),
            ts_event=ts,
            ts_init=ts,
        )
        bars.append(bar)
        
    # Write Bars
    print(f"💾 Writing {len(bars)} Bars to Catalog...")
    catalog.write_data(bars)
    
    # 2. Convert SENTIMENT (Custom)
    # -----------------------------
    # FIXME: CustomData init issues. Postponed to Phase 2.
    # if 'news_impact_score' in df.columns:
    #     print("🧠 Converting Sentiment Data...")
    #     sentiment_data = []
        
    #     # Filter only when score > 0 to save space? Or keep all aligned?
    #     # Let's keep all for alignment
    #     for _, row in df.iterrows():
    #         ts = int(row['time'].timestamp() * 1e9)
    #         score = float(row.get('news_impact_score', 0.0))
            
    #         # Create Custom Data
    #         # Start timestamp is handled by ts_event
    #         # data = SentimentData(
    #         #     instrument_id=instrument_id,
    #         #     ts_event=ts,
    #         #     score=score,
    #         #     source="Historical_Augmented"
    #         # )
    #         # sentiment_data.append(data)
            
    #     # Write Sentiment
    #     # print(f"💾 Writing {len(sentiment_data)} Sentiment points to Catalog...")
    #     # catalog.write_data(sentiment_data)
        
    print(f"✅ Conversion Complete! Catalog at: {catalog_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/XAUUSDm_M15_L2_News.parquet")
    parser.add_argument("--output", type=str, default="data/nautilus_catalog")
    args = parser.parse_args()
    
    convert_to_nautilus(args.input, args.output)
