import pandas as pd
from pathlib import Path
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Venue, Symbol
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

def main():
    print("🚀 Starting Data Ingestion (Full History)...")
    
    # 1. Load CSV
    csv_path = Path("data/raw/XAUUSDm_M15.csv")
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} not found.")
        return

    print(f"📄 Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['time'])
    
    # Sort and Deduplicate
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    
    print(f"📊 Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # 2. Setup Catalog
    catalog_path = Path("data/nautilus_catalog")
    catalog_path.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(catalog_path)
    
    # 3. Define Bar Type
    bar_type_str = "XAUUSD.SIM-15-MINUTE-LAST-EXTERNAL"
    bar_type = BarType.from_str(bar_type_str)
    print(f"🔹 Bar Type: {bar_type}")
    
    # 4. Convert to Nautilus Objects
    bars = []
    print("⚙️  Converting to Nautilus Bars...")
    
    # Performance optimization: iterate efficiently
    # We use ticks/volume
    for row in df.itertuples():
        ts = int(row.timestamp.value) # ns
        
        # Scale: Nautilus might require scaled int if Price(int, precision) used
        # But Price.from_str handles decimal strings 
        
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(str(row.open)),
            high=Price.from_str(str(row.high)),
            low=Price.from_str(str(row.low)),
            close=Price.from_str(str(row.close)),
            volume=Quantity.from_str(str(row.tick_volume)),
            ts_event=ts,
            ts_init=ts
        )
        bars.append(bar)

    print(f"✅ Converted {len(bars)} bars.")

    # 5. Write to Catalog
    print(f"💾 Writing to {catalog_path}...")
    catalog.write_data(bars)
    
    print("🎉 Data Ingestion Complete!")

if __name__ == "__main__":
    main()
