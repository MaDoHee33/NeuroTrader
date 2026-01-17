
import pandas as pd
import ta
import numpy as np
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"

def process_file(file_path):
    print(f"🔄 Processing: {file_path.name}")
    
    try:
        # 1. Load CSV
        df = pd.read_csv(file_path, parse_dates=['time'])
        
        # 2. Basic Cleaning
        df = df.drop_duplicates(subset=['time'])
        df = df.sort_values('time')
        df = df.fillna(method='ffill')
        
        # 3. Feature Engineering (using 'ta' library)
        # Ensure we have enough data
        if len(df) < 200:
             print(f"⚠️  Skipping {file_path.name}: Not enough data ({len(df)} rows)")
             return

        # Trend
        df['ema_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(close=df['close'], window=200).ema_indicator()
        
        # Momentum
        rsi = ta.momentum.RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()
        
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Volatility
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()
        
        atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
        
        # Drop NaN values created by indicators
        df = df.dropna()
        
        # 4. Storage Optimization
        # Create output filename (change .csv to .parquet)
        out_name = file_path.stem + ".parquet"
        out_path = DATA_PROCESSED / out_name
        
        # Save as Parquet (Snappy compression is default and good balance)
        df.to_parquet(out_path, index=False)
        
        print(f"✅ Saved: {out_name} | Shape: {df.shape}")
        
    except Exception as e:
        print(f"❌ Error processing {file_path.name}: {e}")

def main():
    # Ensure processed directory exists
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Scan for CSV files
    files = list(DATA_RAW.glob("*.csv"))
    
    if not files:
        print("⚠️  No CSV files found in data/raw/")
        return
        
    print(f"🚀 Starting Data Pipeline for {len(files)} files...")
    
    for file in files:
        process_file(file)
        
    print("\n✨ Data Processing Complete.")

if __name__ == "__main__":
    main()
