import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "processed"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
CALENDAR_FILE = ASSETS_DIR / "calendar.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "augmented"

def augment_with_news(processed_file):
    print(f"🔄 Augmenting {processed_file.name} with News Data...")
    
    # 1. Load Price Data
    df_price = pd.read_parquet(processed_file)
    df_price['time'] = pd.to_datetime(df_price['time'])
    df_price = df_price.set_index('time').sort_index()
    
    # 2. Load Calendar Data
    if not CALENDAR_FILE.exists():
        print(f"❌ Calendar file not found at {CALENDAR_FILE}")
        return
        
    df_news = pd.read_csv(CALENDAR_FILE)
    
    # 3. Filter & Clean News
    # Filter for USD only (since we trade XAUUSD)
    df_news = df_news[df_news['currency'] == 'USD'].copy()
    
    # Filter for HIGH importance
    # Note: Importance might be 'High', 'Medium', 'Low' or labeled differently.
    # Let's check unique values or assume 'high' contains 'High' or 'high'
    df_news['importance'] = df_news['importance'].astype(str).str.lower()
    df_news = df_news[df_news['importance'].str.contains('high')]
    
    # Create DateTime column
    # Date format might vary. Assuming ISO or standard.
    # User file likely has separate date/time.
    df_news['start_time'] = pd.to_datetime(df_news['date'] + ' ' + df_news['time'], errors='coerce')
    df_news = df_news.dropna(subset=['start_time'])
    df_news = df_news.set_index('start_time').sort_index()
    
    # 4. Merge Logic (Feature Engineering)
    # We want to know: "Is there High Impact News in this candle?"
    # Resample news to match price timeframe (e.g., H1)
    # Count number of high impact events per hour
    
    # Create a dummy series to resample
    news_counts = df_news['event'].resample('1h').count()
    news_counts.name = 'news_impact_score'
    
    # 5. Join
    # Left join to Price data (keep all price rows)
    df_merged = df_price.join(news_counts, how='left')
    
    # Fill NaN with 0 (No news)
    df_merged['news_impact_score'] = df_merged['news_impact_score'].fillna(0)
    
    # Log transform or clip? Just binary/count is fine for now.
    # Let's clip at 5 events to normalize slightly
    df_merged['news_impact_score'] = np.clip(df_merged['news_impact_score'], 0, 5)
    
    # 6. Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"{processed_file.stem}_News.parquet"
    out_path = OUTPUT_DIR / out_name
    
    df_merged = df_merged.reset_index() # Restore time column
    df_merged.to_parquet(out_path, index=False)
    
    print(f"✅ Saved: {out_name} | Rows: {len(df_merged)}")
    print(f"📊 News Stats: {df_merged['news_impact_score'].value_counts().to_dict()}")
    
    return out_path

def main():
    parser = argparse.ArgumentParser()
    # Find default Level 2 file if not specified
    default_file = list(RAW_DATA_DIR.glob("*XAUUSD*_L2.parquet"))
    default_path = str(default_file[0]) if default_file else None
    
    parser.add_argument("--file", type=str, default=default_path, help="Path to processed parquet file")
    args = parser.parse_args()
    
    if not args.file:
        print("❌ No input file found. Please run process_data.py first.")
        return
        
    augment_with_news(Path(args.file))

if __name__ == "__main__":
    main()
