import pandas as pd
import numpy as np
import os
import glob
from src.skills.historical_sentiment import HistoricalSentiment
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_sentiment_to_datasets(data_dir="data/processed", suffix="_with_sentiment"):
    """
    Iterates over all parquet files in data_dir, merges sentiment data,
    and saves as new files with suffix.
    """
    
    # 1. Fetch Sentiment History
    logger.info("Fetching Global Market Sentiment History...")
    hs = HistoricalSentiment()
    # Fetch from 2018 to cover enough history
    sentiment_df = hs.get_combined_sentiment(start_date="2018-01-01")
    
    if sentiment_df.empty:
        logger.error("Failed to fetch sentiment data. Aborting.")
        return

    logger.info(f"Sentiment Data Range: {sentiment_df.index.min()} to {sentiment_df.index.max()}")
    
    # 2. Process Files
    files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    for file_path in files:
        # Skip already processed files
        if suffix in file_path:
            continue
            
        logger.info(f"Processing {file_path}...")
        
        try:
            df = pd.read_parquet(file_path)
            
            # Check for Timestamp/Date column
            # Usually 'time' or 'date' or index
            date_col = None
            if 'time' in df.columns:
                date_col = 'time'
            elif 'date' in df.columns:
                date_col = 'date'
            
            if date_col:
                # Ensure datetime
                df[date_col] = pd.to_datetime(df[date_col])
                # Create a temporary date column for merging
                df['_merge_date'] = df[date_col].dt.date
            else:
                # Check index
                if isinstance(df.index, pd.DatetimeIndex):
                    df['_merge_date'] = df.index.date
                else:
                    logger.warning(f"Skipping {file_path}: No datetime column or index found.")
                    continue
            
            # Merge
            # Reset index of sentiment_df to make 'date' a column
            sentiment_reset = sentiment_df.reset_index()
            # Rename 'date' in sentiment to '_merge_date' to match
            sentiment_reset = sentiment_reset.rename(columns={'date': '_merge_date'})
            
            # Merge left (keep all candle rows)
            merged_df = pd.merge(df, sentiment_reset, on='_merge_date', how='left')
            
            # Forward Fill: Sentiment is daily, so all M5 candles in same day get same value
            # Actually merge does this automatically if we merge on date
            # But what if there are missing days?
            # We already ffilled sentiment_df, so it should be fine.
            
            # Drop helper column
            merged_df = merged_df.drop(columns=['_merge_date'])
            
            # Check for NaNs (e.g. data before sentiment start date)
            # We can ffill/bfill or drop
            before_len = len(merged_df)
            merged_df = merged_df.dropna()
            after_len = len(merged_df)
            
            if before_len != after_len:
                logger.warning(f"Dropped {before_len - after_len} rows due to missing sentiment data.")
            
            # Save
            new_path = file_path.replace(".parquet", f"{suffix}.parquet")
            merged_df.to_parquet(new_path)
            logger.info(f"Saved enriched data to {new_path}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    add_sentiment_to_datasets()
