import requests
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class HistoricalSentiment:
    """
    Fetches historical sentiment data for backtesting/training:
    1. Fear & Greed Index (Complete History)
    2. VIX (Volatility Index)
    3. US 10Y Treasury Yield
    4. DXY (Dollar Index)
    """
    
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        
    def fetch_fear_greed_history(self, limit=0) -> pd.DataFrame:
        """
        Fetch historical Fear & Greed Index from alternative.me.
        limit=0 for all available history.
        """
        try:
            url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
            response = requests.get(url, headers=self.headers, timeout=10)
            data = response.json()
            
            if data['metadata']['error']:
                logger.error(f"F&G API Error: {data['metadata']['error']}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            df['value'] = df['value'].astype(int)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            df['date'] = df['timestamp'].dt.date
            
            # Set index to date for easy merging
            df = df.set_index('date').sort_index()
            
            # Keep only value and classification
            df = df[['value', 'value_classification']]
            df.columns = ['fear_greed_value', 'fear_greed_label']
            
            logger.info(f"Fetched {len(df)} days of Fear & Greed history.")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed History: {e}")
            return pd.DataFrame()

    def fetch_market_history(self, start_date="2010-01-01") -> pd.DataFrame:
        """
        Fetch historical VIX, US10Y, DXY from Yahoo Finance.
        """
        tickers = {
            'VIX': '^VIX',   # CBOE Volatility Index
            'US10Y': '^TNX', # 10-Year Treasury Yield
            'DXY': 'DX-Y.NYB' # US Dollar Index
        }
        
        try:
            # Download all at once
            data = yf.download(list(tickers.values()), start=start_date, progress=False)
            
            # Extract Close prices
            if isinstance(data.columns, pd.MultiIndex):
                # Dropping the 'Close' level or accessing it directly
                df = data['Close'].copy()
            else:
                # If only one ticker was downloaded (unlikely here), structure is different
                df = data['Close'].copy() if 'Close' in data else data
            
            # Rename columns to friendly names
            # Map symbol to name
            inv_map = {v: k for k, v in tickers.items()}
            df = df.rename(columns=inv_map)
            df.columns = [c.lower() for c in df.columns]
            
            # Convert index to date (it's usually DatetimeIndex with time 00:00)
            df.index = df.index.date
            df.index.name = 'date'
            
            logger.info(f"Fetched {len(df)} days of Market Data history.")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch Market Data History (YF): {e}")
            return pd.DataFrame()

    def get_combined_sentiment(self, start_date="2020-01-01") -> pd.DataFrame:
        """
        Returns a combined DataFrame with all sentiment features, indexed by Date.
        Forward filled to handle weekends/holidays (sentiment persists).
        """
        fg_df = self.fetch_fear_greed_history()
        mkt_df = self.fetch_market_history(start_date=start_date)
        
        # Merge on Date (Outer join to keep all days)
        # Fear & Greed is 7 days/week (crypto based sort of, but sentiment is daily)
        # Market data is 5 days/week.
        
        combined = pd.concat([fg_df, mkt_df], axis=1)
        
        # Sort
        combined = combined.sort_index()
        
        # Filter by start date
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        combined = combined[combined.index >= start_date_obj]
        
        # Forward Fill (If market closed on weekend, VIX stays same)
        combined = combined.ffill()
        
        return combined

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hs = HistoricalSentiment()
    df = hs.get_combined_sentiment(start_date="2023-01-01")
    print("\nSample Data (Last 5 days):")
    print(df.tail())
