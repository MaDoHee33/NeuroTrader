
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SentimentFetcher:
    """
    Fetches market sentiment data from various sources:
    1. Fear & Greed Index (Alternative.me)
    2. VIX (Volatility Index) via Yahoo Finance
    3. US 10Y Treasury Yield via Yahoo Finance
    4. DXY (Dollar Index) via Yahoo Finance
    """
    
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        
    def get_fear_greed(self) -> dict:
        """
        Fetch Fear & Greed Index from alternative.me
        Returns: {'value': int, 'classification': str} or None
        """
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            response = requests.get(url, headers=self.headers, timeout=10)
            data = response.json()
            
            if data['metadata']['error']:
                logger.error(f"F&G API Error: {data['metadata']['error']}")
                return None
                
            item = data['data'][0]
            return {
                'value': int(item['value']),
                'classification': item['value_classification'],
                'timestamp': int(item['timestamp'])
            }
        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed: {e}")
            return None

    def get_market_data(self) -> dict:
        """
        Fetch VIX, US10Y, DXY from Yahoo Finance
        """
        tickers = {
            'VIX': '^VIX',   # CBOE Volatility Index
            'US10Y': '^TNX', # 10-Year Treasury Yield
            'DXY': 'DX-Y.NYB' # US Dollar Index
        }
        
        result = {}
        
        try:
            data = yf.download(list(tickers.values()), period="1d", progress=False)
            
            # Helper to safely extract last close
            def get_last_close(ticker_symbol):
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        # YFinance new structure
                        val = data['Close'][ticker_symbol].iloc[-1]
                    else:
                        val = data['Close'].iloc[-1]
                    return float(val)
                except:
                    return None

            for name, symbol in tickers.items():
                val = get_last_close(symbol)
                if val is not None:
                    result[name] = val
                    
        except Exception as e:
            logger.error(f"Failed to fetch Market Data (YF): {e}")
            
        return result

    def get_sentiment_summary(self) -> dict:
        """
        Get aggregated sentiment summary.
        """
        fg = self.get_fear_greed()
        market = self.get_market_data()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'fear_greed': fg['value'] if fg else 50,
            'fear_greed_label': fg['classification'] if fg else 'Neutral',
            'vix': market.get('VIX', 0.0),
            'us10y': market.get('US10Y', 0.0),
            'dxy': market.get('DXY', 0.0)
        }
        
        # Simple Interpretation
        # VIX > 30 = High Fear
        # VIX < 15 = Complacency
        if summary['vix'] > 30:
            summary['market_mood'] = 'Extreme Fear'
        elif summary['vix'] > 20:
            summary['market_mood'] = 'Fear'
        elif summary['vix'] < 15:
            summary['market_mood'] = 'Risk On'
        else:
            summary['market_mood'] = 'Neutral'
            
        return summary

if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    fetcher = SentimentFetcher()
    print("Fetching Sentiment Data...")
    summary = fetcher.get_sentiment_summary()
    print("\n--------- SENTIMENT REPORT ---------")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("------------------------------------")
