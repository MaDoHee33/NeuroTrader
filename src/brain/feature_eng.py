
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the DataFrame.
    Expected input columns: 'close', 'high', 'low', 'volume' (optional)
    """
    df = df.copy()
    
    # Ensure required columns exist
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    # 1. RSI (14)
    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi()

    # 2. MACD (12, 26, 9)
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # 3. Bollinger Bands (20, 2)
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()

    # 4. EMA (20, 50)
    ema20 = EMAIndicator(close=df['close'], window=20)
    df['ema_20'] = ema20.ema_indicator()
    
    ema50 = EMAIndicator(close=df['close'], window=50)
    df['ema_50'] = ema50.ema_indicator()

    # 5. ATR (14)
    if 'high' in df.columns and 'low' in df.columns:
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
    else:
        df['atr'] = 0.0 # Placeholder if high/low missing

    # 6. Log Returns & Lags
    # r_t = ln(P_t / P_{t-1})
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Fill initial NaNs from shift with 0
    df['log_ret'] = df['log_ret'].fillna(0)
    
    # Lags
    for lag in [1, 2, 3, 5]:
        df[f'log_ret_lag_{lag}'] = df['log_ret'].shift(lag)

    # 7. Cleanup
    # Indicators introduce NaNs at the beginning. 
    # Valid options: Drop, Fill with 0, or Backfill. 
    # Dropping is safest for training logic to avoid noisy 0s.
    df = df.dropna()
    
    return df
