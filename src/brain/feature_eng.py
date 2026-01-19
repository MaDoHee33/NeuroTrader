import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

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
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close'] # Normalized width

    # 4. EMA (20, 50)
    ema20 = EMAIndicator(close=df['close'], window=20)
    df['ema_20'] = ema20.ema_indicator()
    
    ema50 = EMAIndicator(close=df['close'], window=50)
    df['ema_50'] = ema50.ema_indicator()

    # 5. ATR (14) - Volatility
    if 'high' in df.columns and 'low' in df.columns:
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
    else:
        df['atr'] = 0.0 # Placeholder if high/low missing
        
    # 6. Stochastic Oscillator (14, 3, 3) - Momentum
    if 'high' in df.columns and 'low' in df.columns:
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
    else:
        df['stoch_k'] = 50.0
        df['stoch_d'] = 50.0

    # 7. VWAP (Volume Weighted Average Price) - Intraday Trend
    if 'volume' in df.columns and 'high' in df.columns and 'low' in df.columns:
        # Note: accurate VWAP resets daily, but rolling VWAP is useful for short-term trend
        # ta library VolumeWeightedAveragePrice is rolling window or cumulative?
        # Usually checking docs: it's cumulative. For rolling, we might need custom.
        # But for RL, even a rolling approximation is fine.
        vwap = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14)
        df['vwap'] = vwap.volume_weighted_average_price()
    else:
        df['vwap'] = df['close'] # Fallback

    # 8. Log Returns & Lags (Market State)
    # r_t = ln(P_t / P_{t-1})
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Fill initial NaNs from shift with 0
    df['log_ret'] = df['log_ret'].fillna(0)
    
    # Lags
    for lag in [1, 2, 3, 5]:
        df[f'log_ret_lag_{lag}'] = df['log_ret'].shift(lag)

    # 9. Cleanup
    # Indicators introduce NaNs at the beginning. 
    # Backfill to preserve data length (though RL env handles skipping first N steps usually)
    df = df.bfill().fillna(0)
    
    return df 
    # Dropping is safest for training logic to avoid noisy 0s.
    df = df.dropna()
    
    return df
