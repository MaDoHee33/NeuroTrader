import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

def add_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified & Robust Feature Engineering using 'ta' library.
    Ensures no data loss from aggressive dropna.
    """
    df = df.copy()
    
    # 1. Price Action
    df['body_size'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['is_bullish'] = np.where(df['close'] >= df['open'], 1.0, -1.0)
    
    # 2. Indicators using 'ta'
    # RSI
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi() / 100.0
    
    # EMA
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['dist_ema_50'] = (df['close'] - df['ema_50']) / df['close']
    # EMA 200 (Mock for synthetic if not enough data, or calc)
    df['ema_200'] = df['close'] # Placeholder if data too short, else: EMAIndicator(close=df['close'], window=200).ema_indicator()
    df['dist_ema_200'] = 0.0
    
    # MACD
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
    df['bb_upper'] = bb.bollinger_hband() 
    df['bb_lower'] = bb.bollinger_lband()
    # BB Position
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-6)
    
    # ATR
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df['atr_norm'] = df['atr'] / df['close']
    
    # Stochastic
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['stoch_k'] = stoch.stoch() / 100.0
    df['stoch_d'] = stoch.stoch_signal() / 100.0
    
    # VWAP
    vwap = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
    df['vwap'] = vwap.volume_weighted_average_price()
    
    # Log Returns & Lags
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    df['log_ret_lag_1'] = df['log_ret'].shift(1).fillna(0)
    df['log_ret_lag_2'] = df['log_ret'].shift(2).fillna(0)
    
    # Market Structure
    df['rolling_high'] = df['high'].rolling(14).max()
    df['dist_to_high'] = (df['rolling_high'] - df['close']) / df['close']
    df['rolling_low'] = df['low'].rolling(14).min()
    df['dist_to_low'] = (df['close'] - df['rolling_low']) / df['close']

    # Time Features (Mocked if index is not datetime or just filling)
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['day'] = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)
    else:
        df['hour_sin'] = 0.0
        df['hour_cos'] = 0.0
        df['day_sin'] = 0.0
        df['day_cos'] = 0.0

    # Exit Signals (Placeholders/simplified)
    df['rsi_extreme'] = ((df['rsi'] > 0.7) | (df['rsi'] < 0.3)).astype(float)
    df['macd_weakening'] = 0.0
    df['price_overextended'] = 0.0
    df['exit_signal_score'] = 0.0
    df['turbulence'] = 0.0

    
    # Normalized features for Agent
    df['normalized_close'] = df['close'] / df['close'].iloc[0] # Simple norm
    df['normalized_volume'] = df['volume'] / (df['volume'].rolling(50).mean() + 1e-6)
    
    # Cleanup NaNs (Fill or Drop)
    # We drop only the warmup period (e.g., first 50 rows)
    # Instead of dropna() which might drop everything if one column is all NaN
    df = df.iloc[50:].copy() 
    df = df.fillna(0) # Safety fill for remaining NaNs
    
    return df
