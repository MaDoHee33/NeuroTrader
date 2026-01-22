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

    # --- 1. Price Action Fundamentals (The Core) ---
    
    # Candle Geometry
    # Body Size: Absolute value of Open - Close
    df['body_size'] = (df['close'] - df['open']).abs()
    # Upper Wick: High - Max(Open, Close)
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    # Lower Wick: Min(Open, Close) - Low
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    # Bullish Flag: 1 if Green, -1 if Red
    df['is_bullish'] = np.where(df['close'] >= df['open'], 1.0, -1.0)
    
    # --- 2. Trend (Context) ---
    # EMA 50 & 200 (Standard Trend Filters)
    ema50 = EMAIndicator(close=df['close'], window=50)
    df['ema_50'] = ema50.ema_indicator()
    
    ema200 = EMAIndicator(close=df['close'], window=200)
    df['ema_200'] = ema200.ema_indicator()
    
    # Distance from EMAs (Normalized by Close)
    df['dist_ema_50'] = (df['close'] - df['ema_50']) / df['close']
    df['dist_ema_200'] = (df['close'] - df['ema_200']) / df['close']

    # --- 3. Momentum (Timing) ---
    # RSI (14) - Unchanged
    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi() / 100.0 # Normalize to 0-1 range

    # --- 4. Volatility (Risk Management) ---
    # ATR (14)
    if 'high' in df.columns and 'low' in df.columns:
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
        # Normalized ATR (Volatility relative to price)
        df['atr_norm'] = df['atr'] / df['close']
    else:
        df['atr'] = 0.0
        df['atr_norm'] = 0.0

    # --- 5. Market Structure (Local Extrema) ---
    # Convert series to numpy for rolling
    high_prices = df['high']
    low_prices = df['low']
    
    # Rolling Max/Min (Window 14) to find recent Highs/Lows
    df['rolling_high'] = high_prices.rolling(window=14).max()
    df['rolling_low'] = low_prices.rolling(window=14).min()
    
    # Distance to Support/Resistance (Normalized)
    df['dist_to_high'] = (df['rolling_high'] - df['close']) / df['close']
    df['dist_to_low'] = (df['close'] - df['rolling_low']) / df['close']

    # --- 6. Market State Features ---
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['log_ret'] = df['log_ret'].fillna(0) # Fill after calculation, before lags
    df['log_ret_lag_1'] = df['log_ret'].shift(1)
    df['log_ret_lag_2'] = df['log_ret'].shift(2)
    
    # --- 7. Time Features (NeuroTrader 2.0) ---
    # Ensure time column is datetime
    if 'time' in df.columns and not np.issubdtype(df['time'].dtype, np.datetime64):
        df['time'] = pd.to_datetime(df['time'])
    
    if 'time' in df.columns:
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        
        # Normalize features for Neural Net
        # Hour: 0-23 -> 0-1
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day: 0-6 -> 0-1
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    else:
        # If 'time' column is not present, fill time features with NaNs or zeros
        df['hour'] = np.nan
        df['day_of_week'] = np.nan
        df['hour_sin'] = np.nan
        df['hour_cos'] = np.nan
        df['day_sin'] = np.nan
        df['day_cos'] = np.nan

    # 8. Clean up
    # Normalize features if needed? 
    # For PPO, inputs like Price should be normalized or relative.
    # We used 'dist_' and 'norm' features which is good.
    # Raw 'close' is tricky for neural nets across long timeframes? 
    # Usually we rely on return-based or relative inputs.
    # But Env observes 'close' for execution. 
    # We keep 'close' for the Env, but maybe model should focus on derived features.

    df = df.dropna()
    
    return df
