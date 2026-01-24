import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from scipy.spatial.distance import mahalanobis
import scipy.linalg as la

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

    # --- 8. Short-Term Trading Features (Added for V2.1) ---
    
    # Fast EMAs (Trend Context on lower TF)
    ema9 = EMAIndicator(close=df['close'], window=9)
    df['ema_9'] = ema9.ema_indicator()
    
    ema21 = EMAIndicator(close=df['close'], window=21)
    df['ema_21'] = ema21.ema_indicator()
    
    # Distance to Fast EMAs
    df['dist_ema_9'] = (df['close'] - df['ema_9']) / df['close']
    df['dist_ema_21'] = (df['close'] - df['ema_21']) / df['close']

    # MACD (Momentum)
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff() # Histogram
    # Normalize MACD by price to make it scale-invariant approximately
    df['macd_norm'] = df['macd'] / df['close']
    df['macd_signal_norm'] = df['macd_signal'] / df['close']
    df['macd_diff_norm'] = df['macd_diff'] / df['close']

    # Bollinger Bands (Volatility)
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']
    df['dist_bb_high'] = (df['bb_high'] - df['close']) / df['close']
    df['dist_bb_low'] = (df['close'] - df['bb_low']) / df['close']
    
    # Stochastic Oscillator (Overbought/Oversold)
    # Using default windows: 14, 3
    if 'high' in df.columns and 'low' in df.columns:
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch() / 100.0 # Normalize 0-1
        df['stoch_d'] = stoch.stoch_signal() / 100.0 # Normalize 0-1
    else:
        df['stoch_k'] = 0.5
        df['stoch_d'] = 0.5

    # --- 8.5 Exit Signal Features (Phase 1 - Scalper Improvement) ---
    # Features to help model learn when to EXIT positions
    
    # RSI Extreme Levels (Overbought/Oversold signals)
    df['rsi_overbought'] = (df['rsi'] > 0.70).astype(float)  # RSI > 70%
    df['rsi_oversold'] = (df['rsi'] < 0.30).astype(float)    # RSI < 30%
    df['rsi_extreme'] = df['rsi_overbought'] + df['rsi_oversold']  # Any extreme
    
    # MACD Crossover Signals (Momentum shift)
    df['macd_cross_up'] = ((df['macd_diff'] > 0) & (df['macd_diff'].shift(1) <= 0)).astype(float)
    df['macd_cross_down'] = ((df['macd_diff'] < 0) & (df['macd_diff'].shift(1) >= 0)).astype(float)
    df['macd_weakening'] = (df['macd_diff'].abs() < df['macd_diff'].shift(1).abs()).astype(float)
    
    # Price Extension (How far price moved from recent mean)
    df['price_extension'] = (df['close'] - df['close'].rolling(20).mean()) / df['atr']
    df['price_extension'] = df['price_extension'].fillna(0)
    df['price_overextended'] = (df['price_extension'].abs() > 2.0).astype(float)  # > 2 ATR from mean
    
    # Bollinger Band Position (0 = at lower band, 1 = at upper band)
    bb_range = df['bb_high'] - df['bb_low']
    df['bb_position'] = (df['close'] - df['bb_low']) / bb_range.replace(0, np.nan)
    df['bb_position'] = df['bb_position'].fillna(0.5)
    df['at_bb_upper'] = (df['bb_position'] > 0.95).astype(float)
    df['at_bb_lower'] = (df['bb_position'] < 0.05).astype(float)
    
    # Stochastic Extreme
    df['stoch_overbought'] = (df['stoch_k'] > 0.80).astype(float)
    df['stoch_oversold'] = (df['stoch_k'] < 0.20).astype(float)
    
    # Combined Exit Signal Score (higher = more reason to exit)
    df['exit_signal_score'] = (
        df['rsi_extreme'] +
        df['macd_weakening'] +
        df['price_overextended'] +
        df['at_bb_upper'] + df['at_bb_lower'] +
        df['stoch_overbought'] + df['stoch_oversold']
    ) / 7.0  # Normalize to 0-1

    df = df.dropna()
    

    # --- 9. Turbulence Index (Risk Management) ---
    # Based on FinRL implementation using Mahalanobis Distance
    # We use a rolling window to estimate the "normal" covariance
    
    # Features to calculate turbulence on (Key market descriptors)
    turb_features = ['log_ret', 'rsi', 'atr_norm'] 
    
    # Ensure they exist and have no NaNs for the calculation
    if all(feat in df.columns for feat in turb_features):
        try:
            # Need a history window to establish baseline covariance
            window = 50 
            if len(df) > window:
                # Calculate turbulence for the most recent points
                # Use a simplified approach: Compare current point to recent history distribution
                
                # Get history for covariance
                hist_data = df[turb_features].iloc[-window:].values
                
                try:
                    # Calculate covariance and mean of the window
                    cov_matrix = np.cov(hist_data, rowvar=False)
                    mean_vector = np.mean(hist_data, axis=0)
                    
                    # Inverse covariance matrix
                    # Add small noise to diagonal for stability
                    cov_inv = la.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)
                    
                    # Calculate Mahalanobis distance for each point (vectorized is hard for rolling)
                    # We compute it for the entire series roughly, or just the last point?
                    # For performance in live trading, we compute for the *current* row based on *past* window
                    
                    # Let's compute just for the last row to add the column (rest 0 for backfill speed)
                    # Ideally this should be a rolling apply, but that's slow.
                    # We will optimize for live usage: 
                    # "turbulence" column will be 0 except for the last calculated point if we just did it strictly
                    # usage pattern: add_features calls on growing history.
                    
                    current_vec = df[turb_features].iloc[-1].values
                    distance = mahalanobis(current_vec, mean_vector, cov_inv)
                    
                    # Store turbulence value
                    df['turbulence'] = 0.0 # Initialize 
                    df.iloc[-1, df.columns.get_loc('turbulence')] = distance
                    
                except Exception as e:
                     # Fallback if matrix singular
                     df['turbulence'] = 0.0
            else:
                 df['turbulence'] = 0.0
        except Exception as e:
            # print(f"Turbulence calc error: {e}")
            df['turbulence'] = 0.0
    else:
        df['turbulence'] = 0.0

    df = df.dropna()
    
    return df
