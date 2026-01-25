import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

from src.utils.config_loader import config

class FeatureEngine:
    """
    Centralized Feature Engineering Class.
    Handles calculation, normalization, and validation of technical indicators.
    """
    def __init__(self):
        self.history_size = config.get("environment.history_size", 60)
        self.required_cols = ['open', 'high', 'low', 'close', 'volume']
        
    def validate(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has necessary columns"""
        return all(col in df.columns for col in self.required_cols)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point. Adds indicators to the DataFrame.
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # 1. Base Price Action
        df['body_size'] = (df['close'] - df['open']).abs()
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['is_bullish'] = np.where(df['close'] >= df['open'], 1.0, -1.0)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)

        # 2. Volatility
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-6)
        
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        df['atr_norm'] = df['atr'] / df['close']

        # 3. Momentum
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi() / 100.0
        
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['stoch_k'] = stoch.stoch() / 100.0
        df['stoch_d'] = stoch.stoch_signal() / 100.0

        # 4. Trend
        df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['dist_ema_50'] = (df['close'] - df['ema_50']) / df['close']
        
        # Mock EMA 200 if not enough data
        if len(df) > 200:
             df['ema_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
        else:
             df['ema_200'] = df['close']
        df['dist_ema_200'] = (df['close'] - df['ema_200']) / df['close']

        # 5. Volume
        vwap = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
        df['vwap'] = vwap.volume_weighted_average_price()
        df['normalized_volume'] = df['volume'] / (df['volume'].rolling(50).mean() + 1e-6)

        # 6. Time Features (Safe handling)
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        else:
            df['hour_sin'] = 0.0
            df['hour_cos'] = 0.0
            df['day_sin'] = 0.0
            df['day_cos'] = 0.0

        # 7. Agent Normalization (For Observation Space)
        df['normalized_close'] = df['close'] / df['close'].iloc[0]

        # 8. Signals (Placeholders for complex logic)
        df['rsi_extreme'] = ((df['rsi'] > 0.7) | (df['rsi'] < 0.3)).astype(float)
        
        # MACD Weakening (Histogram shrinking)
        df['macd_weakening'] = (df['macd_diff'].abs() < df['macd_diff'].shift(1).abs()).astype(float)
        
        # Price Overextended (> 2 ATR from moving average)
        df['price_extension'] = (df['close'] - df['close'].rolling(20).mean()) / (df['atr'] + 1e-6)
        df['price_overextended'] = (df['price_extension'].abs() > 2.0).astype(float)

        df['exit_signal_score'] = 0.0
        df['turbulence'] = 0.0
        df['log_ret_lag_1'] = df['log_ret'].shift(1).fillna(0)
        df['log_ret_lag_2'] = df['log_ret'].shift(2).fillna(0)
        
        # Market Structure
        df['rolling_high'] = df['high'].rolling(14).max()
        df['dist_to_high'] = (df['rolling_high'] - df['close']) / df['close']
        df['rolling_low'] = df['low'].rolling(14).min()
        df['dist_to_low'] = (df['close'] - df['rolling_low']) / df['close']

        # Cleanup (Warmup)
        # Drop initial rows where indicators are NaN
        # But ensure we don't return empty if df is small
        if len(df) > 50:
            df = df.iloc[50:].copy()
        
        df = df.fillna(0)
        return df

# Singleton exposure
feature_engine = FeatureEngine()
def add_features(df):
    return feature_engine.add_features(df)
