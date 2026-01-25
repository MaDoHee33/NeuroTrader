
import pandas as pd
import numpy as np
from collections import deque
from typing import List, Dict, Union
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

class UnifiedFeatureEngine:
    """
    Stateful feature engine ensuring 100% parity between Batch (Training) and Stream (Inference).
    """
    
    # EXACT definition of the Observation Space
    FEATURE_NAMES = [
        'rsi', 'macd', 'macd_signal', 'bb_width', 'bb_position',
        'ema_20_dist', 'ema_50_dist', 'atr_norm', 
        'stoch_k', 'stoch_d', 'log_ret',
        'vwap_dist', 'normalized_volume',
        'body_size_norm', 'is_bullish',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]

    def __init__(self, history_size: int = 200):
        self.history_size = history_size
        self.history = deque(maxlen=history_size)
        
    def get_output_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def compute_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for a full DataFrame (Training Mode).
        Returns DataFrame with ONLY the feature columns.
        """
        if df.empty:
            return pd.DataFrame(columns=self.FEATURE_NAMES)
            
        # Avoid crashes on very small data (Cold Start)
        if len(df) < 20: 
            # Not enough data for indicators (window=14 or 20)
            # Return zeros but with correct columns/index
            z = pd.DataFrame(0.0, index=df.index, columns=self.FEATURE_NAMES)
            # Fill basic columns that don't need history if possible?
            # For consistency, it's safer to return 0 until warmed up
            # But let's at least preserve 'log_ret' if len >= 2
            return z

        df = df.copy()
        
        # Ensure DatetimeIndex for Time Embeddings
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
        
        # --- 1. Base Calcs ---
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        df['body_size'] = (df['close'] - df['open']).abs()
        df['body_size_norm'] = df['body_size'] / df['close']
        df['is_bullish'] = np.where(df['close'] >= df['open'], 1.0, -1.0)
        
        # --- 2. Indicators ---
        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi() / 100.0
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-6)
        
        # EMAs
        df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['ema_20_dist'] = (df['close'] - df['ema_20']) / df['close']
        df['ema_50_dist'] = (df['close'] - df['ema_50']) / df['close']
        
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
        df['vwap_dist'] = (df['close'] - df['vwap']) / df['close']
        
        # Volume
        df['normalized_volume'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-6)
        
        # --- 3. Time Embeddings ---
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        else:
            # Fallback if no datetime index
            df['hour_sin'] = 0.0
            df['hour_cos'] = 0.0
            df['day_sin'] = 0.0
            df['day_cos'] = 0.0
            
        # Select and Fill
        features = df[self.FEATURE_NAMES].fillna(0.0)
        
        # Clip to avoid massive outliers exploding the NN
        features = features.clip(lower=-10.0, upper=10.0)
        
        return features

    def update_stream(self, tick: Dict[str, Union[float, str]]) -> np.ndarray:
        """
        Streaming Update (Inference Mode).
        Adds tick to history -> Recomputes tail -> Returns 1D array.
        """
        # 1. Parse Tick
        # Ensure tick has basic columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(k in tick for k in required):
            raise ValueError(f"Tick missing required keys: {required}")
            
        # Add to history
        self.history.append(tick)
        
        # 2. Convert history to DataFrame
        # Optimization: We only need enough history for the longest lookback (50 eps)
        # But we keep more (200) to be safe.
        df_hist = pd.DataFrame(list(self.history))
        
        # Handle datetime if present (for 'time' column)
        if 'time' in tick:
            df_hist['time'] = pd.to_datetime(df_hist['time'])
            df_hist.set_index('time', inplace=True)
            
        # 3. Compute Batch on Window
        df_features = self.compute_batch(df_hist)
        
        # 4. Return ONLY the last row
        return df_features.iloc[-1].values.astype(np.float32)

# Global Singleton
class FeatureRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FeatureRegistry, cls).__new__(cls)
            cls._instance.engine = UnifiedFeatureEngine()
        return cls._instance
    
    @property
    def feature_names(self):
        return self.engine.FEATURE_NAMES
        
    @property
    def output_dim(self):
        return self.engine.get_output_dim()

    def compute_batch(self, df):
        return self.engine.compute_batch(df)

    def update_stream(self, tick):
        return self.engine.update_stream(tick)

# Compatibility wrapper if needed (deprecated)
def add_features(df):
    reg = FeatureRegistry()
    return reg.compute_batch(df)
