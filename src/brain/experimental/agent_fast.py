
import numpy as np
import pandas as pd
from collections import deque
from stable_baselines3 import PPO
from pathlib import Path
import os
from src.brain.feature_eng import add_features

class FastRLAgent:
    """
    Optimized RLAgent for High-Speed Training
    - Reduced History Window (60 vs 100)
    - Normalized Observation Space
    """
    def __init__(self, config=None, model_path=None):
        self.config = config or {}
        self.model_path = model_path
        self.model = None
        
        # OPTIMIZATION: Reduced history size for faster processing
        self.history_size = 60 
        self.history = deque(maxlen=self.history_size)
    
    def process_bar(self, bar_dict):
        """
        Process incoming bar and return trading action
        """
        # Add to history
        self.history.append(bar_dict)
        
        # Warmup
        if len(self.history) < self.history_size: 
            return 0  # HOLD
        
        # Feature Engineering
        # In a highly optimized version, we would calculate features incrementally.
        # For now, we stick to DataFrame but with shorter history.
        df = pd.DataFrame(list(self.history))
        
        try:
            df_features = add_features(df)
            
            if len(df_features) > 0:
                latest = df_features.iloc[-1]
                
                # OPTIMIZATION: Select only most stable/normalized features
                # 19 Features total to match PPO input
                feature_cols = [
                    'rsi', 'macd', 'macd_signal', 'bb_width', 
                    'ema_20', 'atr', 'log_return', 'stoch_k', 
                    'stoch_d', 'vwap', 
                    # Normalized Price Action (Crucial for stability)
                    'normalized_close', 'normalized_volume',
                    # Momentum relative to ranges
                    'high', 'low', 'close', 'open', 'volume', # Raw values (should be normalized in real prod)
                    'bb_upper', 'bb_lower' 
                ]
                
                obs_values = []
                for col in feature_cols:
                    val = float(latest.get(col, 0.0))
                    
                    # EXPERIMENTAL: dynamic normalization for raw prices
                    if col in ['open', 'high', 'low', 'close', 'ema_20', 'bb_upper', 'bb_lower', 'vwap']:
                         # Normalize relative to close to make it scale-invariant
                         val = (val / float(latest.get('close', 1.0))) - 1.0
                    
                    obs_values.append(val)
                
                # Fill to ensure 19
                while len(obs_values) < 19:
                    obs_values.append(0.0)
                    
                obs = np.array(obs_values[:19], dtype=np.float32)
                
                if self.model:
                    action, _ = self.model.predict(obs, deterministic=True)
                    return int(action)
                
        except Exception as e:
            # print(f"Error: {e}") 
            pass
            
        return 0 # Default HOLD

    def load_model(self, path):
        if os.path.exists(path):
            self.model = PPO.load(path)
            # print(f"Loaded model from {path}")
            return True
        return False
