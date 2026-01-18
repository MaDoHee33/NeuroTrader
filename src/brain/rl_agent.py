
import asyncio
import numpy as np
import pandas as pd
from collections import deque
from stable_baselines3 import PPO
import os

from src.brain.feature_eng import add_features

class RLAgent:
    def __init__(self, config=None):
        self.config = config or {}
        self.model_path = "models/checkpoints/ppo_neurotrader.zip"
        self.model = None
        
        # History buffer for feature engineering
        # Needs at least 50-100 bars for EMA50/MACD etc.
        self.history_size = 100 
        self.history = deque(maxlen=self.history_size)
        
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                print(f"🧠 RLAgent: Loading model from {self.model_path}")
                self.model = PPO.load(self.model_path)
            except Exception as e:
                print(f"❌ RLAgent: Failed to load model: {e}")
        else:
            print(f"⚠️ RLAgent: Model not found at {self.model_path}. Agent will act randomly or HOLD.")

    async def decide(self, market_data: dict, sentiment: dict, forecast: dict, portfolio_state: dict = None):
        """
        Main decision method.
        market_data: {close, volume, high, low, open, timestamp}
        """
        # 1. Update History
        self.history.append(market_data)
        
        # 2. Warmup Check
        if len(self.history) < 60: # Need at least 50 for EMA + safety buffer
            # Return HOLD (0) or Random until we have enough data
            return {"action": "HOLD", "volume": 0.0}

        # 3. Feature Engineering (State Construction)
        try:
            df = pd.DataFrame(list(self.history))
            df = df.sort_values('timestamp') # Ensure order
            
            # This adds rsi, macd, etc. 
            # Note: add_features DROPS rows with NaNs.
            # So if we have 60 rows, we might end up with just 1 or 2 valid rows at the end.
            df_features = add_features(df)
            
            if df_features.empty:
                # Not enough valid data yet
                return {"action": "HOLD"}
            
            # Get the very last row (most recent observation)
            last_row = df_features.iloc[-1]
            
            # Construct observation vector matching TradingEnv
            # self.feature_cols + balance + position
            # Note: We need to know the 'Balance' and 'Position' state from the environment/strategy.
            # Currently 'market_data' doesn't include portfolio state.
            # FIXME: Strategy needs to pass portfolio state!
            
            # For now, let's assume fully invested or something fixed, 
            # OR request strategy to pass this info.
            # To avoid breaking interface now, let's use placeholders 
            # provided by market_data if available, or static assumptions.
            
            # Ideally, NeuroBridgeStrategy should pass 'portfolio': {'balance': ..., 'position': ...}
            # Let's check keys in last_row:
            # ['close', 'rsi', 'macd', 'macd_signal', 'bb_high', 'bb_low', 'ema_20', 'ema_50', 'atr', 'log_ret_lag_1'...]
            
            required_cols = [
                'close', 'rsi', 'macd', 'macd_signal', 
                'bb_high', 'bb_low', 'ema_20', 'ema_50',
                'atr', 'log_ret_lag_1', 'log_ret_lag_2', 'log_ret_lag_3', 'log_ret_lag_5'
            ]
            
            # Filter just the feature columns
            obs_features = last_row[required_cols].values.astype(np.float32)
            
            # Add Balance/Position
            if portfolio_state:
                balance = portfolio_state.get('balance', 10000.0)
                position = portfolio_state.get('position', 0.0)
            else:
                balance = 10000.0 # Mock
                position = 0.0    # Mock
            
            obs = np.concatenate([obs_features, [balance, position]])
            
            # 4. Predict
            if self.model:
                action_idx, _ = self.model.predict(obs, deterministic=True)
                # Action Map: 0=HOLD, 1=BUY, 2=SELL
                action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
                action_str = action_map.get(int(action_idx), "HOLD")
                
                return {
                    "action": action_str,
                    "volume": 0.02, # Fixed small size for safety
                    "raw_action": int(action_idx)
                }
            else:
                return {"action": "HOLD"} # No model loaded
                
        except Exception as e:
            print(f"RLAgent Error: {e}")
            return {"action": "HOLD"}
