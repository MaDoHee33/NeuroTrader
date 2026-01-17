
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional

class TradingEnv(gym.Env):
    """
    A reinforcement learning environment for trading using Gymnasium.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance=10000, max_steps=None, render_mode=None):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.render_mode = render_mode
        self.max_steps = max_steps if max_steps else len(df) - 1
        
        # Action Space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: 
        # [Close Price, RSI, MACD, MACD_Signal, BB_High, BB_Low, EMA_20, EMA_50, Balance, Position]
        # We assume these columns exist in the dataframe        # Features expected in DF
        self.feature_cols = [
            'close', 'rsi', 'macd', 'macd_signal', 
            'bb_high', 'bb_low', 'ema_20', 'ema_50',
            'atr', 'log_ret_lag_1', 'log_ret_lag_2', 'log_ret_lag_3', 'log_ret_lag_5'
        ]
        
        # Check if cols exist
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataframe missing required columns: {missing_cols}")

        num_features = len(self.feature_cols) + 2 # +2 for balance and position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0 # Asset amount held
        self.equity = initial_balance
        self.trades_history = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.equity = self.initial_balance
        self.trades_history = []
        
        # Random start index if data is large enough for variety?
        # For now, start at 0 (or strictly start after window size if we use window)
        # But our DF is already stripped of NaNs, so safe to start at 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Get current row
        row = self.df.iloc[self.current_step]
        
        obs = [
            row[col] for col in self.feature_cols
        ]
        # Add account state
        obs.append(self.balance)
        obs.append(self.position)
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        # Execute Action
        # Simplification: All-in Buy/Sell for now to train basic logic
        # OR Fixed Size. Let's do fixed size for stability: 0.1 units (assuming like crypto/forex)
        # Or better: Spend 10% of balance (Buying) / Sell 100% of position (Selling)
           
        reward = 0
        prev_equity = self.equity
        
        trade_info = None
        
        if action == 1: # BUY
            # Can only buy if we have balance
            if self.balance > 0:
                # Buy as much as possible? Or fixed amount?
                # Let's say we buy with 99% of balance (minus fee placeholder)
                cost = self.balance
                fee = cost * 0.001 # 0.1% fee
                invest_amount = cost - fee
                
                if invest_amount > 0:
                    units = invest_amount / current_price
                    self.position += units
                    self.balance = 0 # Spent all
                    trade_info = {'action': 'BUY', 'price': current_price, 'units': units}
                    
        elif action == 2: # SELL
            # Can only sell if we have position
            if self.position > 0:
                revenue = self.position * current_price
                fee = revenue * 0.001
                self.balance += (revenue - fee)
                self.position = 0
                trade_info = {'action': 'SELL', 'price': current_price, 'units': 0}

        # Update Step
        self.current_step += 1
        
        # Calculate Equity
        self.equity = self.balance + (self.position * self.df.iloc[self.current_step]['close'] if self.current_step < len(self.df) else 0)
        
        # Reward: Change in equity (Log return is better for training stability)
        # reward = self.equity - prev_equity
        # Using Log Return: ln(current_equity / prev_equity)
        # Handle zero division just in case
        if prev_equity > 0:
             reward = np.log(self.equity / prev_equity)
        else:
             reward = 0

        # Terminate
        done = False
        truncated = False
        
        if self.current_step >= len(self.df) - 1:
            done = True
        
        if self.equity <= 0: # Bust
            done = True
            reward = -10 # Heavy penalty for busting
            
        info = {
            'equity': self.equity,
            'step': self.current_step
        }
        if trade_info:
            info['last_trade'] = trade_info
            
        return self._get_observation(), reward, done, truncated, info

    def render(self):
        if self.render_mode == 'human':
            print(f"Step: {self.current_step} | Balance: {self.balance:.2f} | Position: {self.position:.4f} | Equity: {self.equity:.2f}")

    def close(self):
        pass
