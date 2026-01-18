
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional
from collections import deque

class TradingEnv(gym.Env):
    """
    A reinforcement learning environment for trading using Gymnasium.
    Enhanced with research-based reward functions (Sharpe ratio, risk management).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance=10000, max_steps=None, render_mode=None):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.render_mode = render_mode
        self.max_steps = max_steps if max_steps else len(df) - 1
        
        # Sharpe ratio tracking (research-based)
        self.returns_history = deque(maxlen=100)  # Last 100 returns for Sharpe calculation
        
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
        
        # Level 3: Add News Impact if available
        if 'news_impact_score' in df.columns:
            self.feature_cols.append('news_impact_score')
        
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
        
        # CRITICAL FIX: Start with BALANCED portfolio (50/50)
        # This gives BUY and SELL equal opportunity from the start
        initial_price = self.df.iloc[0]['close']
        
        # Split initial balance 50/50 between cash and position
        self.balance = self.initial_balance * 0.5  # 50% cash
        position_value = self.initial_balance * 0.5  # 50% in asset  
        self.position = position_value / initial_price  # Convert to units
        
        # Start equity = balance + position value
        self.equity = self.initial_balance
        self.trades_history = []
        
        # Reset Sharpe tracking
        self.returns_history.clear()
        
        # Log initial state
        print(f"🔄 Reset: Balance=${self.balance:.2f}, Position={self.position:.4f} units @ ${initial_price:.2f}")
        
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
        
        # Position limits (prevent unlimited positions)
        MAX_POSITION_VALUE = self.initial_balance * 2.0  # Max 2x initial balance in position
        current_position_value = abs(self.position * current_price)
        
        reward = 0
        prev_equity = self.equity
        trade_info = None
        
        if action == 1:  # BUY
            # Can only buy if we have balance AND not over position limit
            if self.balance > 0 and current_position_value < MAX_POSITION_VALUE:
                # Use 99% of balance (keep 1% buffer)
                cost = self.balance * 0.99
                fee = cost * 0.001  # 0.1% transaction fee
                invest_amount = cost - fee
                
                if invest_amount > 0:
                    units = invest_amount / current_price
                    self.position += units
                    self.balance -= cost
                    trade_info = {'action': 'BUY', 'price': current_price, 'units': units}
                    
        elif action == 2:  # SELL
            # Can only sell if we have position
            if self.position > 0:
                # Sell all position
                revenue = self.position * current_price
                fee = revenue * 0.001  # 0.1% transaction fee
                self.balance += (revenue - fee)
                sold_units = self.position
                self.position = 0
                trade_info = {'action': 'SELL', 'price': current_price, 'units': sold_units}
        
        # Update Step
        self.current_step += 1
        
        # Calculate Equity
        if self.current_step < len(self.df):
            next_price = self.df.iloc[self.current_step]['close']
            self.equity = self.balance + (self.position * next_price)
        else:
            self.equity = self.balance + (self.position * current_price)
        
        # ===== RESEARCH-BASED REWARD FUNCTION =====
        # Reference: Multiple research papers on PPO for trading
        
        # 1. Calculate log return (base component)
        if prev_equity > 0:
            log_return = np.log(self.equity / prev_equity)
        else:
            log_return = 0
        
        # Track returns for Sharpe calculation
        self.returns_history.append(log_return)
        
        # 2. Calculate Differential Sharpe Ratio (research-based)
        # This approximates change in Sharpe ratio at each step
        if len(self.returns_history) >= 10:  # Need minimum history
            returns_array = np.array(self.returns_history)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array) + 1e-6  # Avoid division by zero
            sharpe_contribution = (log_return - mean_return) / std_return
        else:
            sharpe_contribution = 0
        
        # 3. Transaction cost penalty (if trade occurred)
        transaction_penalty = 0
        if trade_info is not None:
            # Already deducted from balance, but signal it in reward
            transaction_penalty = -0.01  # Small penalty to discourage over-trading
        
        # 4. Drawdown penalty (research-based risk management)
        current_drawdown = (self.equity - self.initial_balance) / self.initial_balance
        drawdown_penalty = 0
        if current_drawdown < -0.1:  # More than 10% loss
            drawdown_penalty = -0.05  # Significant penalty
        elif current_drawdown < -0.05:  # More than 5% loss
            drawdown_penalty = -0.02  # Moderate penalty
        
        # 5. Composite reward (weighted combination)
        # Weights based on research best practices
        reward = (
            0.5 * log_return +           # Base profit/loss
            0.3 * sharpe_contribution +  # Risk-adjusted performance
            0.1 * transaction_penalty +  # Discourage over-trading
            0.1 * drawdown_penalty       # Risk management
        )
        
        # 6. Portfolio balance bonus (encourage diversification)
        # Small bonus for keeping balanced portfolio
        portfolio_imbalance = abs(self.balance - (self.position * current_price))
        if portfolio_imbalance < self.initial_balance * 0.3:  # Well balanced
            reward += 0.005  # Small bonus
        
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
