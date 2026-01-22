import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from collections import deque
from src.brain.risk_manager import RiskManager

class TradingEnv(gym.Env):
    """
    A reinforcement learning environment for trading using Gymnasium.
    Enhanced with research-based reward functions (Sharpe ratio, risk management).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df_dict: dict, initial_balance=10000, max_steps=None, render_mode=None):
        super(TradingEnv, self).__init__()
        
        # Determine if single DF or Dict
        if isinstance(df_dict, pd.DataFrame):
            self.assets = {'DEFAULT': df_dict}
        else:
            self.assets = df_dict
            
        self.asset_names = list(self.assets.keys())
        self.current_asset_name = self.asset_names[0]
        self.df = self.assets[self.current_asset_name]
        
        self.initial_balance = initial_balance
        self.render_mode = render_mode
        self.max_steps = max_steps # dynamic per episode usually
        
        # Initialize Risk Manager
        self.risk_manager = RiskManager({
            'max_lots': 1.0,           # Max 1.0 Lot per trade
            'daily_loss_pct': 0.05,    # 5% Daily Loss Stop
            'max_drawdown_pct': 0.20   # 20% Max Drawdown Circuit Breaker
        })
        
        # Sharpe ratio tracking
        self.returns_history = deque(maxlen=100)  # Last 100 returns for Sharpe calculation
        
        # Action Space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        
        # Updated Hybrid Features
        self.feature_cols = [
            'body_size', 'upper_wick', 'lower_wick', 'is_bullish',
            'ema_50', 'dist_ema_50', 'dist_ema_200',
            'rsi', 'atr_norm',
            'dist_to_high', 'dist_to_low',
            'log_ret', 'log_ret_lag_1', 'log_ret_lag_2',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        # Check cols on first asset
        missing_cols = [c for c in self.feature_cols if c not in self.df.columns]
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
        
        # Randomly select asset
        import random
        self.current_asset_name = random.choice(self.asset_names)
        self.df = self.assets[self.current_asset_name]
        
        # Reset constraints
        self.max_steps_episode = len(self.df) - 1
        
        self.current_step = 0
        initial_price = self.df.iloc[0]['close']
        
        # Start Balanced
        self.balance = self.initial_balance * 0.5  # 50% cash
        position_value = self.initial_balance * 0.5  # 50% in asset  
        self.position = position_value / initial_price  # Convert to units
        
        self.equity = self.initial_balance
        self.trades_history = []
        self.returns_history.clear()
        
        print(f"🔄 Reset [{self.current_asset_name}]: Balance=${self.balance:.2f}")
        
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
        # 0. Risk Check: Circuit Breaker
        current_status = self.risk_manager.get_status()
        if current_status['circuit_breaker']:
            # Halt trading immediately
            return self._get_observation(), -1.0, True, False, {'error': 'Circuit Breaker Active'}

        current_price = self.df.iloc[self.current_step]['close']
        
        # Get Current Time for Risk Manager
        # Try index first, then 'date'/'time' column
        current_time = self.df.index[self.current_step]
        if not isinstance(current_time, (pd.Timestamp, datetime, np.datetime64)):
             if 'time' in self.df.columns:
                 current_time = self.df.iloc[self.current_step]['time']
             elif 'date' in self.df.columns:
                 current_time = self.df.iloc[self.current_step]['date']
        
        # Position limits (prevent unlimited positions)
        MAX_POSITION_VALUE = self.initial_balance * 2.0  # Max 2x initial balance
        current_position_value = abs(self.position * current_price)
        
        reward = 0
        prev_equity = self.equity
        trade_info = None
        risk_penalty = 0
        
        if action == 1:  # BUY
            # Can only buy if we have balance AND not over position limit
            if self.balance > 0 and current_position_value < MAX_POSITION_VALUE:
                # Use 99% of balance (keep 1% buffer)
                cost = self.balance * 0.99
                invest_amount = cost * (1 - 0.001) # Deduct fee approx
                
                if invest_amount > 0:
                    units = invest_amount / current_price
                    
                    # RISK MANAGER CHECK
                    if self.risk_manager.check_order("PAIR", units, "BUY", current_time):
                        # Allowed
                        self.position += units
                        self.balance -= cost
                        trade_info = {'action': 'BUY', 'price': current_price, 'units': units}
                    else:
                        # Blocked
                        risk_penalty = -0.1 # Small penalty for attempting forbidden trade
                        
        elif action == 2:  # SELL
            # Can only sell if we have position
            if self.position > 0:
                units_to_sell = self.position
                
                # RISK MANAGER CHECK (Usually reducing risk is always allowed, but Max Lots might apply if we treat it as an order)
                # For closing positions, we usually ALLOW it. 
                # But if 'SELL' means 'Short Selling' (opening new position), we check.
                # In this env, action 2 is "Close/Sell All" or "Short"? 
                # Judging by logic: self.position = 0, it means CLOSE ALL.
                # Reducing exposure should generally be allowed.
                # So we bypass check_order for CLOSING positions, or pass is_close=True if our manager supported it.
                # For now, we assume reducing position is always safe.
                
                revenue = self.position * current_price
                fee = revenue * 0.001  # 0.1% transaction fee
                self.balance += (revenue - fee)
                self.position = 0
                trade_info = {'action': 'SELL', 'price': current_price, 'units': units_to_sell}
        
        # Update metrics AFTER action (to see new equity)
        # But we need equity first. So we calculate equity next block.
        
        # Update Step
        self.current_step += 1
        
        # Calculate Equity
        if self.current_step < len(self.df):
            next_price = self.df.iloc[self.current_step]['close']
            self.equity = self.balance + (self.position * next_price)
        else:
            self.equity = self.balance + (self.position * current_price)
        
        # Update Risk Manager Metrics
        self.risk_manager.update_metrics(self.equity, current_time)
        
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
        
        # 3. Transaction cost penalty (Spam filter)
        transaction_penalty = 0
        if trade_info is not None:
             # Penalize each trade slightly to prevent churning
             transaction_penalty = -0.0005 
 
        # 4. Holding Reward (Encourage riding trends)
        holding_reward = 0
        if self.position > 0 and log_return > 0:
            holding_reward = 0.0002 # Bonus for holding while profitable
            
        # 5. Drawdown penalty (research-based risk management)
        current_drawdown = (self.equity - self.initial_balance) / self.initial_balance
        drawdown_penalty = 0
        if current_drawdown < -0.15: # Critical Drawdown (-15%)
            drawdown_penalty = -0.5  # Severe penalty
        elif current_drawdown < -0.10: # High Drawdown (-10%)
            drawdown_penalty = -0.2  # Heavy penalty
        elif current_drawdown < -0.05: # Moderate Drawdown (-5%)
            drawdown_penalty = -0.05  # Warning penalty
        
        # 6. Composite reward (weighted combination)
        # Weights based on research best practices
        reward = (
            0.5 * log_return +           # Base profit/loss
            0.3 * sharpe_contribution +  # Risk-adjusted performance
            0.1 * transaction_penalty +  # Discourage over-trading
            0.1 * drawdown_penalty +     # Risk management
            0.1 * holding_reward +       # Trend following
            1.0 * risk_penalty           # Hard Risk violation penalty
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
