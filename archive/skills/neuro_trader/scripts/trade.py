import asyncio
import os
import sys
import argparse
import pandas as pd
import numpy as np
import ta
import time
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

# Add src to path
# Path: skills/neuro_trader/scripts/trade.py -> Root is ../../../
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.body.mt5_driver import MT5Driver
from src.utils.logger import get_logger
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

class LiveTrader:
    def __init__(self, model_path, symbol="XAUUSDm", timeframe="H1", volume=0.01, level=2):
        self.logger = get_logger("LiveTrader")
        self.symbol = symbol
        self.timeframe = timeframe
        self.volume = volume
        self.level = level
        
        # Connect to MT5
        self.driver = MT5Driver()
        
        # Load Model
        self.logger.info(f"🧠 Loading Model: {model_path}")
        if level == 2:
            self.model = RecurrentPPO.load(model_path)
            self.lstm_states = None
            self.episode_starts = np.ones((1,), dtype=bool)
        else:
            self.model = PPO.load(model_path)
            
        # Feature Cols (Must match training exactly)
        self.feature_cols = [
            'close', 'rsi', 'macd', 'macd_signal', 
            'bb_high', 'bb_low', 'ema_20', 'ema_50'
        ]
        if level >= 2:
            self.feature_cols.extend(['atr', 'log_ret_lag_1', 'log_ret_lag_2', 'log_ret_lag_3', 'log_ret_lag_5'])
            
    async def initialize(self):
        return await self.driver.initialize()
        
    def calculate_features(self, df):
        """Replicates process_data.py logic for a rolling window."""
        df = df.copy()
        
        # Basic Indicators
        df['ema_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2.0)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        
        if self.level >= 2:
            df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            for lag in [1, 2, 3, 5]:
                df[f'log_ret_lag_{lag}'] = df['log_ret'].shift(lag)
                
        return df

    async def trade_loop(self):
        self.logger.info(f"🚀 Starting Live Trader on {self.symbol} ({self.timeframe})")
        
        while True:
            try:
                # 1. Fetch Data (Enough for Lag 5 + Macd 26 + EMA 50 -> Safe 200+)
                df = await self.driver.fetch_history(self.symbol, self.timeframe, count=500)
                
                if df is None or len(df) < 100:
                    self.logger.warning("Not enough data. Retrying...")
                    await asyncio.sleep(10)
                    continue
                
                # 2. Process Features
                df = self.calculate_features(df)
                
                # Get last complete candle (iloc[-1] is usually current evolving candle in MT5?)
                # If we want closed candle strategy, we normally take iloc[-2].
                # But for now let's assume we trade on Close of last finished candle.
                # However, MT5 'fetch_history' usually returns *finished* bars + current open bar.
                # Let's use iloc[-1] (latest available) and assume it's the one we decide on.
                # WARNING: If using H1, and minute is 00:30, iloc[-1] is the current forming H1.
                # Models usually trained on closed prices. 
                # Ideally we should wait for candle close. 
                # For simplicity: Use iloc[-2] (Last Closed Candle) to prevent Repainting.
                
                current_row = df.iloc[-2]  # Safe
                
                # 3. Construct Observation
                obs = [current_row[col] for col in self.feature_cols]
                # Append Account Info (Balance, Position) 
                # TODO: Get real balance/position from MT5. For now, mock or 0.
                # If we passed balance=0 during training, it might affect model.
                # Training Env had: [Features] + [Balance] + [Position]
                
                # Mock Account State for Consistency with Training
                # Ideally we fetch AccountInfo from MT5
                obs.append(10000.0) # Balance
                obs.append(0.0)     # Position
                
                obs_array = np.array(obs, dtype=np.float32)
                
                # 4. Predict
                if self.level == 2:
                    action, self.lstm_states = self.model.predict(
                        obs_array, 
                        state=self.lstm_states, 
                        episode_start=self.episode_starts
                    )
                    self.episode_starts = np.zeros((1,), dtype=bool)
                else:
                    action, _ = self.model.predict(obs_array)
                    
                action = int(action)
                
                # 5. Execute
                action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
                price = current_row['close']
                
                decision = {
                    "action": action_map.get(action, "HOLD"),
                    "symbol": self.symbol,
                    "price": price,
                    "volume": self.volume,
                    "time": str(current_row['time'])
                }
                
                self.logger.info(f"📊 Signal: {decision['action']} @ {price} (Time: {current_row['time']})")
                
                if action != 0:
                    await self.driver.execute_trade(decision)
                    
                # Wait for next candle? or Sleep fixed time?
                # H1 = 3600s. Sleeping 60s is fine to check updates.
                print(f"💤 Sleeping... (Last close: {price})")
                await asyncio.sleep(60) 
                
            except Exception as e:
                self.logger.error(f"Error in trade loop: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model.zip")
    parser.add_argument("--symbol", type=str, default="XAUUSDm")
    parser.add_argument("--volume", type=float, default=0.01)
    parser.add_argument("--level", type=int, default=2)
    
    args = parser.parse_args()
    
    trader = LiveTrader(args.model, args.symbol, volume=args.volume, level=args.level)
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(trader.initialize())
    loop.run_until_complete(trader.trade_loop())
