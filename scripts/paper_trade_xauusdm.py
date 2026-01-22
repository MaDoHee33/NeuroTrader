
import time
import sys
import pandas as pd
import numpy as np
import asyncio
import argparse
from pathlib import Path
import MetaTrader5 as mt5
from stable_baselines3 import PPO

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.body.mt5_driver import MT5Driver
from src.brain.feature_eng import add_features

class LiveTrader:
    def __init__(self, model_path, symbol="XAUUSDm", timeframe="M5", live_execution=False):
        self.symbol = symbol
        self.timeframe = timeframe
        self.driver = MT5Driver(config={'system': {'shadow_mode': not live_execution}})
        
        print(f"🧠 Loading Model: {model_path}")
        self.model = PPO.load(model_path)
        
        self.feature_cols = [
            'body_size', 'upper_wick', 'lower_wick', 'is_bullish',
            'ema_50', 'dist_ema_50', 'dist_ema_200',
            'rsi', 'atr_norm',
            'dist_to_high', 'dist_to_low',
            'log_ret', 'log_ret_lag_1', 'log_ret_lag_2'
        ]
        
    async def run_loop(self):
        print("🔌 Connecting to MT5...")
        if not await self.driver.initialize():
            return

        print(f"🚀 Starting Paper Trading on {self.symbol} [{self.timeframe}]...")
        print(f"🛡️  Mode: {'LIVE EXECUTION (DEMO)' if not self.driver.shadow_mode else 'SHADOW (PAPER)'}")
        
        last_candle_time = None
        
        while True:
            try:
                # 1. Fetch Data (Need enough for EMA200)
                df = await self.driver.fetch_history(self.symbol, self.timeframe, count=300)
                if df is None or df.empty:
                    print("⚠️  No data received. Retrying...")
                    await asyncio.sleep(5)
                    continue

                # Check if new candle
                latest_time = df.iloc[-1]['time']
                if last_candle_time == latest_time:
                    # Wait for next update
                    print(".", end="", flush=True)
                    await asyncio.sleep(5) 
                    continue
                
                print(f"\n🕯️  New Candle: {latest_time}")
                last_candle_time = latest_time
                
                # 2. Add Features
                try:
                    df = add_features(df)
                    row = df.iloc[-1]
                except Exception as e:
                    print(f"❌ Feature Eng Error: {e}")
                    continue

                # 3. Get Account Info
                account = await self.driver.get_account_info()
                balance = account['balance'] if account else 10000.0
                
                positions = await self.driver.get_positions(self.symbol)
                current_position_size = sum(p['volume'] for p in positions) if positions else 0.0
                
                # 4. Construct Observation (Must wait feature list)
                feats = [row[c] for c in self.feature_cols]
                
                # Normalize Balance/Pos for Model (Model expects Raw numbers based on Training Env?)
                # In Training Env: obs.append(self.balance), obs.append(self.position)
                # Position in Env was 'units'. MT5 is 'lots'.
                # 1 Lot XAUUSD = 100 Units? Or 100 oz?
                # Usually XAUUSD 1 Lot = 100 oz. Price ~2000. Value $200,000.
                # In our Env, we used price directly. position = value / price -> units.
                # So if we have 0.01 Lots -> 1 oz -> 1 Unit ??
                # Let's approximate: 1 Lot = 100 Units.
                # Current Position (Units) = current_position_size * 100 (Contract Size)
                # But wait, Env logic: units = invest_amount / current_price
                # If invest 1000, price 2000 => 0.5 units.
                # So Unit = 1 oz.
                # MT5 Lot = 100 oz (Standard).
                # So 0.01 Lot = 1 oz = 1 Unit.
                # If we hold 0.01 lot, we hold 1 unit.
                # So position input to model should be: current_position_size * 100.
                
                # Simple check: Contract Size
                symbol_info = mt5.symbol_info(self.symbol)
                contract_size = symbol_info.trade_contract_size if symbol_info else 100
                model_position_units = current_position_size * contract_size
                
                obs = np.array(feats + [balance, model_position_units], dtype=np.float32)
                
                # 5. Predict
                action_arr, _ = self.model.predict(obs, deterministic=True)
                action = action_arr.item()
                
                action_map = {0: "HOLD", 1: "BUY", 2: "SELL (CLOSE)"}
                print(f"🧠 Analysis: {action_map.get(action, 'UNKNOWN')} (Act: {action})")
                
                # 6. Execute
                decision = {
                    'action': action_map[action],
                    'symbol': self.symbol,
                    'volume': 0.01 # Fixed 0.01 Lot for safety
                } # In real logic we might size based on kelly or balance
                
                # Only execute if action != HOLD
                if action != 0:
                    await self.driver.execute_trade(decision)

            except KeyboardInterrupt:
                print("\n🛑 Stopping...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(5)

        self.driver.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='Enable REAL trade execution (Demo)')
    parser.add_argument('--symbol', type=str, default='XAUUSDm')
    parser.add_argument('--model', type=str, default='models/ppo_multi_asset_final.zip')
    args = parser.parse_args()
    
    trader = LiveTrader(args.model, symbol=args.symbol, live_execution=args.live)
    asyncio.run(trader.run_loop())
