
import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.brain.env.trading_env import TradingEnv

# Settings
MODEL_PATH = "models/ppo_multi_asset_final.zip"
DATA_PATH = "data/processed/XAUUSD_D1_processed.parquet" # D1 was the best performer

def analyze_performance():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data not found: {DATA_PATH}")
        return

    print(f"📊 Analyzing Model: {MODEL_PATH}")
    print(f"📈 Data: {DATA_PATH}")

    # Load Data
    df = pd.read_parquet(DATA_PATH)
    
    # Initialize Env
    env = TradingEnv(df)
    # Relax Risk Limits for Analysis
    env.risk_manager.max_lots_per_trade = 100.0
    obs, _ = env.reset()
    
    # Load Model
    model = PPO.load(MODEL_PATH)
    
    # Simulation Loop
    done = False
    equity_curve = []
    trades = []
    
    current_trade = None
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        equity_curve.append(env.equity)
        
        # Track Trades
        # Info contains 'last_trade': {'action': 'BUY', 'price': ..., 'units': ...}
        if 'last_trade' in info:
            trade_event = info['last_trade']
            action_type = trade_event['action']
            price = trade_event['price']
            time = df.iloc[env.current_step]['time']
            
            if action_type == 'BUY':
                if current_trade is None:
                    current_trade = {
                        'entry_time': time,
                        'entry_price': price,
                        'units': trade_event['units']
                    }
            elif action_type == 'SELL':
                if current_trade:
                    # Close Trade
                    profit = (price - current_trade['entry_price']) * current_trade['units']
                    duration = time - current_trade['entry_time']
                    
                    trades.append({
                        'entry_time': current_trade['entry_time'],
                        'exit_time': time,
                        'entry_price': current_trade['entry_price'],
                        'exit_price': price,
                        'profit': profit,
                        'return_pct': (price - current_trade['entry_price']) / current_trade['entry_price'] * 100,
                        'duration': duration,
                        'duration_hours': duration.total_seconds() / 3600
                    })
                    current_trade = None

    # Calculate Statistics
    initial_balance = env.initial_balance
    final_balance = env.equity
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    
    if not trades:
        print("⚠️ No trades executed.")
        return

    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['profit'] > 0]
    losing_trades = trades_df[trades_df['profit'] <= 0]
    
    win_rate = (len(winning_trades) / total_trades) * 100
    avg_profit = trades_df['profit'].mean()
    max_profit = trades_df['profit'].max()
    max_loss = trades_df['profit'].min()
    
    avg_holding_time = trades_df['duration_hours'].mean()
    max_holding_time = trades_df['duration_hours'].max()
    min_holding_time = trades_df['duration_hours'].min()
    
    # Drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    print("\n" + "="*40)
    print(f"📊 PERFORMANCE REPORT (XAUUSD D1)")
    print("="*40)
    print(f"💰 Final Equity:     ${final_balance:,.2f} ({total_return:+.2f}%)")
    print(f"📉 Max Drawdown:     {max_drawdown:.2f}%")
    print("-" * 20)
    print(f"🔢 Total Orders:     {total_trades}")
    print(f"✅ Win Rate:         {win_rate:.2f}% ({len(winning_trades)} W / {len(losing_trades)} L)")
    print(f"⚖️ Avg Profit/Trade: ${avg_profit:.2f}")
    print(f"🚀 Best Trade:       ${max_profit:.2f}")
    print(f"🔻 Worst Trade:      ${max_loss:.2f}")
    print("-" * 20)
    print(f"⏱️ Avg Holding Time: {avg_holding_time:.1f} hours ({avg_holding_time/24:.1f} days)")
    print(f"⏳ Max Holding Time: {max_holding_time:.1f} hours ({max_holding_time/24:.1f} days)")
    print(f"⚡ Min Holding Time: {min_holding_time:.1f} hours")
    print("="*40)

if __name__ == "__main__":
    analyze_performance()
