
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.brain.env.trading_env import TradingEnv

def backtest_force_exit():
    MODEL_PATH = "models/neurotrader_v2_lstm.zip"
    # Use H1 data for Intraday simulation (allows exiting at specific hour)
    DATA_PATH = "data/processed/XAUUSD_H1_processed.parquet" 
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data not found: {DATA_PATH}")
        return

    print("🧠 Loading NeuroTrader 2.0 (LSTM)...")
    model = RecurrentPPO.load(MODEL_PATH)
    
    print(f"📈 Loading Test Data (H1): {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    # Filter last 2000 bars for a faster, recent test
    df = df.tail(2000).reset_index(drop=True)
    
    env = TradingEnv(df)
    env.risk_manager.max_lots_per_trade = 100.0
    
    obs, _ = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    done = False
    equity_curve = []
    trades = []
    current_trade = None
    
    print("🚀 Running Intraday Force Exit Test (Close at 23:00)...")
    
    while not done:
        # 1. Check Time for Force Exit
        current_step_idx = env.current_step
        # Safety check for index bound
        if current_step_idx >= len(df):
            break
            
        current_time = df.iloc[current_step_idx]['time']
        current_hour = current_time.hour
        
        # FORCE EXIT RULE: If Hour >= 23 (End of Day), Force Close.
        # We override the model's decision 
        force_close = False
        if env.position > 0 and current_hour >= 23:
            force_close = True
        
        if force_close:
            # Manually execute CLOSE action (Action 2 = Sell/Close)
            action = np.array([2]) 
            # We still need to update LSTM state, so we predict but ignore action
            _, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        else:
            # Normal AI Decision
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        equity_curve.append(env.equity)
        
        # Track Trades
        if 'last_trade' in info:
            trade_event = info['last_trade']
            if trade_event['action'] == 'BUY':
                if current_trade is None:
                    current_trade = {
                        'entry_time': current_time, 
                        'entry_price': trade_event['price'],
                        'units': trade_event['units']
                    }
            elif trade_event['action'] == 'SELL':
                if current_trade:
                    profit = (trade_event['price'] - current_trade['entry_price']) * current_trade['units']
                    duration = current_time - current_trade['entry_time']
                    trades.append({
                        'profit': profit,
                        'duration_hours': duration.total_seconds() / 3600,
                        'exit_reason': 'Force Close' if force_close else 'AI Signal'
                    })
                    current_trade = None

        episode_starts = np.zeros((1,), dtype=bool)

    # Statistics
    final_return = ((env.equity - env.initial_balance) / env.initial_balance) * 100
    
    if trades:
        trades_df = pd.DataFrame(trades)
        win_rate = (len(trades_df[trades_df['profit'] > 0]) / len(trades_df)) * 100
        avg_profit = trades_df['profit'].mean()
        avg_duration = trades_df['duration_hours'].mean()
        forced_exits = len(trades_df[trades_df['exit_reason'] == 'Force Close'])
    else:
        win_rate = 0
        avg_profit = 0
        avg_duration = 0
        forced_exits = 0

    print("\n" + "="*40)
    print(f"📊 INTRADAY MODE RESULT (XAUUSD H1)")
    print("="*40)
    print(f"💰 Final Return:     {final_return:+.2f}%")
    print(f"🔢 Total Trades:     {len(trades)}")
    print(f"✅ Win Rate:         {win_rate:.2f}%")
    print(f"⏱️ Avg Duration:     {avg_duration:.1f} hours")
    print(f"🛑 Forced Exits:     {forced_exits} (Trades closed by Time Rule)")
    print("="*40)

if __name__ == "__main__":
    backtest_force_exit()
