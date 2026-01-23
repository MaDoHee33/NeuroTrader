
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

def backtest_v2():
    MODEL_PATH = "models/neurotrader_v2_lstm.zip"
    DATA_PATH = "data/processed/XAUUSD_D1_processed.parquet" 
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    print("🧠 Loading NeuroTrader 2.0 (LSTM)...")
    # Load Model (must match training env generally, or be compatible)
    model = RecurrentPPO.load(MODEL_PATH)
    
    print(f"📈 Loading Test Data: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    
    # Init Env
    env = TradingEnv(df)
    # Relax risk for testing
    env.risk_manager.max_lots_per_trade = 100.0
    
    obs, _ = env.reset()
    
    # LSTM Memory State
    # Note: RecurrentPPO requires passing 'lstm_states'
    # Start with None/Zeros
    lstm_states = None
    # Start of episode flag
    episode_starts = np.ones((1,), dtype=bool)
    
    done = False
    equity_curve = []
    
    print("🚀 Running Backtest...")
    trades = []
    current_trade = None
    
    while not done:
        # Predict with LSTM State
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts,
            deterministic=True
        )
        
        # Track previous step data for trade logic
        current_step_idx = env.current_step
        current_time = df.iloc[current_step_idx]['time']
        current_price = df.iloc[current_step_idx]['close']
        
        obs, reward, done, truncated, info = env.step(action)
        equity_curve.append(env.equity)
        
        # Track Trades from Info
        if 'last_trade' in info:
            trade_event = info['last_trade']
            action_type = trade_event['action']
            trade_price = trade_event['price']
            
            if action_type == 'BUY':
                if current_trade is None:
                    current_trade = {
                        'entry_time': current_time,
                        'entry_price': trade_price,
                        'units': trade_event['units']
                    }
            elif action_type == 'SELL':
                if current_trade:
                    # Close Trade
                    profit = (trade_price - current_trade['entry_price']) * current_trade['units']
                    duration = current_time - current_trade['entry_time']
                    
                    trades.append({
                        'entry_time': current_trade['entry_time'],
                        'exit_time': current_time,
                        'return_pct': (trade_price - current_trade['entry_price']) / current_trade['entry_price'] * 100,
                        'profit': profit,
                        'duration_hours': duration.total_seconds() / 3600
                    })
                    current_trade = None
        
        # Next step is NOT start of episode
        episode_starts = np.zeros((1,), dtype=bool)

    # Force Close Open Trade at End
    if current_trade:
        last_price = df.iloc[-1]['close']
        last_time = df.iloc[-1]['time']
        
        profit = (last_price - current_trade['entry_price']) * current_trade['units']
        duration = last_time - current_trade['entry_time']
        
        trades.append({
            'entry_time': current_trade['entry_time'],
            'exit_time': last_time,
            'return_pct': (last_price - current_trade['entry_price']) / current_trade['entry_price'] * 100,
            'profit': profit,
            'duration_hours': duration.total_seconds() / 3600,
            'status': 'Force Closed'
        })
        print(f"⚠️ Force closed active trade at end of data. Profit: {profit:.2f}")

    # Analysis
    initial_balance = 5000.0 # Standardize
    final_balance = env.equity
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    
    # Drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Trade Stats
    if trades:
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        win_rate = (len(trades_df[trades_df['profit'] > 0]) / total_trades) * 100
        avg_holding_time = trades_df['duration_hours'].mean()
        max_holding_time = trades_df['duration_hours'].max()
        best_trade = trades_df['profit'].max()
        worst_trade = trades_df['profit'].min()
    else:
        total_trades = 0
        win_rate = 0
        avg_holding_time = 0
        max_holding_time = 0
        best_trade = 0
        worst_trade = 0
    
    print("\n" + "="*40)
    print(f"📊 NEUROTRADER 2.0 RESULT (XAUUSD D1)")
    print("="*40)
    print(f"💰 Final Equity:     ${final_balance:,.2f} ({total_return:+.2f}%)")
    print(f"📉 Max Drawdown:     {max_drawdown:.2f}%")
    print("-" * 20)
    print(f"🔢 Total Orders:     {total_trades}")
    print(f"✅ Win Rate:         {win_rate:.2f}%")
    print(f"⏱️ Avg Holding Time: {avg_holding_time:.1f} hours ({avg_holding_time/24:.1f} days)")
    print(f"⏳ Max Holding Time: {max_holding_time:.1f} hours ({max_holding_time/24:.1f} days)")
    print(f"🚀 Best Trade:       ${best_trade:.2f}")
    print(f"🔻 Worst Trade:      ${worst_trade:.2f}")
    
    # Save Chart
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label='NeuroTrader 2.0 (LSTM)')
    plt.title('NeuroTrader 2.0: XAUUSD D1 Backtest')
    plt.xlabel('Days')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    
    chart_path = os.path.join(ROOT_DIR, "analysis/v2_backtest_chart.png")
    plt.savefig(chart_path)
    print(f"📉 Check chart: {chart_path}")
    print("="*40)

if __name__ == "__main__":
    backtest_v2()
