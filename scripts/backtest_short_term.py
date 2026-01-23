
import argparse
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from sb3_contrib import RecurrentPPO # Need RecurrentPPO for loading
from stable_baselines3 import PPO

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.brain.env.trading_env import TradingEnv
from src.brain.feature_eng import add_features

def calculate_metrics(df_res):
    """Calculate trading metrics from results dataframe"""
    initial_balance = df_res['equity'].iloc[0]
    final_balance = df_res['equity'].iloc[-1]
    
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    # Drawdown
    df_res['peak'] = df_res['equity'].cummax()
    df_res['drawdown'] = (df_res['equity'] - df_res['peak']) / df_res['peak'] * 100
    max_drawdown = df_res['drawdown'].min()
    
    df_res['returns'] = df_res['equity'].pct_change().dropna()
    mean_ret = df_res['returns'].mean()
    std_ret = df_res['returns'].std()
    
    # Annualized Sharpe (M5 = 288 steps/day)
    if std_ret == 0:
        sharpe = 0
    else:
        sharpe = mean_ret / std_ret * np.sqrt(288 * 252)
        
    return {
        'Initial Equity': initial_balance,
        'Final Equity': final_balance,
        'Total Return (%)': total_return,
        'Max Drawdown (%)': max_drawdown,
        'Sharpe Ratio': sharpe
    }

def main():
    parser = argparse.ArgumentParser(description='Backtest Short-Term Model (M5)')
    parser.add_argument('--data-file', type=str, default='XAUUSD_M5_processed.parquet')
    parser.add_argument('--model-path', type=str, default='models/neurotrader_short_term.zip')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'train', 'all'], help='Data split to use')
    args = parser.parse_args()

    data_path = ROOT_DIR / 'data' / 'processed' / args.data_file
    model_path = ROOT_DIR / args.model_path
    
    print(f"🚀 Starting Short-Term Backtest...")
    print(f"📊 Data: {data_path.name}")
    print(f"🧠 Model: {model_path.name}")

    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return
        
    if not model_path.exists():
        print(f"⚠️ Model not found at {model_path}")
        print("   If you haven't trained yet, run: python scripts/train_short_term.py")
        return

    # Load Data
    df = pd.read_parquet(data_path)
    
    # --- TRAIN/TEST SPLIT ---
    split_idx = int(len(df) * 0.8)
    
    if args.mode == 'test':
        print(f"🔹 Mode: TEST (Using last 20% of data)")
        df = df.iloc[split_idx:]
    elif args.mode == 'train':
        print(f"🔹 Mode: TRAIN (Using first 80% of data - In-Sample)")
        df = df.iloc[:split_idx]
    else:
        print(f"🔹 Mode: ALL (Using full dataset)")
    
    # Log range
    start_date = df.iloc[0]['time'] if 'time' in df.columns else df.index[0]
    end_date = df.iloc[-1]['time'] if 'time' in df.columns else df.index[-1]
    print(f"📅 Date Range: {start_date} - {end_date} ({len(df)} rows)")

    # Add Features
    try:
         df = add_features(df)
    except Exception as e:
         print(f"❌ Feature Engineering failed: {e}")
         return
    
    # Define Short-Term Features (Must match training!)
    SHORT_TEAM_FEATURES = [
        'body_size', 'upper_wick', 'lower_wick', 'is_bullish',
        'dist_ema_9', 'dist_ema_21',     # Fast Trend
        'dist_ema_50',                   # Mid Trend Filter
        'rsi', 'atr_norm',
        'macd_norm', 'macd_signal_norm', 'macd_diff_norm', # Momentum
        'dist_bb_high', 'dist_bb_low', 'bb_width',         # Mean Reversion / Volatility
        'stoch_k', 'stoch_d',                              # Overbought/Oversold
        'log_ret', 'log_ret_lag_1',
        'hour_sin', 'hour_cos'
    ]
    
    # Create Env
    env = TradingEnv(df, feature_cols=SHORT_TEAM_FEATURES)
    
    # Load Model
    # Important: Use RecurrentPPO.load for LSTM models
    try:
        model = RecurrentPPO.load(model_path)
    except:
        print("⚠️ Failed to load as RecurrentPPO, trying standard PPO...")
        model = PPO.load(model_path)
    
    # Run Backtest
    obs, _ = env.reset()
    done = False
    
    history = []
    lstm_states = None
    # For RecurrentPPO, we need to manage states if doing step-by-step manually without VecEnv
    # But model.predict handles it if we pass state?
    # Actually model.predict returns ACTION, STATE.
    # Initial state is None
    
    print("🏃 Running simulation...")
    num_dones = 0
    
    while not done:
        # Predict
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        
        step_data = {
            'step': info['step'],
            'equity': info['equity'],
            'price': df.iloc[info['step']]['close'],
            'balance': info.get('balance', 0),
            'position': env.position
        }
        if 'last_trade' in info:
            step_data['trade'] = info['last_trade']['action']
            
        history.append(step_data)
        
        if done or truncated:
            break
            
    # Process Results
    df_res = pd.DataFrame(history)
    metrics = calculate_metrics(df_res)
    
    # Print Metrics
    print("\n" + "="*40)
    print("📊 BACKTEST RESULTS (Short-Term)")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.2f}")
    print("="*40)
    
    # Save Plot
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df_res['equity'], label='Equity')
        plt.title(f'Backtest Equity Curve - {args.data_file}')
        plt.xlabel('Steps')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        
        plot_path = ROOT_DIR / 'analysis' / 'backtest_short_term_result.png'
        plt.savefig(plot_path)
        print(f"\n📈 Chart saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️  Could not save chart: {e}")

    # Save CSV
    csv_path = ROOT_DIR / 'analysis' / 'backtest_short_term_data.csv'
    df_res.to_csv(csv_path, index=False)
    print(f"💾 Data saved to: {csv_path}")

if __name__ == "__main__":
    main()
