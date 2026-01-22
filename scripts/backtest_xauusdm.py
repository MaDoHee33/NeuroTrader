
import argparse
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.brain.env.trading_env import TradingEnv

def calculate_metrics(df_res):
    """Calculate trading metrics from results dataframe"""
    initial_balance = df_res['equity'].iloc[0]
    final_balance = df_res['equity'].iloc[-1]
    
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    # Drawdown
    df_res['peak'] = df_res['equity'].cummax()
    df_res['drawdown'] = (df_res['equity'] - df_res['peak']) / df_res['peak'] * 100
    max_drawdown = df_res['drawdown'].min()
    
    # Win Rate (approximate based on trade log if available, or positive returns)
    # We'll rely on the environment's trade history if we can access it, 
    # but for now let's just use daily/step returns for Sharpe
    df_res['returns'] = df_res['equity'].pct_change().dropna()
    mean_ret = df_res['returns'].mean()
    std_ret = df_res['returns'].std()
    
    # Annualized Sharpe (assuming M5 data, 288 steps/day, 252 days)
    # steps_per_year = 288 * 252 
    # Actually let's just verify based on data duration.
    
    if std_ret == 0:
        sharpe = 0
    else:
        sharpe = mean_ret / std_ret * np.sqrt(288 * 252) # Proxy
        
    return {
        'Initial Equity': initial_balance,
        'Final Equity': final_balance,
        'Total Return (%)': total_return,
        'Max Drawdown (%)': max_drawdown,
        'Sharpe Ratio': sharpe
    }

def main():
    parser = argparse.ArgumentParser(description='Backtest PPO on XAUUSDm')
    parser.add_argument('--data-file', type=str, default='XAUUSDm_M5_processed.parquet')
    parser.add_argument('--model-path', type=str, default='models/ppo_xauusdm_final.zip')
    args = parser.parse_args()

    data_path = ROOT_DIR / 'data' / 'processed' / args.data_file
    model_path = ROOT_DIR / args.model_path
    
    print(f"🚀 Starting Backtest...")
    print(f"📊 Data: {data_path.name}")
    print(f"🧠 Model: {model_path.name}")

    if not data_path.exists() or not model_path.exists():
        print("❌ Data or Model not found.")
        return

    # Load Data
    df = pd.read_parquet(data_path)
    
    # Create Env
    env = TradingEnv(df)
    
    # Load Model
    model = PPO.load(model_path)
    
    # Run Backtest
    obs, _ = env.reset()
    done = False
    
    history = []
    
    print("🏃 Running simulation...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        step_data = {
            'step': info['step'],
            'equity': info['equity'],
            'price': df.iloc[info['step']]['close'], # Approx
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
    print("📊 BACKTEST RESULTS")
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
        
        plot_path = ROOT_DIR / 'analysis' / 'backtest_result.png'
        # ensure analysis dir exists (it does)
        plt.savefig(plot_path)
        print(f"\n📈 Chart saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️  Could not save chart: {e}")

    # Save CSV
    csv_path = ROOT_DIR / 'analysis' / 'backtest_data.csv'
    df_res.to_csv(csv_path, index=False)
    print(f"💾 Data saved to: {csv_path}")

if __name__ == "__main__":
    main()
