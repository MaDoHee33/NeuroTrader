
import os
import sys
import glob
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO # For Level 2+

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from brain.env.trading_env import TradingEnv

def calculate_max_drawdown(equity_curve):
    """Calculates the maximum drawdown percentage."""
    peak = equity_curve[0]
    max_dd = 0
    
    for x in equity_curve:
        if x > peak: 
            peak = x
        dd = (peak - x) / peak
        if dd > max_dd:
            max_dd = dd
            
    return max_dd * 100

def backtest(model_path, data_path, model_type="mlp"):
    print(f"📉 Loading Data: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"✅ Data loaded: {len(df):,} rows")

    # Verify columns (similar to train script)
    # This ensures we don't crash if using a Level 2 model on Level 1 data
    feature_cols = TradingEnv(df).feature_cols # Get expected cols from Env
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"❌ Error: Dataset missing features required by Env: {missing}")
        return

    # Create Env
    env = TradingEnv(df)
    obs, _ = env.reset()

    # Load Model
    print(f"🧠 Loading Model: {model_path} ({model_type})")
    if model_type.lower() == "lstm":
        model = RecurrentPPO.load(model_path)
        lstm_states = None # Init LSTM state
        dones = np.ones(1) # Start of episode
    else:
        model = PPO.load(model_path)

    # Tracking
    equity_curve = [env.balance]
    start_balance = env.balance
    
    print("🏃‍♂️ Running Backtest...")
    
    # Loop
    done = False
    step_count = 0
    
    while not done:
        # Predict
        if model_type.lower() == "lstm":
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=dones)
            dones = np.array([False]) # Not done yet
        else:
            action, _ = model.predict(obs)

        # Step
        obs, reward, done, truncated, info = env.step(action)
        
        # Track Equity
        equity_curve.append(info['equity'])
        step_count += 1
        
        if step_count % 10000 == 0:
            print(f"   Step {step_count}: Equity = {info['equity']:.2f}", end='\r')

    # Metrics
    final_balance = equity_curve[-1]
    profit_pct = ((final_balance - start_balance) / start_balance) * 100
    max_dd = calculate_max_drawdown(equity_curve)
    
    print("\n" + "="*40)
    print(f"📊 BACKTEST RESULTS: {Path(model_path).name}")
    print("="*40)
    print(f"💰 Initial Balance: ${start_balance:,.2f}")
    print(f"💸 Final Balance:   ${final_balance:,.2f}")
    print(f"📈 Total Return:    {profit_pct:+.2f}%")
    print(f"📉 Max Drawdown:    {max_dd:.2f}%")
    print(f"⏳ Total Steps:     {step_count}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model.zip")
    parser.add_argument("--data", type=str, required=True, help="Path to .parquet data file")
    parser.add_argument("--type", type=str, default="mlp", choices=["mlp", "lstm"], help="Model type: mlp or lstm")
    
    args = parser.parse_args()
    
    backtest(args.model, args.data, args.type)
