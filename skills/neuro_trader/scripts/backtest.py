
import os
import sys
import glob
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib
# Force Agg backend to prevent Colab display errors
matplotlib.use('Agg')
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO # For Level 2+

# Add src to path
# Path: skills/neuro_trader/scripts/backtest.py -> Root is ../../../
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
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
    parser.add_argument("--model", type=str, help="Path to trained model.zip")
    parser.add_argument("--data", type=str, help="Path to .parquet data file")
    parser.add_argument("--level", type=int, help="Level (1=MLP, 2=LSTM) - Auto-resolves model/data paths")
    parser.add_argument("--type", type=str, default="mlp", choices=["mlp", "lstm"], help="Model type override")
    
    args = parser.parse_args()
    
    # Auto-resolve paths if Level is provided
    if args.level:
        base_dir = "/content/drive/MyDrive/NeuroTrader_Workspace"
        if not os.path.exists(base_dir): # If local
             base_dir = str(ROOT_DIR)
             
        # Resolve Model
        if not args.model:
            model_type_str = "LSTM" if args.level == 2 else "MLP"
            # Try to find final model
            sub_dir = f"L{args.level}_{model_type_str}"
            model_path = os.path.join(base_dir, "models", sub_dir, "final_model.zip")
            if not os.path.exists(model_path):
                 print(f"⚠️  Could not find auto-resolved model: {model_path}")
                 # Fallback to searching
                 import glob
                 candidates = glob.glob(os.path.join(base_dir, "models", sub_dir, "*.zip"))
                 if candidates:
                     model_path = candidates[-1] # Take latest
            args.model = model_path
            
        # Resolve Data
        if not args.data:
            pattern = f"*_L{args.level}.parquet"
            # Search logic similar to train_ppo
            search_dirs = [
                os.path.join(base_dir, "data"), 
                os.path.join(ROOT_DIR, "data", "processed"),
                "."
            ]
            for d in search_dirs:
                matches = glob.glob(os.path.join(d, pattern))
                if matches:
                    args.data = matches[0]
                    break
        
        # Resolve Type
        if args.level == 2:
            args.type = "lstm"
    
    if not args.model or not args.data:
        print("❌ Error: Must provide --model and --data OR --level")
        sys.exit(1)

    backtest(args.model, args.data, args.type)
