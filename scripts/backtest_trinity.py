
import argparse
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import brain modules
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO 
from src.brain.env.trading_env import TradingEnv
from src.brain.feature_eng import add_features
from src.analysis.behavior import calculate_behavioral_metrics, generate_text_report
from src.skills.reporter import TraderReporter

def main():
    parser = argparse.ArgumentParser(description='Trinity System Backtester')
    parser.add_argument('--role', type=str, required=True, choices=['scalper', 'swing', 'trend'], help="Agent role")
    parser.add_argument('--data', type=str, required=True, help="Path to data file (parquet)")
    parser.add_argument('--model', type=str, required=True, help="Path to model file (.zip)")
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'train', 'all'], help='Data split to usage')
    args = parser.parse_args()
    
    # Paths
    print(f"\n🚀 TRINITY BACKTEST SYSTEM")
    print(f"Role: {args.role.upper()}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    
    # 1. Load Data
    df = pd.read_parquet(args.data)
    
    # Fix Types
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    # Split
    split_idx = int(len(df) * 0.8)
    if args.mode == 'test':
        df = df.iloc[split_idx:]
        print("Dataset: TEST (20% Out-of-Sample)")
    elif args.mode == 'train':
        df = df.iloc[:split_idx]
        print("Dataset: TRAIN (80% In-Sample)")
        
    print(f"Rows: {len(df)}")
    
    # 2. Features
    try:
        # Check if features need adding (if raw data)
        # If 'ema_9' not in columns, add features
        if 'ema_9' not in df.columns:
            print("Adding features...")
            # Ensure index
            if 'time' in df.columns:
                df.set_index('time', drop=False, inplace=True)
            df.sort_index(inplace=True)
            df = add_features(df)
            df.dropna(inplace=True)
    except Exception as e:
        print(f"Feature Error: {e}")
        return

    # 3. Environment
    # We use agent_type to ensure reward calculation matches (though irrelevant for Backtesting prediction, good for consistency)
    env = TradingEnv(df, agent_type=args.role)
    
    # 4. Load Model
    print("Loading Model...")
    try:
        model = RecurrentPPO.load(args.model, env=env)
    except:
        print("Warning: RecurrentPPO load failed, trying PPO...")
        model = PPO.load(args.model, env=env)
        
    # 5. Run Simulation
    print("running simulation...")
    obs, _ = env.reset()
    done = False
    history = []
    lstm_states = None
    
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # Log Step
        log = {
            'step': info['step'],
            'price': df.iloc[info['step']]['close'] if info['step'] < len(df) else 0,
            'equity': info['equity'],
            'position': env.position,
            'balance': info.get('balance', 0)
        }
        history.append(log)
        
        if done or truncated:
            break
            
    # 6. Analysis
    df_res = pd.DataFrame(history)
    
    # PnL Metrics
    initial_equity = df_res['equity'].iloc[0]
    final_equity = df_res['equity'].iloc[-1]
    ret_pct = (final_equity - initial_equity) / initial_equity * 100
    
    peak = df_res['equity'].cummax()
    dd = (df_res['equity'] - peak) / peak * 100
    max_dd = dd.min()
    
    print("\n" + "="*30)
    print("💰 FINANCIAL PERFORMANCE")
    print(f"Return: {ret_pct:.2f}%")
    print(f"Max DD: {max_dd:.2f}%")
    print("="*30)
    
    # Behavioral Metrics
    beh_metrics = calculate_behavioral_metrics(df_res)
    beh_report = generate_text_report(beh_metrics)
    print(beh_report)
    
    # 7. Save Artifacts
    os.makedirs('reports', exist_ok=True)
    os.makedirs('analysis', exist_ok=True)
    
    # Save CSV
    csv_name = f"analysis/backtest_{args.role}_{os.path.basename(args.data).split('.')[0]}.csv"
    df_res.to_csv(csv_name, index=False)
    
    # Generate Report using Skill
    reporter = TraderReporter(report_dir='reports')
    report_path = reporter.generate_report(df_res, model_name=os.path.basename(args.model), role=args.role)
        
    print(f"📝 Report saved to: {report_path}")
    print(f"💾 Data saved to: {csv_name}")

if __name__ == "__main__":
    main()
