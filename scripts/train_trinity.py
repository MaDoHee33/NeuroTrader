
import argparse
import os
import glob
import sys
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from src.brain.env.trading_env import TradingEnv
from src.brain.feature_eng import add_features

def load_data(data_path):
    print(f"Loading data from {data_path}...")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported format")
        
    print(f"Initial shape: {df.shape}")
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', drop=False, inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', drop=False, inplace=True)
        
    df.sort_index(inplace=True)
    print(f"Shape after index: {df.shape}")
    
    try:
        df = add_features(df)
        print(f"Shape after features: {df.shape}")
    except Exception as e:
        print(f"Features Error: {e}")
        
    before_drop = len(df)
    df.dropna(inplace=True)
    after_drop = len(df)
    print(f"Dropped {before_drop - after_drop} rows. Final shape: {df.shape}")
    
    if len(df) == 0:
        print("DEBUG: Columns with NaNs:")
        print(df.isnull().sum())
        
    return df

def train_trinity(role, data_path, total_timesteps=1000000):
    role = role.lower()
    print(f"\n--- Initiating Trinity Training Protocol ---")
    print(f"Role: {role.upper()}")
    print(f"Data: {data_path}")
    
    # 1. Configuration per Role
    # Defaults
    if role == 'scalper':
        n_steps = 256
        batch_size = 64
        gamma = 0.85 
        learning_rate = 3e-4
        ent_coef = 0.01
    elif role == 'swing':
        n_steps = 1024
        batch_size = 128
        gamma = 0.95 
        learning_rate = 2e-4
        ent_coef = 0.005
    else: # trend
        n_steps = 2048
        batch_size = 256
        gamma = 0.999 
        learning_rate = 1e-4
        ent_coef = 0.001
        
    # Check for Optimized Params
    param_file = f"best_params_{role}.json"
    if os.path.exists(param_file):
        print(f"✨ Loading Optimized Hyperparameters from {param_file}")
        import json
        with open(param_file, 'r') as f:
            params = json.load(f)
            n_steps = params.get('n_steps', n_steps)
            batch_size = params.get('batch_size', batch_size)
            gamma = params.get('gamma', gamma)
            learning_rate = params.get('learning_rate', learning_rate)
            ent_coef = params.get('ent_coef', ent_coef)
            print(f"   -> Gamma: {gamma:.4f}, LR: {learning_rate:.6f}, Batch: {batch_size}")
        
    # 2. Load Data
    df = load_data(data_path)
    
    # 3. Split Data (Train/Test)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    print(f"Training Samples: {len(train_df)}")
    
    # 4. Create Environment
    # We pass the 'agent_type' (role) to the environment so it picks the right reward function
    env = DummyVecEnv([lambda: TradingEnv(train_df, agent_type=role)])
    
    # 5. Model Setup (RecurrentPPO)
    tensorboard_log = f"./logs/trinity/{role}"
    os.makedirs(tensorboard_log, exist_ok=True)
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=0.95,
        ent_coef=ent_coef,
        tensorboard_log=tensorboard_log
    )
    
    # 6. Train
    model_name = f"models/trinity_{role}_{os.path.basename(data_path).split('.')[0]}"
    print(f"Training Model: {model_name}")
    
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        model.save(model_name)
        print(f"✅ Model saved to {model_name}")
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train specific Trinity agent roles")
    parser.add_argument('--role', type=str, required=True, choices=['scalper', 'swing', 'trend'], help="Agent role (scalper/swing/trend)")
    parser.add_argument('--data', type=str, required=True, help="Path to training data")
    parser.add_argument('--steps', type=int, default=1000000, help="Total timesteps")
    
    args = parser.parse_args()
    
    if args.role == 'scalper' and 'M5' not in args.data and 'M1' not in args.data:
        print("⚠️ WARNING: Scalpers should preferably trade on M1/M5 data.")
        
    train_trinity(args.role, args.data, args.steps)
