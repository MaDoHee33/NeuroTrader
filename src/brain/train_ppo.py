
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib
# Force Agg backend for Colab/Headless to prevent display errors
matplotlib.use('Agg')
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import glob
import sys
import argparse
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from brain.env.trading_env import TradingEnv

# Paths
# Paths
DATA_DIR_DRIVE = "/content/drive/MyDrive/NeuroTrader_Workspace/data"
DATA_DIR_LOCAL = str(ROOT_DIR / "data" / "processed")

if os.path.exists("/content/drive/MyDrive"):
    WORKSPACE_DRIVE = "/content/drive/MyDrive/NeuroTrader_Workspace"
else:
    # Use local workspace if not in Colab
    WORKSPACE_DRIVE = str(ROOT_DIR / "workspace")
    os.makedirs(WORKSPACE_DRIVE, exist_ok=True)

def find_data_file(level=1):
    # Look for files with _L{level}.parquet suffix
    pattern = f"*_L{level}.parquet"
    if level == 3:
        pattern = "*_News.parquet"
    
    # Priority: Drive -> Local
    files = glob.glob(os.path.join(DATA_DIR_DRIVE, pattern))
    if not files:
        files = glob.glob(os.path.join(DATA_DIR_LOCAL, pattern))
    if not files:
        files = glob.glob(pattern) # CWD
        
    return files[0] if files else None

def find_latest_checkpoint(model_dir):
    """Finds the latest checkpoint in the model directory."""
    checkpoints = glob.glob(os.path.join(model_dir, "*.zip"))
    if not checkpoints:
        return None
    # Sort by modification time
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Train PPO/RecurrentPPO Agent")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3], help="Training Level (1=MLP, 2=LSTM, 3=Hybrid)")
    args = parser.parse_args()
    
    level = args.level
    model_type = "MLP" if level == 1 else ("LSTM" if level == 2 else "Hybrid")
    
    # Namespaced Directories
    # e.g. models/L1_MLP/, logs/L1_MLP/
    sub_dir = f"L{level}_{model_type}"
    
    log_dir = os.path.join(WORKSPACE_DRIVE, "logs", sub_dir)
    model_dir = os.path.join(WORKSPACE_DRIVE, "models", sub_dir)
    
    print(f"🚀 Starting NeuroTrader Training (Level {level}: {model_type})...")
    print(f"📂 Directories: \n  Logs: {log_dir}\n  Models: {model_dir}")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Load Data
    data_file = find_data_file(level)
    if not data_file:
        print(f"❌ No Parquet data found for Level {level} (*_L{level}.parquet)!")
        print(f"👉 Please run 'python tools/process_data.py --level {level}' first.")
        return
        
    print(f"📂 Loading Data: {data_file}")
    df = pd.read_parquet(data_file)
    print(f"✅ Data Loaded: {len(df):,} rows")
    
    # 2. Verify Features (Level 2+)
    if level >= 2:
        required_cols = ['atr', 'log_ret_lag_1', 'log_ret_lag_5']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"❌ Data missing Level 2 features: {missing}")
            return

    # 3. Create Env
    env = TradingEnv(df)
    env = Monitor(env) 
    
    # 4. Define Agent (Load or Create)
    latest_ckpt = find_latest_checkpoint(model_dir)
    
    if latest_ckpt:
        print(f"🔄 Resuming from Checkpoint: {latest_ckpt}")
        if level == 1:
            model = PPO.load(latest_ckpt, env=env, tensorboard_log=log_dir)
        else:
            model = RecurrentPPO.load(latest_ckpt, env=env, tensorboard_log=log_dir)
    else:
        print(f"🧠 Initializing New {model_type} Agent...")
        if level == 1:
            model = PPO(
                "MlpPolicy", 
                env, 
                verbose=1, 
                tensorboard_log=log_dir,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                ent_coef=0.01 # Force exploration to prevent "Frozen Agent"
            )
        else:
            model = RecurrentPPO(
                "MlpLstmPolicy", 
                env, 
                verbose=1, 
                tensorboard_log=log_dir,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                ent_coef=0.01, # Force exploration
                policy_kwargs={
                    "enable_critic_lstm": False, 
                    "lstm_hidden_size": 256,
                    "n_lstm_layers": 1
                }
            )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=model_dir,
        name_prefix=f"neurotrader_L{level}"
    )

    total_timesteps = 3_000_000 if level == 1 else 5_000_000
    if level == 3:
        total_timesteps = 6_000_000 # More steps for hybrid learning
    
    # 5. Train
    print(f"🏃‍♂️ Training started... (Steps: {total_timesteps})")
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            progress_bar=True,
            callback=checkpoint_callback,
            reset_num_timesteps=False # Crucial for resuming correct tensorboard logging
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training Interrupted! Saving current model...")
        
    # 6. Save Final
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"✨ Training Complete. Model saved to: {final_path}.zip")

if __name__ == "__main__":
    main()
