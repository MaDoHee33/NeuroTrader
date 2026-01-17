
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import glob
import sys
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from brain.env.trading_env import TradingEnv

# Paths
# In Colab, we expect data in /content/drive/MyDrive/NeuroTrader_Workspace/data
# OR uploaded locally
DATA_DIR_DRIVE = "/content/drive/MyDrive/NeuroTrader_Workspace/data"
DATA_DIR_LOCAL = str(ROOT_DIR / "data" / "processed")

LOG_DIR = "/content/drive/MyDrive/NeuroTrader_Workspace/logs"
MODEL_DIR = "/content/drive/MyDrive/NeuroTrader_Workspace/models"

def find_data_file():
    # Check Drive
    files = glob.glob(os.path.join(DATA_DIR_DRIVE, "*.parquet"))
    if not files:
        # Check Local
        files = glob.glob(os.path.join(DATA_DIR_LOCAL, "*.parquet"))
        
    if not files:
        # Check current dir
        files = glob.glob("*.parquet")
        
    return files[0] if files else None

def main():
    print("🚀 Starting NeuroTrader PPO Training...")
    
    # 1. Setup Directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 2. Load Data
    data_file = find_data_file()
    if not data_file:
        print("❌ No Parquet data found! Please upload data to Drive or local folder.")
        return
        
    print(f"📂 Loading Data: {data_file}")
    df = pd.read_parquet(data_file)
    print(f"✅ Data Loaded: {len(df):,} rows")
    
    # 3. Create Environment
    env = TradingEnv(df)
    env = Monitor(env) 
    
    # 4. Define Model
    print("🧠 Initializing PPO Agent...")
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=MODEL_DIR,
        name_prefix="neurotrader_ppo"
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )
    
    # 5. Train
    print("🏃‍♂️ Training started... (Monitor logs in TensorBoard)")
    try:
        model.learn(
            total_timesteps=3_000_000, 
            progress_bar=True,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training Interrupted! Saving current model...")
        
    # 6. Save Final
    final_path = os.path.join(MODEL_DIR, "neurotrader_brain_final")
    model.save(final_path)
    print(f"✨ Training Complete. Model saved to: {final_path}.zip")

if __name__ == "__main__":
    main()
