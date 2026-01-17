
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO  # [NEW] LSTM supported PPO
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
DATA_DIR_DRIVE = "/content/drive/MyDrive/NeuroTrader_Workspace/data"
DATA_DIR_LOCAL = str(ROOT_DIR / "data" / "processed")

LOG_DIR = "/content/drive/MyDrive/NeuroTrader_Workspace/logs"
MODEL_DIR = "/content/drive/MyDrive/NeuroTrader_Workspace/models"

def find_data_file():
    files = glob.glob(os.path.join(DATA_DIR_DRIVE, "*.parquet"))
    if not files:
        files = glob.glob(os.path.join(DATA_DIR_LOCAL, "*.parquet"))
    if not files:
        files = glob.glob("*.parquet")
    return files[0] if files else None

def main():
    print("🚀 Starting NeuroTrader Recurrent PPO (LSTM) Training...")
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    data_file = find_data_file()
    if not data_file:
        print("❌ No Parquet data found!")
        return
        
    print(f"📂 Loading Data: {data_file}")
    df = pd.read_parquet(data_file)
    print(f"✅ Data Loaded: {len(df):,} rows")
    
    # [NEW] Verify DataFrame has new features
    required_cols = ['atr', 'log_ret_lag_1', 'log_ret_lag_5']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Data missing Level 2 features: {missing}")
        print("👉 Please run 'python tools/process_data.py' locally and re-upload data!")
        return

    # Create Env
    env = TradingEnv(df)
    env = Monitor(env) 
    
    # Define Model (LSTM)
    print("🧠 Initializing RecurrentPPO (LSTM) Agent...")
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Save more often as LSTM is harder to train
        save_path=MODEL_DIR,
        name_prefix="neurotrader_lstm"
    )
    
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64, # Batch size for PPO
        gamma=0.99,
        policy_kwargs={
            "enable_critic_lstm": False, 
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1
        }
    )
    
    print("🏃‍♂️ Training started... (Monitor logs in TensorBoard)")
    try:
        model.learn(
            total_timesteps=5_000_000, # Increased for LSTM
            progress_bar=True,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training Interrupted! Saving current model...")
        
    final_path = os.path.join(MODEL_DIR, "neurotrader_brain_lstm_final")
    model.save(final_path)
    print(f"✨ Training Complete. Model saved to: {final_path}.zip")

if __name__ == "__main__":
    main()
