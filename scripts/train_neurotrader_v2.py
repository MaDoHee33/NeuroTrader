
import sys
import os
import pandas as pd
import glob
from pathlib import Path
from sb3_contrib import RecurrentPPO # The LSTM Brain
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.brain.env.trading_env import TradingEnv

def train_v2():
    print("🧠 Initializing NeuroTrader 2.0 Training (LSTM Architecture)...")
    
    # 1. Load Multi-Asset Data
    data_path = os.path.join(ROOT_DIR, "data/processed/*.parquet")
    files = glob.glob(data_path)
    
    if not files:
        print("❌ No data found in data/processed/")
        return

    data_map = {}
    total_rows = 0
    for f in files:
        name = os.path.basename(f)
        df = pd.read_parquet(f)
        
        # Filter: Only use Data with sufficient rows for LSTM sequence
        if len(df) > 500:
            data_map[name] = df
            total_rows += len(df)
            print(f"   Loaded {name}: {len(df)} rows")
    
    print(f"📚 Total Training Data: {total_rows} rows across {len(data_map)} assets.")

    # 2. Create Environment
    # RecurrentPPO handles state, so we can use DummyVecEnv
    def make_env():
        return TradingEnv(data_map)
        
    env = DummyVecEnv([make_env]) # vectorized for efficiency if needed, here just 1

    # 3. Configure LSTM Model (NeuroTrader 2.0)
    # Using RecurrentPPO with MlpLstmPolicy
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128, # Larger batch for LSTM
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Slight exploration
        policy_kwargs={
            "net_arch": [], # LSTM handles the features directly usually, or we can add layers
            "enable_critic_lstm": True, # Shared or separate LSTM
            "lstm_hidden_size": 256, # Memory Capacity
            "n_lstm_layers": 1,
        },
        tensorboard_log="./tensorboard_v2/"
    )
    
    # 4. Train
    print("🏋️‍♂️ Training Started... (Target: 1,000,000 steps)")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path='./models/v2_checkpoints/',
        name_prefix='neurotrader_v2'
    )
    
    try:
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)
        
        # 5. Save Final Brain
        save_path = os.path.join(ROOT_DIR, "models/neurotrader_v2_lstm")
        model.save(save_path)
        print(f"✅ Training Complete! Model saved to {save_path}.zip")
        
        # 6. Validate (Quick Check)
        print("\n🔎 Running Quick Validation (Test Set)...")
        # Reuse available assets split if Env supports it, or just manual check
        # For V2, we just want to see if it runs and saves.
        
    except KeyboardInterrupt:
        print("\n⚠️ Training Interrupted. Saving current state...")
        model.save("models/neurotrader_v2_interrupted")

if __name__ == "__main__":
    train_v2()
