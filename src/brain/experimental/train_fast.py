#!/usr/bin/env python3
"""
Optimized Parallel Training Script
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# Fix Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.brain.env.trading_env import TradingEnv
from src.brain.features import add_features  # Use new consolidated features
from src.utils.config_loader import config
from nautilus_trader.model.data import BarType

# CONFIG
N_ENVS = config.get("training.parallel_envs", 4)
TOTAL_TIMESTEPS = 10_000 # Force short run for verification test
DATA_PATH = BASE_DIR / config.get("paths.data", 'data/nautilus_store')
BAR_TYPE = config.get("environment.bar_type", 'XAUUSD.SIM-5-MINUTE-LAST-EXTERNAL')

def load_data():
    """Load and prep data once"""
    try:
        # Force synthetic data for testing parallelization logic first
        print("[Info] Using synthetic data for system connectivity test")
        return generate_synthetic_data()
        
        # catalog = ParquetDataCatalog(str(DATA_PATH))
        bars = list(catalog.bars(bar_types=[BAR_TYPE]))
        if not bars:
             # Fallback for testing if no data found
             print("[Warning] No Nautlius data found. Generating synthetic data for test.")
             return generate_synthetic_data()
             
        data = []
        for b in bars:
            data.append({
                'timestamp': b.ts_init,
                'open': b.open.as_double(),
                'high': b.high.as_double(),
                'low': b.low.as_double(),
                'close': b.close.as_double(),
                'volume': b.volume.as_double()
            })
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return add_features(df)
    except Exception as e:
        print(f"Error loading data: {e}")
        return generate_synthetic_data()

def generate_synthetic_data():
    """Generate dummy data for connectivity testing"""
    dates = pd.date_range(start='2024-01-01', periods=2000, freq='5min')
    df = pd.DataFrame(index=dates)
    df['close'] = 2000.0 + np.random.randn(2000).cumsum()
    df['open'] = df['close'] + np.random.randn(2000)
    df['high'] = df[['open', 'close']].max(axis=1) + 1.0
    df['low'] = df[['open', 'close']].min(axis=1) - 1.0
    df['volume'] = np.abs(np.random.randn(2000)) * 1000
    return add_features(df)

# Global data variable for forked processes
global_df = None

def make_env():
    """Utility function for multiprocess env"""
    # Each env needs its own instance, but can share read-only df
    return TradingEnv(global_df)

def main():
    global global_df
    
    print(f"{'='*50}")
    print(f"[FAST TRAINER PARALLEL SYSTEM (Experimental)]")
    print(f"   - Cores: {N_ENVS}")
    print(f"   - Bar Type: {BAR_TYPE}")
    print(f"{'='*50}\n")
    
    # 1. Load Data
    print("[Loading data...]")
    global_df = load_data()
    print(f"[Data Shape: {global_df.shape}]")
    
    # 2. Vectorized Environment
    print(f"[Creating {N_ENVS} Parallel Environments... (Debugging Mode: DummyVecEnv)]")
    # Use DummyVecEnv for debugging (single process but multiple envs)
    vec_env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    
    # 3. Model Setup (Optimized Hyperparams)
    print("[Initializing PPO (Optimized)...]")
    
    hyperparams = config.get("training.hyperparameters")
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=hyperparams.get("learning_rate", 3e-4),
        n_steps=hyperparams.get("n_steps", 2048) // N_ENVS,
        batch_size=hyperparams.get("batch_size", 64),
        gamma=hyperparams.get("gamma", 0.99),
        verbose=1,
        tensorboard_log=str(BASE_DIR / 'logs' / 'fast_train')
    )
    
    # 4. Training
    print("[Starting Training Loop...]")
    start = time.time()
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    except KeyboardInterrupt:
        print("\n[Training interrupted by user.]")
    final_end = time.time()
    
    # 5. Save
    save_path = BASE_DIR / 'models' / 'experimental' / 'fast_ppo_final'
    model.save(save_path)
    
    print(f"\n[DONE! Saved to {save_path}]")
    print(f"[Total Time: {(final_end - start)/60:.2f} min]")
    print(f"[Speed: {TOTAL_TIMESTEPS / (final_end - start):.0f} fps]")

if __name__ == "__main__":
    # Fix for Windows Multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()
