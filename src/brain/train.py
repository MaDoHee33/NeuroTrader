
import os
import pandas as pd
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from src.brain.env.trading_env import TradingEnv
from src.brain.feature_eng import add_features

# Configuration
DATA_DIR = "data/nautilus_store/data/bar/XAUUSD.SIM-5-MINUTE-LAST-EXTERNAL"
MODEL_PATH = "models/checkpoints/ppo_neurotrader"
LOG_DIR = "logs"
TIMESTEPS = 100_000 # Adjust as needed

from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import BarType

def load_data(data_dir):
    # data_dir ends in .../XAUUSD.SIM-5-MINUTE-LAST-EXTERNAL
    # We need the root of the store, which is data/nautilus_store
    # And the BarType string is the last folder name.
    
    # But wait, ParquetDataCatalog expects the root path.
    # The provided data_dir in main is the deep path.
    # Let's extract root and bar_type_str.
    
    # Assuming standard structure: root/data/bar/BARTYPE
    # So root is data_dir.split("/data/bar")[0]
    
    if "/data/bar" not in data_dir:
        # Fallback for flexibility
        root_path = "data/nautilus_store"
        bar_type_str = "XAUUSD.SIM-5-MINUTE-LAST-EXTERNAL"
    else:
        root_path = data_dir.split("/data/bar")[0]
        bar_type_str = data_dir.split("/")[-1]
        
    print(f"Catalog Root: {root_path}")
    print(f"Bar Type: {bar_type_str}")
    
    catalog = ParquetDataCatalog(root_path)
    bar_type = BarType.from_str(bar_type_str)
    
    bars = list(catalog.bars(bar_types=[bar_type]))
    print(f"Loaded {len(bars)} bars from catalog.")
    
    if not bars:
        raise ValueError("No bars loaded.")
        
    # Convert to DataFrame
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
    df = df.sort_values('timestamp')
    return df

def main():
    print(f"Loading data from {DATA_DIR}...")
    df = load_data(DATA_DIR)
    print(f"Raw data shape: {df.shape}")
    
    print("Feature Engineering...")
    df = add_features(df)
    print(f"Processed data shape: {df.shape}")
    
    # Create Env
    print("Initializing Environment...")
    env = TradingEnv(df)
    
    # Check Env
    check_env(env)
    print("Environment check passed.")
    
    # Verify vectorization support
    vec_env = DummyVecEnv([lambda: env])
    
    print(f"Training PPO for {TIMESTEPS} timesteps...")
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)
    
    model.learn(total_timesteps=TIMESTEPS)
    
    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
