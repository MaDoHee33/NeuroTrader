#!/usr/bin/env python3
"""
NeuroNautilus Training Script
Supports both Local and Google Colab environments
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Environment Detection
def is_colab():
    try:
        import google.colab
        return True
    except:
        return False

def setup_colab():
    """Mount Google Drive and clone repo if needed"""
    if not is_colab():
        return None
        
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Define workspace path
    workspace = '/content/drive/MyDrive/NeuroTrader_Workspace'
    os.makedirs(workspace, exist_ok=True)
    
    # Clone repo if not exists
    repo_path = '/content/NeuroTrader'
    if not os.path.exists(repo_path):
        print("📥 Cloning NeuroTrader repository...")
        os.system('git clone https://github.com/MaDoHee33/NeuroTrader.git /content/NeuroTrader')
        os.system('cd /content/NeuroTrader && git checkout neuronautilus-v1')
    
    # Add to path
    sys.path.insert(0, repo_path)
    
    return workspace

# Initialize environment
WORKSPACE = setup_colab() if is_colab() else str(Path(__file__).resolve().parent.parent.parent)

# Import after path setup
from src.brain.env.trading_env import TradingEnv
from src.brain.feature_eng import add_features
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import BarType

def get_paths(workspace, args):
    """Get platform-specific paths"""
    base = Path(workspace)
    
    paths = {
        'data': base / 'data' / 'nautilus_store',
        'models': base / 'models' / 'checkpoints',
        'logs': base / 'logs'
    }
    
    # Create directories
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    
    # Data can be overridden
    if args.data_dir:
        paths['data'] = Path(args.data_dir)
    
    return paths

def load_data(data_path, bar_type_str):
    """Load Nautilus Parquet data and convert to DataFrame"""
    print(f"📂 Loading from: {data_path}")
    print(f"📊 Bar Type: {bar_type_str}")
    
    catalog = ParquetDataCatalog(str(data_path))
    bar_type = BarType.from_str(bar_type_str)
    
    bars = list(catalog.bars(bar_types=[bar_type]))
    print(f"✅ Loaded {len(bars):,} bars")
    
    if not bars:
        raise ValueError("❌ No bars loaded!")
    
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
    
    df = pd.DataFrame(data).sort_values('timestamp').reset_index(drop=True)
    return df

def train_model(args):
    """Main training function"""
    print(f"{'='*60}")
    print(f"🧠 NeuroNautilus Training")
    print(f"📍 Environment: {'Google Colab' if is_colab() else 'Local'}")
    print(f"{'='*60}\n")
    
    # Get paths
    paths = get_paths(WORKSPACE, args)
    
    # Load data
    df = load_data(paths['data'], args.bar_type)
    print(f"📈 Raw data shape: {df.shape}")
    
    # Feature Engineering
    print("\n⚙️  Applying feature engineering...")
    df = add_features(df)
    print(f"✅ Processed shape: {df.shape}")
    
    # Create Environment
    print("\n🏗️  Creating trading environment...")
    env = TradingEnv(df)
    check_env(env)
    vec_env = DummyVecEnv([lambda: env])
    print("✅ Environment validated")
    
    # Model Configuration
    model_path = paths['models'] / args.model_name
    print(f"\n🤖 Initializing PPO Model...")
    print(f"   - Policy: MlpPolicy")
    print(f"   - Timesteps: {args.timesteps:,}")
    print(f"   - Learning Rate: {args.learning_rate}")
    print(f"   - Save Path: {model_path}")
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log=str(paths['logs'])
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(paths['models']),
        name_prefix='ppo_checkpoint'
    )
    
    # Train
    print(f"\n🏃 Training started...\n")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save
    print(f"\n💾 Saving model to {model_path}...")
    model.save(str(model_path))
    print("✅ Training complete!")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train NeuroNautilus PPO Model')
    
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='Total training timesteps (default: 1M)')
    parser.add_argument('--bar-type', type=str, 
                        default='XAUUSD.SIM-5-MINUTE-LAST-EXTERNAL',
                        help='Bar type to load')
    parser.add_argument('--model-name', type=str, default='ppo_neurotrader',
                        help='Model save name')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='PPO learning rate')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory')
    
    args = parser.parse_args()
    
    train_model(args)

if __name__ == "__main__":
    main()
