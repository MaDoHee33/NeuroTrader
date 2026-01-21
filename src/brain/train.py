#!/usr/bin/env python3
"""
NeuroNautilus Training Script
Supports both Local and Google Colab environments
"""

import os
import sys
import argparse
import pandas as pd
import time
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
    
    # Check if Drive is already mounted
    drive_path = '/content/drive'
    if not os.path.exists(f'{drive_path}/MyDrive'):
        try:
            from google.colab import drive
            print("📂 Mounting Google Drive...")
            drive.mount(drive_path)
        except Exception as e:
            print(f"⚠️  Could not mount Drive automatically: {e}")
            print("💡 Please mount Drive manually in the notebook before running this script.")
            print("   Run this in a cell: from google.colab import drive; drive.mount('/content/drive')")
            return None
    else:
        print("✅ Google Drive already mounted")
    
    # Define workspace path
    workspace = f'{drive_path}/MyDrive/NeuroTrader_Workspace'
    os.makedirs(workspace, exist_ok=True)
    
    # Clone repo if not exists
    repo_path = '/content/NeuroTrader'
    if not os.path.exists(repo_path):
        print("📥 Cloning NeuroTrader repository...")
        os.system('git clone https://github.com/MaDoHee33/NeuroTrader.git /content/NeuroTrader')
        os.system('cd /content/NeuroTrader && git checkout neuronautilus-v1')
    else:
        print("✅ Repository already cloned")
    
    # Add to path
    if repo_path not in sys.path:
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
    
    # DEBUG: Check path
    import os
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Path does not exist!")
        print(f"   Looking for: {data_path}")
        raise FileNotFoundError(f"Path not found: {data_path}")
    print(f"✅ Path exists")
    
    catalog = ParquetDataCatalog(str(data_path))
    
    # DEBUG: Show what's available
    try:
        print(f"\n🔍 Catalog contents:")
        instruments = list(catalog.instruments())
        print(f"   Instruments: {[str(i.id) for i in instruments[:5]]}")
        bar_types_available = list(catalog.bar_types())
        print(f"   Bar types: {[str(bt) for bt in bar_types_available[:5]]}")
    except Exception as e:
        print(f"⚠️  Could not inspect catalog: {e}")
    
    bar_type = BarType.from_str(bar_type_str)
    print(f"\n🎯 Requesting bar_type: {bar_type}")
    
    bars = list(catalog.bars(bar_types=[bar_type_str]))
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
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def train_model(args):
    """Main training function"""
    print(f"{'='*60}")
    print(f"🧠 NeuroNautilus Training")
    print(f"📍 Environment: {'Google Colab' if is_colab() else 'Local'}")
    if args.resume:
        print(f"🔄 Resuming from: {args.resume}")
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
    
    if args.resume:
        # Resume from checkpoint
        print(f"\n📂 Loading checkpoint: {args.resume}")
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"❌ Checkpoint not found: {args.resume}")
        
        model = PPO.load(
            args.resume,
            env=vec_env,
            tensorboard_log=str(paths['logs'])
        )
        print("✅ Checkpoint loaded successfully")
        print(f"   - Continuing training for {args.timesteps:,} more steps")
    else:
        # Create new model with RESEARCH-BASED HYPERPARAMETERS
        print(f"\n🤖 Initializing PPO Model with Research-Optimized Parameters...")
        print(f"   📚 Based on academic papers for financial trading")
        print(f"   - Policy: MlpPolicy")
        print(f"   - Timesteps: {args.timesteps:,}")
        print(f"   - Learning Rate: 0.0003 (research optimal)")
        print(f"   - Gamma: 0.99 (long-term focus)")
        print(f"   - N-Steps: 2048 (high stability)")
        
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=0.0003,      # Research optimal for trading
            gamma=0.99,                 # Long-term focus (standard for finance)
            gae_lambda=0.95,           # Generalized Advantage Estimation
            clip_range=0.2,            # PPO clipping (stability)
            ent_coef=0.01,             # Entropy for exploration
            vf_coef=0.5,               # Value function coefficient
            max_grad_norm=0.5,         # Gradient clipping
            n_steps=2048,              # Steps per update (large for stability)
            batch_size=64,             # Batch size for gradient descent
            n_epochs=10,               # Epochs per PPO update
            verbose=1,
            tensorboard_log=str(paths['logs'])
        )
    
    print(f"   - Save Path: {model_path}")
    
    # Checkpoint callback (save intermediate models)
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,  # Save every 1M steps
        save_path=str(paths['models'] / 'checkpoints'),
        name_prefix='ppo_checkpoint',
        save_replay_buffer=False,
        verbose=1
    )
    
    # Training
    print(f"\n{'='*60}")
    print(f"🏃 Starting Training")
    print(f"{'='*60}")
    print(f"⏱️  Estimated time: {args.timesteps / 1_000_000 * 30:.0f}-{args.timesteps / 1_000_000 * 60:.0f} minutes")
    print(f"📊 Progress logged to: {paths['logs']}")
    print(f"💾 Checkpoints saved to: {paths['models'] / 'checkpoints'}")
    print()
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Training Complete!")
    print(f"⏱️  Time elapsed: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    
    # Save final model
    final_model_path = model_path.with_suffix('.zip')
    print(f"\n💾 Saving final model to: {final_model_path}")
    model.save(final_model_path)
    print(f"✅ Final model saved!")
    
    # AUTO-CLEANUP: Delete intermediate checkpoints
    print(f"\n🧹 Cleaning up intermediate checkpoints...")
    checkpoint_dir = paths['models'] / 'checkpoints'
    if checkpoint_dir.exists():
        deleted_count = 0
        kept_files = []
        
        for checkpoint_file in checkpoint_dir.glob('ppo_checkpoint_*_steps.zip'):
            try:
                checkpoint_file.unlink()
                deleted_count += 1
                print(f"   🗑️  Deleted: {checkpoint_file.name}")
            except Exception as e:
                print(f"   ⚠️  Could not delete {checkpoint_file.name}: {e}")
        
        # List remaining files
        remaining = list(checkpoint_dir.glob('*.zip'))
        if remaining:
            print(f"\n📁 Kept files in checkpoints folder:")
            for f in remaining:
                print(f"   ✅ {f.name}")
        
        print(f"\n✅ Cleanup complete! Deleted {deleted_count} intermediate checkpoint(s)")
        print(f"💾 Final model kept at: {final_model_path}")
    else:
        print(f"   ⚠️  Checkpoint directory not found, skipping cleanup")
    
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
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    train_model(args)

if __name__ == "__main__":
    main()
