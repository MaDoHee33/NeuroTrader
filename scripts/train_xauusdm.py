
import argparse
import pandas as pd
import sys
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.brain.env.trading_env import TradingEnv

def main():
    parser = argparse.ArgumentParser(description='Train PPO on XAUUSDm')
    parser.add_argument('--timesteps', type=int, default=1_000_000, help='Total training timesteps')
    parser.add_argument('--data-file', type=str, default='XAUUSDm_M5_processed.parquet', help='Filename in data/processed/')
    parser.add_argument('--model-name', type=str, default='ppo_xauusdm', help='Model save name')
    args = parser.parse_args()

    # Paths
    data_path = ROOT_DIR / 'data' / 'processed' / args.data_file
    models_dir = ROOT_DIR / 'models'
    log_dir = ROOT_DIR / 'logs'
    
    models_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    print(f"🚀 Starting Training for {args.timesteps:,} steps")
    print(f"📂 Loading data from: {data_path}")

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return

    # Load Data
    df = pd.read_parquet(data_path)
    print(f"✅ Loaded {len(df):,} rows")

    # Create Env
    print("🏗️  Creating Environment...")
    # The processed data already has features, so we pass it directly
    env = TradingEnv(df)
    
    # Check Env
    check_env(env)
    print("✅ Environment validated")
    
    vec_env = DummyVecEnv([lambda: env])

    # Model
    print("🤖 Initializing PPO Model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
        tensorboard_log=str(log_dir)
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, 
        save_path=str(models_dir / 'checkpoints'),
        name_prefix=args.model_name
    )

    # Train
    print("🏃 Training Started...")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback, progress_bar=True)
    
    # Save
    save_path = models_dir / f"{args.model_name}_final"
    model.save(save_path)
    print(f"✅ Training Complete. Model saved to {save_path}.zip")

if __name__ == "__main__":
    main()
