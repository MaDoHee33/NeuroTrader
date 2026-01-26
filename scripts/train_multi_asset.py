
import argparse
import pandas as pd
import glob
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

def load_all_assets(data_dir: Path):
    """Loads all processed parquet files into a dictionary"""
    assets = {}
    files = list(data_dir.glob("*_processed.parquet"))
    
    if not files:
        print("❌ No processed data files found!")
        return {}
        
    print(f"📂 Found {len(files)} data files.")
    
    for f in files:
        try:
            # Key = Filename without extension (e.g., XAUUSDm_M5_processed)
            key = f.stem.replace('_processed', '')
            df = pd.read_parquet(f)
            if not df.empty:
                assets[key] = df
                # print(f"   Loaded {key}: {len(df):,} rows")
        except Exception as e:
            print(f"   ⚠️ Error loading {f.name}: {e}")
            
    return assets

def main():
    parser = argparse.ArgumentParser(description='Train PPO on Multi-Asset Data')
    parser.add_argument('--timesteps', type=int, default=1_000_000, help='Total training timesteps')
    parser.add_argument('--model-name', type=str, default='ppo_multi_asset', help='Model save name')
    parser.add_argument('--ent-coef', type=float, default=0.05, help='Entropy coefficient for exploration')
    args = parser.parse_args()

    # Paths
    processed_dir = ROOT_DIR / 'data' / 'processed'
    models_dir = ROOT_DIR / 'models'
    log_dir = ROOT_DIR / 'logs'
    
    models_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    print(f"🚀 Starting Multi-Asset Training")
    
    # 1. Load Data
    all_assets = load_all_assets(processed_dir)
    if not all_assets:
        return

    # 2. Split Data (Train/Test)
    train_assets = {}
    test_assets = {}
    split_ratio = 0.8
    
    print(f"\n✂️  Splitting Data (Ratio: {split_ratio:.0%})")
    for name, df in all_assets.items():
        split_idx = int(len(df) * split_ratio)
        train_assets[name] = df.iloc[:split_idx].copy()
        test_assets[name] = df.iloc[split_idx:].copy()
        # print(f"   - {name}: Train={len(train_assets[name]):,} | Test={len(test_assets[name]):,}")
        
    print(f"✅ Prepared {len(train_assets)} training sets and {len(test_assets)} testing sets.")

    # 3. Create Training Environment
    print("\n🏗️  Creating Multi-Asset Environment (Training)...")
    env = TradingEnv(train_assets)
    
    # Check Env
    env.reset()
    check_env(env) 
    
    vec_env = DummyVecEnv([lambda: env])

    # 4. Model Configuration (Regularized)
    print(f"🤖 Initializing PPO Model (Ent Coef: {args.ent_coef})...")
    
    # Network Architecture: Small [64, 64] to prevent memorization
    policy_kwargs = dict(net_arch=[64, 64])
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir)
    )

    # 5. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, 
        save_path=str(models_dir / 'checkpoints'),
        name_prefix=args.model_name
    )

    # 6. Train
    print("\n🏃 Training Started...")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback, progress_bar=True)
    
    # Save
    save_path = models_dir / f"{args.model_name}_final"
    model.save(save_path)
    print(f"✅ Training Complete. Model saved to {save_path}.zip")
    
    # 7. Validation (Out-of-Sample)
    print("\n" + "="*50)
    print("🧐 STARTING OUT-OF-SAMPLE VALIDATION")
    print("="*50)
    
    validation_results = []
    
    for name, df_test in test_assets.items():
        print(f"   Testing on {name} ({len(df_test):,} bars)...", end=" ")
        
        # Create temp env for this asset
        env_test = TradingEnv(df_test)
        obs, _ = env_test.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env_test.step(action)
            
        final_equity = info['equity']
        return_pct = (final_equity - env_test.initial_balance) / env_test.initial_balance * 100
        
        print(f"Return: {return_pct:.2f}%")
        
        validation_results.append({
            'Asset': name,
            'Return (%)': return_pct,
            'Final Equity': final_equity
        })
        
    # Summary
    print("\n📊 VALIDATION SUMMARY")
    df_val = pd.DataFrame(validation_results)
    print(df_val.to_string(index=False))
    
    avg_return = df_val['Return (%)'].mean()
    print(f"\n⭐ Average Return across all assets: {avg_return:.2f}%")
    
    # Save CSV
    df_val.to_csv(ROOT_DIR / 'analysis' / 'multi_asset_validation_results.csv', index=False)


if __name__ == "__main__":
    main()
