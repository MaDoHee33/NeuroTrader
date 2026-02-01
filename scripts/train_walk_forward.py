
import argparse
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.brain.env.trading_env import TradingEnv

def calculate_metrics(df_res, initial_balance):
    final_balance = df_res['equity'].iloc[-1]
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    # Drawdown
    df_res['peak'] = df_res['equity'].cummax()
    df_res['drawdown'] = (df_res['equity'] - df_res['peak']) / df_res['peak'] * 100
    max_drawdown = df_res['drawdown'].min()
    
    # Sharpe
    df_res['returns'] = df_res['equity'].pct_change().dropna()
    mean_ret = df_res['returns'].mean()
    std_ret = df_res['returns'].std()
    
    if std_ret == 0:
        sharpe = 0
    else:
        sharpe = mean_ret / std_ret * np.sqrt(288 * 252)
        
    return {
        'Initial Equity': initial_balance,
        'Final Equity': final_balance,
        'Total Return (%)': total_return,
        'Max Drawdown (%)': max_drawdown,
        'Sharpe Ratio': sharpe
    }

def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Validation')
    parser.add_argument('--timesteps', type=int, default=1_000_000, help='Training timesteps')
    parser.add_argument('--data-file', type=str, default='XAUUSDm_M5_processed.parquet')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='Train split ratio')
    args = parser.parse_args()

    data_path = ROOT_DIR / 'data' / 'processed' / args.data_file
    models_dir = ROOT_DIR / 'models'
    analysis_dir = ROOT_DIR / 'analysis'
    log_dir = ROOT_DIR / 'logs'
    
    models_dir.mkdir(exist_ok=True)
    analysis_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    print(f"🚀 Starting Walk-Forward Validation")
    print(f"📂 Data: {data_path.name}")
    
    # 1. Load & Split Data
    df = pd.read_parquet(data_path)
    split_idx = int(len(df) * args.split_ratio)
    
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    print(f"📊 Data Split:")
    print(f"   - Train: {len(df_train):,} bars ({df_train.index[0]} to {df_train.index[-1]})")
    print(f"   - Test : {len(df_test):,} bars ({df_test.index[0]} to {df_test.index[-1]})")
    
    # 2. Train (on Train Set)
    print("\n🏗️  Initializing Training Environment...")
    env_train = TradingEnv(df_train)
    vec_env_train = DummyVecEnv([lambda: env_train])
    
    model_name = "ppo_wf_validation"
    print(f"🤖 Training Model ({args.timesteps:,} steps)...")
    
    model = PPO(
        "MlpPolicy",
        vec_env_train,
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
    
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    
    save_path = models_dir / model_name
    model.save(save_path)
    print(f"✅ Training Complete. Model saved to {save_path}.zip")
    
    # 3. Validate (on Test Set)
    print("\n🧐 Validating on Unseen Test Data...")
    env_test = TradingEnv(df_test) # Fresh env with test data
    
    # Reset
    obs, _ = env_test.reset()
    done = False
    history = []
    
    print("🏃 Running Backtest...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env_test.step(action)
        
        step_data = {
            'step': info['step'],
            'equity': info['equity'],
            'price': df_test.iloc[info['step']]['close'],
            'date': df_test.index[info['step']]
        }
        history.append(step_data)
        
        if done or truncated:
            break
            
    # 4. Results
    df_res = pd.DataFrame(history)
    metrics = calculate_metrics(df_res, env_test.initial_balance)
    
    print("\n" + "="*40)
    print("📊 WALK-FORWARD VALIDATION RESULTS")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.2f}")
    print("="*40)
    
    # Chart
    plt.figure(figsize=(12, 6))
    plt.plot(df_res['date'], df_res['equity'], label='Equity (Test Set)')
    plt.title(f'Walk-Forward Validation (Out-of-Sample) - {args.data_file}')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    chart_path = analysis_dir / 'wf_validation_chart.png'
    plt.savefig(chart_path)
    print(f"\n📈 Validation Chart saved to: {chart_path}")

if __name__ == "__main__":
    main()
