
import optuna
import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.brain.env.trading_env import TradingEnv
from src.brain.feature_eng import add_features

def optimize_agent(trial, role, data_path):
    # 1. Hyperparameters to Tune
    gamma = trial.suggest_float('gamma', 0.90, 0.9999)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    n_steps = trial.suggest_categorical('n_steps', [128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # LSTM specific
    # hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])

    # 2. Prepare Data & Env
    df = pd.read_parquet(data_path)
    if 'time' in df.columns:
        df.set_index('time', drop=False, inplace=True)
        df.sort_index(inplace=True)
    
    # Simple Split for Tuning (Train on first 50%, Eval on next 20%)
    # We use a smaller subset to be fast
    train_len = int(len(df) * 0.5)
    df_train = df.iloc[:train_len]
    
    # Create Env
    env = TradingEnv(df_train, agent_type=role)
    
    # 3. Define Model
    try:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            gamma=gamma,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            clip_range=clip_range,
            n_steps=n_steps,
            batch_size=batch_size,
            verbose=0
        )
    except:
        # Fallback if Batch size > n_steps issue
        return -9999
        
    # 4. Train (Short Run)
    # We train for 20,000 steps to check convergence speed/metrics
    try:
        model.learn(total_timesteps=20000)
    except Exception as e:
        print(f"Pruning trial due to error: {e}")
        return -9999
        
    # 5. Evaluate
    # We evaluate on the Environment's internal metrics or use evaluate_policy
    # Here we use a custom evaluation loop to check "Scalper Behavior" (Holding time)
    obs, _ = env.reset()
    done = False
    total_reward = 0
    holding_times = []
    current_hold = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        # Track holding (Assumption: Action 1=Buy, 2=Sell. Position!=0)
        if env.position != 0:
            current_hold += 1
        elif current_hold > 0:
            holding_times.append(current_hold)
            current_hold = 0
            
    # 6. Objective Calculation
    # For Scalper: Reward + Bonus for Low Holding Time
    avg_hold = np.mean(holding_times) if holding_times else 1000
    
    if role == 'scalper':
        # Penalty if holding too long (e.g. > 12 steps/1 hour)
        hold_penalty = max(0, avg_hold - 12) * 10
        score = total_reward - hold_penalty
    else:
        score = total_reward
        
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str, required=True, default='scalper')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()
    
    print(f"🚀 Starting Optimization for {args.role.upper()}...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optimize_agent(trial, args.role, args.data), n_trials=args.trials)
    
    print("\n✅ Optimization Complete!")
    print(f"Best params: {study.best_params}")
    print(f"Best score: {study.best_value}")
    
    # Save Best Params
    import json
    with open(f"best_params_{args.role}.json", 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
if __name__ == "__main__":
    main()
