
"""
Train Hybrid AI (Phase 3.2: Trinity)
====================================
Training script for Hybrid Agents using PPO + Hybrid Rewards.

Features:
- Role Selection: Scalper / Swing / Trend
- Reward Mixing: Extrinsic (PnL) + Intrinsic (Curiosity)
- Uses Standard PPO.learn() loop via Wrapper
- Ghost Replay Integration (loads buffer if exists)

Usage:
    python scripts/train_hybrid.py --role scalper --episodes 1000 --profit_focus
"""

import argparse
import sys
import yaml
import time
from pathlib import Path
from stable_baselines3 import PPO

# Add project root to path
sys.path.insert(0, '.')

from src.brain.env.trading_env import TradingEnv
from src.evolving.curiosity import CuriosityModule
from src.evolving.experience_buffer import ExperienceBuffer
from src.evolving.wrapper import HybridRewardWrapper
# from src.utils.data_loader import load_and_process_data

def train_hybrid(
    role: str = 'scalper',
    total_timesteps: int = 100000,
    extrinsic_weight: float = 1.0,
    intrinsic_weight: float = 0.1,
    model_path: str = None
):
    print(f"\n🚀 STARTING HYBRID TRINITY TRAINING: {role.upper()}")
    print(f"==================================================")
    print(f"Weights: PnL={extrinsic_weight} | Curiosity={intrinsic_weight}")
    
    # 1. Load Configuration
    with open('config/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    role_config = config['roles'].get(role)
    if not role_config:
        print(f"❌ Role '{role}' not found in config.")
        return

    symbol = role_config['symbols'][0] # Use first symbol
    timeframe = role_config['timeframes'][0] # Use first timeframe
    print(f"Config: {symbol} on {timeframe}")

    # 2. Load Data
    # For Scalper, we use the RAW data because features are computed in Env or previously
    # But wait, TradingEnv handles feature computation if we pass raw?
    # No, usually we pass processed features.
    # The config says `data.base_dir` is now `data/raw`.
    # Let's verify data loading logic.
    # We will attempt to load the raw parquet file corresponding to the symbol/timeframe.
    
    data_path = Path(config['data']['base_dir']) / f"{symbol}_{timeframe}_raw.parquet"
    print(f"Loading data from: {data_path}")
    
    import pandas as pd
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
        
    df = pd.read_parquet(data_path)
    if 'tick_volume' in df.columns:
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    
    # 3. Setup Environment
    print("🛠️  Setting up environment...")
    env = TradingEnv(
        {symbol: df},
        agent_type=role,
        initial_balance=10000
    )
    
    # 4. Setup Hybrid Components
    agent_name = f"hybrid_{role}"
    data_dir = Path("data/hybrid_agent")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧠 Initializing Hybrid Cortex...")
    curiosity = CuriosityModule(
        novelty_weight=0.3,
        prediction_weight=0.4
    )
    
    buffer = ExperienceBuffer(
        max_size=50000,
        save_path=data_dir / f"{agent_name}_experiences.json"
    )
    
    # Load existing buffer (Ghost Replay data!)
    if buffer.save_path.exists():
        print(f"📚 Loading existing memories ({len(buffer.experiences)} experiences)...")
        buffer.load()

    # 5. Wrap Environment
    env = HybridRewardWrapper(
        env, 
        curiosity, 
        buffer,
        extrinsic_weight=extrinsic_weight,
        intrinsic_weight=intrinsic_weight
    )
    
    # 6. Initialize Agent
    save_path = Path(f"models/hybrid/{role}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    if model_path and Path(model_path).exists():
        print(f"🔄 Loading checkpoint: {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("👶 Creating new Hybrid Agent")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            ent_coef=0.01
        )
        
    # 7. Training
    print(f"\n🏃 Training for {total_timesteps} steps...")
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            progress_bar=True,
            tb_log_name=f"hybrid_{role}"
        )
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        
    # 8. Save Everything
    print("\n💾 Saving Agent State...")
    model.save(save_path / "final_model")
    buffer.save()
    
    curiosity_path = data_dir / f"{agent_name}_curiosity.json"
    import json
    with open(curiosity_path, 'w') as f:
        json.dump(curiosity.save_state(), f)
        
    print(f"✅ Training Complete!")
    print(f"Model: {save_path / 'final_model.zip'}")
    print(f"Memories: {buffer.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str, default="scalper", choices=['scalper', 'swing', 'trend'])
    parser.add_argument("--steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--profit_focus", action='store_true', help="Set high extrinsic weight (1.0) and low intrinsic (0.1)")
    parser.add_argument("--explore_focus", action='store_true', help="Set high intrinsic weight (1.0) and low extrinsic (0.1)")
    parser.add_argument("--model", type=str, default=None, help="Path to resume model")
    
    args = parser.parse_args()
    
    # Determine weights
    w_ex = 1.0
    w_in = 0.1 # Default hybrid
    
    if args.profit_focus:
        w_ex = 1.0
        w_in = 0.05
    elif args.explore_focus:
        w_ex = 0.1
        w_in = 1.0
        
    train_hybrid(
        role=args.role,
        total_timesteps=args.steps,
        extrinsic_weight=w_ex,
        intrinsic_weight=w_in,
        model_path=args.model
    )
