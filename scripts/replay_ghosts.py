
"""
Ghost Replay: Offline Reinforcement Learning
============================================
This script implements "Ghost Replay" - a technique to learn from past model iterations.
It loads old (potentially flawed) checkpoints and runs them in the *current* corrected environment.

Key Concept:
- Actor: Old Model (The "Ghost") -> Takes actions based on its old policy.
- Critic/Judge: Current Env (The "Teacher") -> Gives rewards based on NEW corrected logic.
- Student: Hybrid Agent -> Observes and records this interaction into its memory.

Benefits:
- Generates massive amounts of training data without waiting for training.
- Helps the agent learn "what NOT to do" (if old models were bad).
- Stabilizes learning by providing a diverse distribution of behaviors.
"""

import glob
import sys
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from stable_baselines3 import PPO

# Add project root to path
sys.path.insert(0, '.')

from src.brain.env.trading_env import TradingEnv
from src.evolving.hybrid_agent import HybridTradingAgent

def replay_ghosts(
    data_path: str,
    checkpoints_dir: str = "models/checkpoints",
    agent_name: str = "hybrid_gen1",
    episodes_per_ghost: int = 10,
    max_ghosts: int = 5
):
    print(f"\n👻 STARTING GHOST REPLAY (Offline RL)")
    print(f"=====================================")
    
    # 1. Find Ghosts (Old Checkpoints)
    # Search recursively for .zip files
    search_path = Path(checkpoints_dir)
    ghost_files = list(search_path.rglob("*.zip"))
    
    if not ghost_files:
        print("❌ No ghost checkpoints found!")
        return

    # Sort by modification time (recent first) to get "mature" ghosts
    ghost_files.sort(key=os.path.getmtime, reverse=True)
    
    # Limit number of ghosts
    selected_ghosts = ghost_files[:max_ghosts]
    print(f"Found {len(ghost_files)} ghosts. Selected top {len(selected_ghosts)}:")
    for g in selected_ghosts:
        print(f"  - {g.name} ({g.parent.name})")

    # 2. Setup Environment (The Judge)
    print(f"\n⚖️  Setting up current environment...")
    # Using 'scalper' config for rewards
    # Important: Data must be RAW for correct simulation
    df = pd.read_parquet(data_path)
    if 'tick_volume' in df.columns:
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
    env = TradingEnv(
        {'XAUUSD': df},  
        agent_type='scalper', # <--- Uses CURRENT corrected reward logic
        initial_balance=10000
    )
    
    # 3. Setup Hybrid Agent (The Student)
    print(f"🧠 Initializing Student ({agent_name})...")
    # We don't need PPO here, just the memory system
    student = HybridTradingAgent(
        use_curiosity=True,
        use_experience_buffer=True,
        agent_name=agent_name
    )
    
    total_new_experiences = 0
    
    # 4. Replay Loop
    for ghost_idx, ghost_path in enumerate(selected_ghosts):
        print(f"\n🔮 Summoning Ghost {ghost_idx+1}/{len(selected_ghosts)}: {ghost_path.name}")
        
        try:
            # Load the Ghost
            ghost_model = PPO.load(ghost_path)
            
            pbar = tqdm(range(episodes_per_ghost), desc=f"Replaying {ghost_path.parent.name}")
            for _ in pbar:
                obs, _ = env.reset()
                done = False
                truncated = False
                episode_pnl = 0
                
                while not done and not truncated:
                    # GHOST ACTS
                    action, _ = ghost_model.predict(obs, deterministic=False)
                    
                    # ENV JUDGES (New Reward)
                    next_obs, reward, done, truncated, info = env.step(action)
                    
                    # STUDENT LEARNS (Records corrected reward)
                    student.store_experience(
                        obs, 
                        action.item() if hasattr(action, 'item') else action, 
                        reward, 
                        next_obs, 
                        info
                    )
                    
                    obs = next_obs
                    total_new_experiences += 1
                    
                    if info and 'pnl' in info:
                         episode_pnl += info['pnl']
                
                pbar.set_postfix({'PnL': f"{episode_pnl:.2f}"})
                
            # Save student's memory after each ghost
            student.save()
            
        except Exception as e:
            print(f"⚠️ Failed to replay ghost {ghost_path.name}: {e}")
            continue

    print(f"\n✅ Ghost Replay Completed!")
    print(f"Total new experiences stored: {total_new_experiences}")
    print(f"Student memory updated at: data/hybrid_agent/{agent_name}_experiences.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to raw parquet data")
    parser.add_argument("--dir", type=str, default="models", help="Directory to search for checkpoints")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per ghost")
    parser.add_argument("--max", type=int, default=5, help="Max ghosts to replay")
    
    args = parser.parse_args()
    
    replay_ghosts(
        data_path=args.data, 
        checkpoints_dir=args.dir,
        episodes_per_ghost=args.episodes,
        max_ghosts=args.max
    )
