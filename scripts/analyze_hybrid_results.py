
import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_hybrid(data_dir: str = "data/hybrid_agent", agent_name: str = "hybrid_gen1"):
    base_path = Path(data_dir)
    
    # 1. Load Experiences (Trade History)
    exp_path = base_path / f"{agent_name}_experiences.json"
    if not exp_path.exists():
        print("No experience file found.")
        return

    print(f"Loading {exp_path}...")
    with open(exp_path, 'r') as f:
        data = json.load(f)
        
    # Handle wrapped structure (check if list or dict)
    if isinstance(data, dict) and 'experiences' in data:
        # It's a dict of {id: exp_dict}
        experiences = list(data['experiences'].values())
    else:
        experiences = data
        
    df = pd.DataFrame(experiences)
    print(f"Loaded {len(df)} experiences.")
    
    if df.empty:
        print("Experience buffer empty.")
        return

    # Filter for Step with Trades (where action != 0 or pnl != 0)
    # Actually, experiences preserve every step. We need to find "Trade Exits".
    # In this simplified buffer, we stored 'pnl' in every step (usually 0).
    # Non-zero PnL means a trade closed? 
    # Let's check 'lesson_tags'.
    
    trades = df[df['pnl'] != 0].copy()
    
    print("\n--- TRADING PERFORMANCE ---")
    print(f"Total Trades: {len(trades)}")
    
    if not trades.empty:
        total_pnl = trades['pnl'].sum()
        win_rate = (trades['pnl'] > 0).mean() * 100
        avg_pnl = trades['pnl'].mean()
        
        wins = trades[trades['pnl'] > 0]['pnl']
        losses = trades[trades['pnl'] < 0]['pnl']
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Avg PnL per Trade: ${avg_pnl:.2f}")
        
    # 2. Curiosity & Patterns
    cur_path = base_path / f"{agent_name}_curiosity.json"
    if cur_path.exists():
        with open(cur_path, 'r') as f:
            cur_data = json.load(f)
            
        print("\n--- CURIOSITY INSIGHTS ---")
        stats = cur_data.get('stats', {})
        print(f"Unique States Visited: {stats.get('unique_states', 0)}")
        print(f"Patterns Found: {stats.get('patterns_found', 0)}")
        
        patterns = cur_data.get('discovered_patterns', {})
        if patterns:
            # Sort by total_reward
            sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['total_reward'], reverse=True)
            print("\nTop 5 Discovered Patterns:")
            for k, v in sorted_patterns[:5]:
                print(f"  Key: {k[:10]}... | Count: {v['count']} | Avg Reward: {v['avg_reward']:.4f}")

    # 3. Action Distribution
    print("\n--- BEHAVIOR ---")
    action_counts = df['action'].value_counts()
    print("Actions:")
    print(action_counts.to_string())
    
    # 4. Market Regime Analysis (if available)
    if 'market_regime' in df.columns:
        print("\n--- MARKET REGIMES ---")
        regime_counts = df['market_regime'].value_counts()
        print(regime_counts.to_string())
        
        # PnL by Regime
        if not trades.empty and 'market_regime' in trades.columns:
            print("\nPerformance by Regime:")
            print(trades.groupby('market_regime')['pnl'].mean())

if __name__ == "__main__":
    analyze_hybrid()
