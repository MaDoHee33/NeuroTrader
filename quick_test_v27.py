"""
Quick Test Script for V2.7 Scalper
===================================
Run this when RAM is free (close IDE first):

    python quick_test_v27.py

Expected output: Holding time metrics to verify V2.7 improvements.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

def main():
    print("="*60)
    print("V2.7 SCALPER QUICK TEST")
    print("="*60)
    
    # Load model
    print("\n📦 Loading model...")
    from sb3_contrib import RecurrentPPO
    model = RecurrentPPO.load('models/trinity_scalper_XAUUSDm_M5.zip')
    print(f"   Policy: {type(model.policy).__name__}")
    
    # Load data
    print("\n📊 Loading data...")
    df = pd.read_parquet('data/raw/XAUUSDm_M5_raw.parquet')
    if 'tick_volume' in df.columns:
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    print(f"   Total rows: {len(df)}")
    
    # Create environment
    print("\n🌍 Creating test environment...")
    from src.brain.env.trading_env import TradingEnv
    test_df = df.tail(2000).copy()  # Last 2000 rows for testing
    env = TradingEnv({'XAUUSD_M5': test_df}, agent_type='scalper')
    
    # Run evaluation
    print("\n🎮 Running evaluation (1000 steps)...")
    obs, _ = env.reset()
    lstm_states = None
    done = False
    
    total_reward = 0.0
    holding_times = []
    actions = {0: 0, 1: 0, 2: 0}  # hold, buy, sell
    trades_info = []
    
    for step in range(1000):
        if done:
            break
            
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        # Handle 0-d array from newer sb3/numpy versions
        if isinstance(action, np.ndarray) and action.ndim == 0:
            action_val = int(action.item())
        else:
            action_val = int(action[0]) if hasattr(action, '__len__') else int(action)
        actions[action_val] = actions.get(action_val, 0) + 1
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Track trades
        if info.get('last_trade'):
            trade = info['last_trade']
            trades_info.append(trade)
            if 'holding_time' in trade:
                holding_times.append(trade['holding_time'])
    
    # Results
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    
    print(f"\n💰 Performance:")
    print(f"   Total Reward: {total_reward:.4f}")
    print(f"   Final Balance: ${env.balance:.2f}")
    print(f"   Final Equity:  ${env.equity:.2f}")
    print(f"   Return: {((env.equity - 10000) / 10000 * 100):.2f}%")
    
    print(f"\n🎯 Actions Distribution:")
    print(f"   Hold: {actions.get(0, 0)}")
    print(f"   Buy:  {actions.get(1, 0)}")
    print(f"   Sell: {actions.get(2, 0)}")
    
    print(f"\n📈 Trade Analysis:")
    print(f"   Total Trades: {len(trades_info)}")
    
    if holding_times:
        avg_hold = np.mean(holding_times)
        print(f"\n⏱️ HOLDING TIME (Key Metric for V2.7):")
        print(f"   Average: {avg_hold:.1f} steps ({avg_hold * 5:.0f} minutes)")
        print(f"   Max: {max(holding_times)} steps ({max(holding_times) * 5} min)")
        print(f"   Min: {min(holding_times)} steps ({min(holding_times) * 5} min)")
        print(f"   Quick trades (<12 steps): {sum(1 for h in holding_times if h < 12)}/{len(holding_times)}")
        
        # Verdict
        print("\n" + "="*60)
        if avg_hold < 12:
            print("✅ V2.7 SUCCESS! Average holding time is under 1 hour.")
        elif avg_hold < 24:
            print("🟡 V2.7 PARTIAL: Holding time improved but still over 1 hour avg.")
        else:
            print("❌ V2.7 NEEDS WORK: Holding time still too long (>2 hours avg).")
    else:
        print("   No completed trades in test period.")
        print("\n⚠️ Model may not be trading actively - needs more investigation.")
    
    print("="*60)


if __name__ == "__main__":
    main()
