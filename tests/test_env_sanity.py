
import pytest
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from brain.env.trading_env import TradingEnv

def create_mock_df():
    # Create sample dataframe
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    data = {
        'time': dates,
        'close': np.linspace(100, 200, 100) + np.random.normal(0, 1, 100),
        'rsi': np.random.uniform(20, 80, 100),
        'macd': np.random.normal(0, 1, 100),
        'macd_signal': np.random.normal(0, 1, 100),
        'bb_high': np.linspace(105, 205, 100),
        'bb_low': np.linspace(95, 195, 100),
        'ema_20': np.linspace(100, 200, 100),
        'ema_50': np.linspace(100, 200, 100)
    }
    return pd.DataFrame(data)

def test_gym_api_compliance():
    df = create_mock_df()
    env = TradingEnv(df)
    
    # Gym utility to check API compliance
    # It checks output shapes, types, etc.
    check_env(env)
    print("✅ Gym API Compliance Check Passed")

def test_step_logic():
    df = create_mock_df()
    env = TradingEnv(df, initial_balance=1000)
    
    obs, info = env.reset()
    
    # Action 1: BUY
    next_obs, reward, done, truncated, info = env.step(1)
    
    # Check if balance decreased (spent) and position increased
    assert env.balance < 1000
    assert env.position > 0
    print("✅ Buy Logic Verified")
    
    # Action 2: SELL
    next_obs, reward, done, truncated, info = env.step(2)
    
    # Check if balance increased (revenue) and position is 0
    assert env.position == 0
    assert env.balance > 0 # Likely around 1000 (minus fees)
    print("✅ Sell Logic Verified")
    
def test_episode_run():
    df = create_mock_df()
    env = TradingEnv(df)
    env.reset()
    
    done = False
    truncated = False
    step = 0
    while not done and not truncated:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        
    assert step == 99 # Length of DF - 1
    print(f"✅ Full Episode Run {step} steps")

if __name__ == "__main__":
    try:
        test_gym_api_compliance()
        test_step_logic()
        test_episode_run()
        print("\n🎉 All Tests Passed!")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
