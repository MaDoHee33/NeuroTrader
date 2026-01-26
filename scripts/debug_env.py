
import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from src.brain.env.trading_env import TradingEnv
from src.brain.feature_eng import add_features

# Load real data
data_path = "data/processed/XAUUSD_M5_processed.parquet"
print(f"Loading data from {data_path}...")
df = pd.read_parquet(data_path).iloc[:1000] # Take small slice
# Fix index if needed
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
    # df.set_index('time', drop=False, inplace=True) # Env handles this usually

# Add features (V4)
df = add_features(df)
df = df.dropna()

print("🧪 Testing TradingEnv Hard Constraints...")

# Case 1: Default (36 steps)
env = TradingEnv(df, agent_type='scalper')
env.reset()

# Force Buy at step 0
print("\n--- Test 1: Force Exit at 36 steps ---")
env.step(1) # BUY
print(f"Step 1: Position={env.position}, Held={env.steps_in_position}")

# Hold for 35 steps (Total 36 -> Expect Force Exit)
for i in range(35):
    obs, reward, done, _, info = env.step(0) # HOLD
    # print(f"Step {i+2}: Held={env.steps_in_position}")

print(f"Step 36: Position={env.position}, Held={env.steps_in_position}")

if env.position == 0:
    print("✅ Force Exit WORKING (Position=0)")
else:
    print(f"❌ Force Exit FAILED (Position={env.position})")

# Case 2: Custom Config (10 steps)
print("\n--- Test 2: Custom Limit (10 steps) ---")
env = TradingEnv(df, agent_type='scalper', reward_config={'max_holding_steps': 10})
env.reset()
env.step(1) # BUY
for i in range(9):
    env.step(0)
    
print(f"Step 10: Position={env.position}")
if env.position == 0:
    print("✅ Custom Limit WORKING")
else:
    print(f"❌ Custom Limit FAILED")
