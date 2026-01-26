import asyncio
import sys
import pandas as pd
import numpy as np
import traceback
import gymnasium as gym
from gymnasium import spaces

print(f"🔹 NumPy Version in Wine: {np.__version__}", flush=True)

# --- NUMPY COMPATIBILITY HACK ---
# Fix for loading models trained on NumPy 2.0+ in NumPy 1.x environment
# Force patch to ensure mapped modules exist
if True: # Always run patch to correspond with cloudpickle expectations
    # 1. Patch numpy._core
    try:
        import numpy.core
        sys.modules['numpy._core'] = numpy.core
        
        # 2. Ensure submodules are mapped and accessible as attributes
        submodules = [
            ('start', None), # Dummy
            ('multiarray', 'multiarray'),
            ('numeric', 'numeric'),
            ('umath', 'umath'),
            ('defchararray', 'defchararray'),
            ('records', 'records'),
            ('memmap', 'memmap'),
            ('function_base', 'function_base'),
            ('machar', 'machar'),
            ('getlimits', 'getlimits'),
            ('shape_base', 'shape_base'),
            ('einsumfunc', 'einsumfunc'),
            ('fromnumeric', 'fromnumeric'),
            ('numerictypes', 'numerictypes'),
            ('_methods', '_methods'),
            ('arrayprint', 'arrayprint'),
            ('_internal', '_internal')
        ]
        
        for attr, real_attr in submodules:
            if not real_attr: continue
            
            # Map module in sys.modules
            target_mod_name = f'numpy.core.{real_attr}'
            alias_mod_name = f'numpy._core.{attr}'
            
            if target_mod_name in sys.modules:
                 sys.modules[alias_mod_name] = sys.modules[target_mod_name]
            elif hasattr(numpy.core, real_attr):
                 try:
                     mod = getattr(numpy.core, real_attr)
                     if isinstance(mod, type(sys)): 
                         sys.modules[alias_mod_name] = mod
                 except:
                     pass
                     
            if hasattr(numpy.core, real_attr):
                setattr(numpy.core, attr, getattr(numpy.core, real_attr))

        print(f"🔧 Applied NumPy attributes patch. numpy._core.numeric present? {'numeric' in dir(numpy.core)}")

        # Verify Patch
        try:
             import numpy._core.numeric
             print("   ✅ Internal Verification: numpy._core.numeric found.")
        except ImportError:
             print("   ❌ Internal Verification: numpy._core.numeric NOT found even after patch.")
             
    except Exception as e:
        print(f"⚠️ Failed to apply NumPy compatibility patch: {e}")
        traceback.print_exc()

    # 3. Patch numpy.random._pickle for BitGenerator compatibility
    try:
        import numpy.random._pickle
        original_ctor = numpy.random._pickle.__bit_generator_ctor
        
        def patched_bit_generator_ctor(bit_generator_name):
            # Check for the specific incompatible class name from NumPy 2.0
            if str(bit_generator_name) == "<class 'numpy.random._pcg64.PCG64'>" or \
               str(bit_generator_name) == "PCG64": 
                from numpy.random import PCG64
                return PCG64()
            
            try:
                return original_ctor(bit_generator_name)
            except ValueError:
                print(f"⚠️  Patching unknown BitGenerator: {bit_generator_name} -> Defaulting to PCG64")
                from numpy.random import PCG64
                return PCG64()

        numpy.random._pickle.__bit_generator_ctor = patched_bit_generator_ctor
        print("🔧 Applied NumPy Random BitGenerator patch.")

    except Exception as e:
        print(f"⚠️ Failed to apply Random BitGenerator patch: {e}")
        traceback.print_exc()

import time
from datetime import datetime, timedelta
from pathlib import Path
from stable_baselines3 import PPO

# Add root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.body.mt5_driver import MT5Driver
from src.brain.risk_manager import RiskManager
from src.brain.feature_eng import add_features

# --- CONFIGURATION ---
SYMBOL = "XAUUSD" # Gold
TIMEFRAME = "M15" # 15-Minute Candles
MODEL_PATH = ROOT_DIR / "models" / "ppo_10M_final.zip" # Baseline v1 (for testing)
RISK_CONFIG = {
    'max_lots': 1.0,
    'daily_loss_pct': 0.05,
    'max_drawdown_pct': 0.20
}
DRY_RUN = False # Set True to test logic without sending orders

async def run_paper_trading():
    print(f"🚀 Launching Paper Trading for {SYMBOL} ({TIMEFRAME})...")
    
    # 1. Initialize Driver
    driver = MT5Driver()
    if not await driver.initialize():
        print("❌ Failed to connect to MT5. Exiting.")
        return

    # Check for Demo Account safety
    # (MT5 doesn't easily expose 'is_real' property via simple API without checking account info)
    # We proceed with caution.
    
    # 2. Initialize Risk Manager
    risk_manager = RiskManager(RISK_CONFIG)
    account_info = await driver.get_account_info()
    if account_info:
        risk_manager.update_metrics(account_info['balance'], datetime.now())
        print(f"🛡️ Risk Manager Initialized. Balance: ${account_info['balance']:.2f}")
    else:
        print("⚠️ Cound not fetch account info. Using default balance 10000.")
        risk_manager.update_metrics(10000.0, datetime.now())

    # 3. Load Model
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        return
        
    print(f"🧠 Loading Model from {MODEL_PATH}...")
    try:
        # Reconstruct Spaces for Deserialization
        # Must match TradingEnv exactly: 17 features + 2 (balance, position) = 19
        # BUT v1 Model was trained on 15 features (Old Feature Set)
        # So we must downgrade to 15 for compatibility with ppo_10M_final.zip
        num_features = 15
        custom_objects = {
            "action_space": spaces.Discrete(3),
            "observation_space": spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
        }
        
        model = PPO.load(MODEL_PATH, custom_objects=custom_objects)
        print("✅ Model Loaded Successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("📜 Traceback:")
        traceback.print_exc()
        return

    print("🟢 System Ready. Waiting for next candle...")
    
    # 4. Main Loop
    while True:
        try:
            # Sync Loop: Wait for candle close logic
            # For simplicity, we just fetch latest data. 
            # In production, we should calculate 'seconds_to_next_bar'.
            
            # Fetch Data (enough for indicators, e.g. 500)
            df = await driver.fetch_history(symbol=SYMBOL, timeframe=TIMEFRAME, count=500)
            
            if df is None or df.empty:
                print("⚠️ No data received. Retrying in 10s...")
                await asyncio.sleep(10)
                continue
            
            # Feature Engineering
            df = add_features(df)
            
            # Prepare Observation (Must match Env's observation space!)
            
            latest = df.iloc[-1]
            
            # 1. Feature Columns
            # MUT MATCH v1 Model (13 features)
            # Removed: bb_width, stoch_k, stoch_d, vwap
            feature_cols = [
                'close', 'rsi', 'macd', 'macd_signal', 
                'bb_high', 'bb_low', 'ema_20', 'ema_50',
                'atr', 
                'log_ret_lag_1', 'log_ret_lag_2', 'log_ret_lag_3', 'log_ret_lag_5'
            ]
            
            # Check if columns exist
            obs_features = [latest[col] for col in feature_cols if col in latest.index]
                
            # 2. Account State
            acc = await driver.get_account_info()
            if acc:
                balance = acc['balance']
                # Position: Need to query specific position for this symbol
                # simplified: assume single position
                positions = await driver.get_positions(symbol=SYMBOL) 
                position_units = positions[0]['volume'] if positions else 0.0
                
                # Check Circuit Breaker
                risk_manager.update_metrics(balance, datetime.now())
                status = risk_manager.get_status()
                if status['circuit_breaker']:
                    print("🚨 CIRCUIT BREAKER ACTIVE. TRADING HALTED.")
                    await asyncio.sleep(60)
                    continue
            else:
                balance = 10000
                position_units = 0.0

            obs_features.append(balance)
            obs_features.append(position_units)
            
            obs = np.array(obs_features, dtype=np.float32)
            
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            
            # Interpret Action: 0=HOLD, 1=BUY, 2=SELL
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            decision_str = action_map.get(action, "UNKNOWN")
            
            print(f"📊 {latest['timestamp']} | Price: {latest['close']:.2f} | RSI: {latest['rsi']:.1f} | Action: {decision_str}")
            
            # Risk Check & Execution
            if action == 1: # BUY
                if risk_manager.check_order(SYMBOL, 0.01, "BUY"):
                     if not DRY_RUN:
                         print("   🛒 Executing BUY 0.01 Lot...")
                         # await driver.create_order(SYMBOL, 0.01, 'BUY') 
                         # (Logic to execute needs to be robust)
                     else:
                         print("   🛒 [DRY RUN] Would BUY 0.01 Lot")
                else:
                    print("   ⛔ BUY Blocked by Risk Manager")
            
            elif action == 2: # SELL (Close)
                 if position_units > 0:
                     if not DRY_RUN:
                         print("   💰 Executing SELL (Close All)...")
                         # await driver.close_all_positions(SYMBOL)
                     else:
                         print("   💰 [DRY RUN] Would CLOSE positions")
            
            # Wait for next check
            print("💤 Waiting 1 minute...")
            await asyncio.sleep(60)

        except KeyboardInterrupt:
            print("🛑 Stopping Paper Trading...")
            break
        except Exception as e:
            print(f"⚠️ Error in loop: {e}")
            await asyncio.sleep(10)

    driver.shutdown()

if __name__ == "__main__":
    asyncio.run(run_paper_trading())
