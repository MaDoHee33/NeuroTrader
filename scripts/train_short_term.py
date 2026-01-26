
import sys
import os
import pandas as pd
import glob
from pathlib import Path
from sb3_contrib import RecurrentPPO # The LSTM Brain
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.brain.env.trading_env import TradingEnv
from src.brain.feature_eng import add_features

def train_short_term():
    print("🚀 Initializing NeuroTrader Short-Term Model Training (Top-Trend Scalper)...")
    
    # 1. Load M5 Data (Short Term)
    data_path = os.path.join(ROOT_DIR, "data/processed/*_M5_processed.parquet")
    files = glob.glob(data_path)
    
    if not files:
        # Fallback to M15 if no M5
        print("⚠️ No M5 data found, checking M15...")
        data_path = os.path.join(ROOT_DIR, "data/processed/*_M15_processed.parquet")
        files = glob.glob(data_path)
        
    if not files:
        print("❌ No Short-Term (M5/M15) data found in data/processed/")
        return

    data_map = {}
    total_rows = 0
    
    # Define Short-Term Features
    SHORT_TEAM_FEATURES = [
        'body_size', 'upper_wick', 'lower_wick', 'is_bullish',
        'dist_ema_9', 'dist_ema_21',     # Fast Trend
        'dist_ema_50',                   # Mid Trend Filter
        'rsi', 'atr_norm',
        'macd_norm', 'macd_signal_norm', 'macd_diff_norm', # Momentum
        'dist_bb_high', 'dist_bb_low', 'bb_width',         # Mean Reversion / Volatility
        'stoch_k', 'stoch_d',                              # Overbought/Oversold
        'log_ret', 'log_ret_lag_1',
        'hour_sin', 'hour_cos'           # Intraday patterns are key
    ]
    
    for f in files:
        name = os.path.basename(f)
        df = pd.read_parquet(f)
        
        # Ensure new features are present
        try:
             df = add_features(df)
        except Exception as e:
             print(f"⚠️ Error adding features to {name}: {e}")
             continue

        # Filter: Only use Data with sufficient rows
        if len(df) > 1000:
            # --- TRAIN/TEST SPLIT ---
            # Use first 80% for training
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            
            # Log range
            start_date = train_df.iloc[0]['time'] if 'time' in train_df.columns else train_df.index[0]
            end_date = train_df.iloc[-1]['time'] if 'time' in train_df.columns else train_df.index[-1]
            
            data_map[name] = train_df
            total_rows += len(train_df)
            print(f"   Loaded {name}: {len(train_df)} rows (Train Split: {start_date} - {end_date})")
    
    if not data_map:
        print("❌ No valid data loaded.")
        return
        
    print(f"📚 Total Training Data: {total_rows} rows across {len(data_map)} assets.")

    # 2. Create Environment
    def make_env():
        return TradingEnv(data_map, feature_cols=SHORT_TEAM_FEATURES)
        
    env = DummyVecEnv([make_env])

    # 3. Configure LSTM Model (NeuroTrader Short-Term)
    # Using specific RecurrentPPO parameters for shorter reaction times
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=3e-4, # Standard LR
        n_steps=512,        # Shorter Rollout (vs 2048) for faster updates
        batch_size=64,      # Smaller batch size
        gamma=0.95,         # Lower gamma (0.95 vs 0.99) to prioritize near-term rewards
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={
            "enable_critic_lstm": True,
            "lstm_hidden_size": 256, 
            "n_lstm_layers": 1,
        },
        tensorboard_log="./tensorboard_short_term/"
    )
    
    # 4. Train
    print("🏋️‍♂️ Training Started... (Target: 500,000 steps)")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path='./models/short_term_checkpoints/',
        name_prefix='neurotrader_st'
    )
    
    try:
        model.learn(total_timesteps=500000, callback=checkpoint_callback)
        
        # 5. Save Final Brain
        save_path = os.path.join(ROOT_DIR, "models/neurotrader_short_term")
        model.save(save_path)
        print(f"✅ Training Complete! Model saved to {save_path}.zip")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training Interrupted. Saving current state...")
        model.save("models/neurotrader_st_interrupted")

if __name__ == "__main__":
    train_short_term()
