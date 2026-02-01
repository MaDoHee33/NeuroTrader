"""
NeuroTrader Trinity Training Script (V2)
=========================================
Enhanced training script with:
- Checkpoint saving (resumable training)
- Model Registry integration
- Auto-backtest and evaluation
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from src.brain.env.trading_env import TradingEnv
from src.brain.features import add_features
from src.skills.model_registry import ModelRegistry


class TrainingProgressCallback(BaseCallback):
    """Custom callback for tracking training progress and saving state."""
    
    def __init__(self, save_path: str, save_freq: int = 100000, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = self.save_path / f"checkpoint_{self.n_calls}"
            self.model.save(str(checkpoint_path))
            
            # Save progress state
            state = {
                "steps_completed": self.n_calls,
                "timestamp": datetime.now().isoformat()
            }
            import json
            with open(self.save_path / "training_state.json", 'w') as f:
                json.dump(state, f)
                
            if self.verbose:
                print(f"\n[CHECKPOINT] Checkpoint saved: {checkpoint_path} ({self.n_calls:,} steps)")
        return True


def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess training data."""
    print(f"Loading data from {data_path}...")
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported format. Use .parquet or .csv")
        
    print(f"Initial shape: {df.shape}")
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', drop=False, inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', drop=False, inplace=True)
        
    df.sort_index(inplace=True)
    
    # Ensure columns are lowercase
    df.columns = df.columns.str.lower()
    
    # Handle MT5 column naming (tick_volume -> volume)
    if 'tick_volume' in df.columns and 'volume' not in df.columns:
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    
    # Validation
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing columns! Have: {df.columns.tolist()}")

    # Note: We rely on TradingEnv to compute features via FeatureRegistry
    # df = add_features(df) 
    print(f"Data columns: {df.columns.tolist()}")
        
    before_drop = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {before_drop - len(df)} rows. Final shape: {df.shape}")
    
    return df


def get_hyperparameters(role: str) -> dict:
    """Get hyperparameters for a role, checking for optimized params first."""
    
    # Default parameters per role
    defaults = {
        'scalper': {
            'n_steps': 256,
            'batch_size': 64,
            'gamma': 0.85,
            'learning_rate': 1e-4,
            'ent_coef': 0.05
        },
        'swing': {
            'n_steps': 1024,
            'batch_size': 128,
            'gamma': 0.95,
            'learning_rate': 2e-4,
            'ent_coef': 0.005
        },
        'trend': {
            'n_steps': 2048,
            'batch_size': 256,
            'gamma': 0.999,
            'learning_rate': 1e-4,
            'ent_coef': 0.001
        }
    }
    
    params = defaults.get(role, defaults['trend'])
    
    # Check for Optuna-optimized params
    param_file = ROOT_DIR / f"best_params_{role}.json"
    if param_file.exists():
        print(f"[INFO] Loading Optimized Hyperparameters from {param_file}")
        import json
        with open(param_file, 'r') as f:
            optimized = json.load(f)
            params.update(optimized)
            print(f"   -> Gamma: {params['gamma']:.4f}, LR: {params['learning_rate']:.6f}")
    
    return params


def find_checkpoint(checkpoint_dir: Path) -> tuple:
    """Find latest checkpoint in directory."""
    if not checkpoint_dir.exists():
        return None, 0
    
    state_file = checkpoint_dir / "training_state.json"
    if not state_file.exists():
        return None, 0
    
    import json
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    steps = state.get("steps_completed", 0)
    checkpoint_path = checkpoint_dir / f"checkpoint_{steps}.zip"
    
    if checkpoint_path.exists():
        return str(checkpoint_path), steps
    
    # Try without .zip
    checkpoint_path = checkpoint_dir / f"checkpoint_{steps}"
    if Path(str(checkpoint_path) + ".zip").exists():
        return str(checkpoint_path), steps
        
    return None, 0


def cleanup_checkpoints(checkpoint_dir: Path):
    """
    Remove checkpoint files after successful training.
    Keeps the directory but removes all checkpoint files and state.
    """
    import shutil
    
    if not checkpoint_dir.exists():
        return
    
    print(f"\n[CLEANUP] Cleaning up checkpoints...")
    
    files_removed = 0
    for item in checkpoint_dir.iterdir():
        try:
            if item.is_file():
                item.unlink()
                files_removed += 1
            elif item.is_dir():
                shutil.rmtree(item)
                files_removed += 1
        except Exception as e:
            print(f"   Warning: Could not remove {item.name}: {e}")
    
    print(f"   [SUCCESS] Removed {files_removed} checkpoint file(s)")
    print(f"   [INFO] Checkpoints are only kept if training fails or is interrupted")


def extract_symbol_timeframe(data_path: str) -> tuple:
    """Extract symbol and timeframe from data filename."""
    basename = os.path.basename(data_path).replace('_processed', '').replace('.parquet', '').replace('.csv', '')
    parts = basename.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return basename, 'unknown'


def train_trinity(
    role: str,
    data_path: str,
    total_timesteps: int = 1000000,
    resume: bool = False,
    register: bool = True,
    checkpoint_freq: int = 100000,
    reward_config: dict = None,
    suffix: str = ""
):
    """
    Train a Trinity agent with checkpointing and registry integration.
    
    Args:
        role: Agent role (scalper/swing/trend)
        data_path: Path to training data
        total_timesteps: Total training steps
        resume: Resume from checkpoint if available
        register: Register model in registry after training
        checkpoint_freq: Save checkpoint every N steps
        reward_config: Dictionary of reward parameters (optional)
        suffix: Optional suffix for model and checkpoint names
    """
    role = role.lower()
    symbol, timeframe = extract_symbol_timeframe(data_path)
    
    print(f"\n{'='*60}")
    print(f"[START] NEUROTRADER TRINITY TRAINING PROTOCOL")
    print(f"{'='*60}")
    print(f"Role      : {role.upper()}")
    print(f"Symbol    : {symbol}")
    print(f"Timeframe : {timeframe}")
    print(f"Data      : {data_path}")
    print(f"Steps     : {total_timesteps:,}")
    print(f"{'='*60}\n")
    
    # Get hyperparameters
    params = get_hyperparameters(role)
    
    # Load and split data
    df = load_data(data_path)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    print(f"Training samples: {len(train_df):,}")
    
    # Create environment
    env_kwargs = {'agent_type': role}
    if reward_config:
        env_kwargs['reward_config'] = reward_config
        print(f"[CONFIG] Using Custom Reward Config: {reward_config}")
        
    env = DummyVecEnv([lambda: TradingEnv(train_df, **env_kwargs)])
    
    # Checkpoint directory
    model_name_base = f"{role}_{symbol}_{timeframe}"
    if suffix:
        model_name_base += f"_{suffix}"
        print(f"[INFO]  Using Model Suffix: {suffix}")
    
    checkpoint_dir = ROOT_DIR / "models" / "checkpoints" / model_name_base
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint
    start_steps = 0
    model = None
    
    if resume:
        checkpoint_path, start_steps = find_checkpoint(checkpoint_dir)
        if checkpoint_path:
            print(f"[INFO] Found checkpoint at {start_steps:,} steps")
            print(f"   Loading: {checkpoint_path}")
            model = RecurrentPPO.load(checkpoint_path, env=env)
            print(f"[SUCCESS] Resumed from checkpoint!")
    
    # Create new model if not resuming
    if model is None:
        tensorboard_log = str(ROOT_DIR / "logs" / "trinity" / role)
        os.makedirs(tensorboard_log, exist_ok=True)
        
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            gamma=params['gamma'],
            gae_lambda=0.95,
            ent_coef=params['ent_coef'],
            tensorboard_log=tensorboard_log
        )
    
    # Setup callbacks
    checkpoint_callback = TrainingProgressCallback(
        save_path=str(checkpoint_dir),
        save_freq=checkpoint_freq
    )
    
    # Calculate remaining steps
    remaining_steps = max(0, total_timesteps - start_steps)
    print(f"\n[RUNNING] Training {remaining_steps:,} steps...")
    
    # Final model path
    model_name = f"trinity_{role}_{symbol}_{timeframe}"
    final_model_path = str(ROOT_DIR / "models" / model_name)
    
    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=(start_steps == 0)
        )
        model.save(final_model_path)
        print(f"\n[SUCCESS] Training Complete!")
        print(f"[FILE] Model saved to: {final_model_path}.zip")
        
        # Register in Model Registry
        if register:
            print(f"\n[INFO] Registering model...")
            registry = ModelRegistry(str(ROOT_DIR / "models"))
            
            # Run quick evaluation for metrics (with custom config)
            # Pass reward config to quick_evaluate if needed
            metrics = quick_evaluate(model, df.iloc[train_size:], role, reward_config)
            
            metadata = registry.register_model(
                model_path=f"{final_model_path}.zip",
                role=role,
                symbol=symbol,
                timeframe=timeframe,
                training_steps=total_timesteps,
                training_config=params,
                metrics=metrics,
                data_path=data_path,
                tags=[timeframe, symbol, "auto_trained"] + ([suffix] if suffix else [])
            )
            
            # Auto-promote if better
            primary_metric = {
                'scalper': 'avg_holding_time',
                'swing': 'sharpe_ratio',
                'trend': 'total_return'
            }.get(role, 'total_return')
            
            higher_is_better = role != 'scalper'  # For scalper, lower holding time is better
            
            registry.auto_promote_if_better(
                role=role,
                new_version=metadata.version,
                primary_metric=primary_metric,
                higher_is_better=higher_is_better
            )
        
        # Cleanup checkpoints after successful training
        cleanup_checkpoints(checkpoint_dir)
            
    except KeyboardInterrupt:
        print(f"\n[WARNING] Training interrupted!")
        print(f"   Progress saved to: {checkpoint_dir}")
        print(f"   Resume with: --resume flag")
        # Keep checkpoints for resume
        
    except Exception as e:
        print(f"\n[FAILED] Training failed: {e}")
        import traceback
        traceback.print_exc()
        # Keep checkpoints for debugging/resume


def quick_evaluate(model, test_df: pd.DataFrame, role: str, reward_config: dict = None) -> dict:
    """Quick evaluation on test set for metrics."""
    from src.analysis.behavior import calculate_behavioral_metrics
    
    env_kwargs = {'agent_type': role}
    if reward_config:
        env_kwargs['reward_config'] = reward_config
        
    env = TradingEnv(test_df, **env_kwargs)
    obs, _ = env.reset()
    done = False
    history = []
    lstm_states = None
    
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        history.append({
            'step': info['step'],
            'equity': info['equity'],
            'position': env.position
        })
        if done or truncated:
            break
    
    df_res = pd.DataFrame(history)
    
    # Calculate metrics
    initial = df_res['equity'].iloc[0]
    final = df_res['equity'].iloc[-1]
    total_return = (final - initial) / initial * 100
    
    peak = df_res['equity'].cummax()
    dd = (df_res['equity'] - peak) / peak * 100
    max_dd = dd.min()
    
    # Behavioral metrics
    beh = calculate_behavioral_metrics(df_res)
    
    return {
        'total_return': total_return,
        'max_drawdown': max_dd,
        'avg_holding_time': beh.get('avg_holding_time_steps', 0),
        'win_rate': beh.get('win_rate_pct', 0),
        'profit_factor': beh.get('profit_factor', 0),
        'total_trades': beh.get('total_trades_approx', 0)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroTrader Trinity Training (V2)")
    parser.add_argument('--role', type=str, required=True, 
                       choices=['scalper', 'swing', 'trend'],
                       help="Agent role")
    parser.add_argument('--data', type=str, required=True,
                       help="Path to training data")
    parser.add_argument('--steps', type=int, default=1000000,
                       help="Total timesteps")
    parser.add_argument('--resume', action='store_true',
                       help="Resume from checkpoint")
    parser.add_argument('--no-register', action='store_true',
                       help="Skip model registry")
    parser.add_argument('--checkpoint-freq', type=int, default=100000,
                       help="Checkpoint frequency (steps)")
    
    # Experiment Arguments
    parser.add_argument("--suffix", type=str, default="", help="Model name suffix (e.g. 'experiment1')")
    parser.add_argument("--max_steps_holding", type=int, default=None, help="Max steps to hold position")
    parser.add_argument("--sniper_start", type=int, default=None, help="Steps before sniper penalty starts")
    parser.add_argument("--sniper_amt", type=float, default=None, help="Sniper penalty amount")
    parser.add_argument("--force_exit_penalty", type=float, default=None, help="Force exit penalty")
    
    args = parser.parse_args()
    
    # Construct reward config
    reward_config = {}
    if args.max_steps_holding: reward_config['max_holding_steps'] = args.max_steps_holding
    if args.sniper_start: reward_config['sniper_penalty_start'] = args.sniper_start
    if args.sniper_amt: reward_config['sniper_penalty_amt'] = args.sniper_amt
    if args.force_exit_penalty: reward_config['force_exit_penalty'] = args.force_exit_penalty

    train_trinity(
        role=args.role,
        data_path=args.data,
        total_timesteps=args.steps,
        resume=args.resume,
        register=not args.no_register,
        checkpoint_freq=args.checkpoint_freq,
        reward_config=reward_config if reward_config else None,
        suffix=args.suffix
    )
    
    if args.role == 'scalper' and 'M5' not in args.data and 'M1' not in args.data:
        print("[WARNING] WARNING: Scalpers should preferably trade on M1/M5 data.")
    

