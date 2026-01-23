
import asyncio
import numpy as np
import pandas as pd
from collections import deque
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
import os
from pathlib import Path
import glob

from src.brain.feature_eng import add_features

class RLAgent:
    def __init__(self, config=None, model_path=None):
        self.config = config or {}
        self.model_path = model_path or "models/checkpoints/ppo_neurotrader.zip"
        self.model = None
        
        # History buffer for feature engineering
        self.history_size = 100 
        self.history = deque(maxlen=self.history_size)
        
        # Ensemble Support
        self.ensemble = {}
        self.active_model_name = "ppo" # Default
        self._load_ensemble_models()
        self._load_model() # Legacy load for back compat or fallback
    
    def _find_available_models(self, search_dir):
        """Find all available model files in directory"""
        search_path = Path(search_dir)
        if not search_path.exists():
            return []
        
        # Search for .zip files
        models = list(search_path.glob("*.zip"))
        # Sort by modification time (newest first)
        models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return models
    
    def _load_ensemble_models(self):
        """Attempts to load PPO, A2C, and other models for ensemble."""
        print("🔍 RLAgent: Initializing Ensemble...")
        
        # Hypothetical paths - in real deploy they would be distinct files
        # For now, we look for 'ppo_neurotrader.zip', 'a2c_neurotrader.zip'
        from stable_baselines3 import A2C, PPO, DDPG
        
        model_types = {
            'ppo': (PPO, "ppo_neurotrader.zip"),
            'a2c': (A2C, "a2c_neurotrader.zip"),
            # 'ddpg': (DDPG, "ddpg_neurotrader.zip") 
        }
        
        for name, (cls, filename) in model_types.items():
            path = Path("models/checkpoints") / filename
            if path.exists():
                try:
                    self.ensemble[name] = cls.load(path)
                    print(f"   ✅ Loaded Ensemble Agent: {name.upper()}")
                except Exception as e:
                    print(f"   ❌ Failed to load {name}: {e}")
        
    def _load_model(self):
        """Smart model loading with auto-discovery (Primary/Fallback)"""
        # If we have ensemble models, pick one as primary
        if self.ensemble:
             self.model = self.ensemble.get('ppo') or list(self.ensemble.values())[0]
             print(f"🤖 Active Agent set to: {type(self.model).__name__}")
             return

        print(f"🔍 RLAgent: Looking for single model at {self.model_path}")
        
        if os.path.exists(self.model_path):
            try:
                print(f"✅ RLAgent: Loading model from {self.model_path}")
                self.model = PPO.load(self.model_path)
                print(f"🧠 RLAgent: Model loaded successfully!")
                return
            except Exception as e:
                print(f"❌ RLAgent: Error loading model: {e}")
        
        # Model not found - try smart discovery in multiple locations
        print(f"⚠️  RLAgent: Model not found at {self.model_path}")
        
        # Search paths (in order of priority)
        search_paths = [
            Path(self.model_path).parent,  # Original path
            Path("/content/drive/MyDrive/NeuroTrader_Workspace/models/checkpoints"),  # Colab Drive
            Path("models/checkpoints"),  # Local relative
        ]
        
        available_models = []
        for search_dir in search_paths:
            if search_dir.exists():
                models = self._find_available_models(search_dir)
                if models:
                    available_models = models
                    print(f"💡 Found {len(models)} model(s) in {search_dir}:")
                    for i, model in enumerate(models[:5], 1):
                        size_mb = model.stat().st_size / (1024 * 1024)
                        print(f"   {i}. {model.name} ({size_mb:.1f} MB)")
                    break
        
        if available_models:
            # Auto-load latest
            latest_model = available_models[0]
            print(f"\n🤖 Auto-loading latest model: {latest_model.name}")
            try:
                self.model = PPO.load(str(latest_model))
                self.model_path = str(latest_model)
                print(f"✅ Model loaded successfully!")
                # Add to ensemble as PPO default
                self.ensemble['ppo'] = self.model
                return
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        else:
            print(f"❌ No models found in any search location")
            print(f"💡 Tip: Train a model first with: python -m src.brain.train")
        
        print(f"⚠️  Agent will act randomly (untrained)")
        self.model = None

    def update_ensemble_strategy(self, recent_performance_metric):
        """
        Selector Logic: Switch active model based on performance.
        (Placeholder for full implementation which needs backtesting engine here)
        """
        # In a real impl, we would evaluate all self.ensemble models on recent 
        # data history and pick the winner.
        pass

    def decide_action(self, observation, portfolio_state=None):
        """
        Make trading decision based on observation
        Returns: 0 (HOLD), 1 (BUY), 2 (SELL)
        """
        if self.model is None:
            # Random fallback if no model
            return np.random.choice([0, 1, 2])
        
        try:
            # Use the trained model
            action, _ = self.model.predict(observation, deterministic=True)
            return int(action)
        except Exception as e:
            print(f"❌ RLAgent: Error in prediction: {e}")
            return 0  # HOLD on error

    def process_bar(self, bar_dict, portfolio_state=None):
        """
        Process incoming bar and return trading action
        """
        # Add to history
        self.history.append(bar_dict)
        
        # Need enough history for features
        if len(self.history) < 60:  # Warmup period
            return 0  # HOLD during warmup
        
        # Convert history to DataFrame
        df = pd.DataFrame(list(self.history))
        
        # Feature engineering
        try:
            df_features = add_features(df)
            
            # Get latest observation
            if len(df_features) > 0:
                latest = df_features.iloc[-1]
                
                # CRITICAL: Must match training env observation space (19 features for V2)
                # Training env uses: 8 technical + 7 price features + 4 new features = 19 total
                feature_cols = [
                    'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                    'ema_20', 'atr', 'log_return',  # 8 technical
                    'close', 'open', 'high', 'low', 'volume',  # 5 price
                    'normalized_close', 'normalized_volume',  # 2 normalized
                    'stoch_k', 'stoch_d', 'vwap', 'bb_width' # 4 new features
                ]
                
                # Build observation vector
                obs_values = []
                for col in feature_cols:
                    if col in latest.index:
                        obs_values.append(float(latest[col]))
                    elif col == 'normalized_close':
                        # Fallback: normalize close
                        obs_values.append(float(latest.get('close', 0)) / 2000.0)
                    elif col == 'normalized_volume':
                        # Fallback: normalize volume
                        obs_values.append(float(latest.get('volume', 0)) / 100000.0)
                    else:
                        obs_values.append(0.0)  # Missing feature
                
                obs = np.array(obs_values, dtype=np.float32)
                
                # Ensure exactly 19 features
                if len(obs) < 19:
                    obs = np.pad(obs, (0, 19 - len(obs)), 'constant')
                elif len(obs) > 19:
                    obs = obs[:19]
                
                # Retrieve turbulence
                turbulence_val = float(latest.get('turbulence', 0.0))

                return self.decide_action(obs, portfolio_state), turbulence_val
            else:
                return 0, 0.0
        except Exception as e:
            # Log error but don't crash
            print(f"⚠️  RLAgent: Feature error - {e}")
            return 0, 0.0  # HOLD on error
