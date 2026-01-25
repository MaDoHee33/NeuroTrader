
import asyncio
import numpy as np
import pandas as pd
from collections import deque
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
import os
from pathlib import Path
import glob

from src.brain.features import FeatureRegistry

class RLAgent:
    def __init__(self, config=None, model_path=None):
        self.config = config or {}
        self.model_path = model_path or "models/checkpoints/ppo_neurotrader.zip"
        self.model = None
        
        # Unified Feature Engine
        self.registry = FeatureRegistry()
        
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
        from stable_baselines3 import A2C, PPO, DDPG
        
        model_types = {
            'ppo': (PPO, "ppo_neurotrader.zip"),
            'a2c': (A2C, "a2c_neurotrader.zip"),
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
        
        search_paths = [
            Path(self.model_path).parent,
            Path("models/checkpoints"),
        ]
        
        available_models = []
        for search_dir in search_paths:
            if search_dir.exists():
                models = self._find_available_models(search_dir)
                if models:
                    available_models = models
                    break
        
        if available_models:
            latest_model = available_models[0]
            print(f"\n🤖 Auto-loading latest model: {latest_model.name}")
            try:
                self.model = PPO.load(str(latest_model))
                self.model_path = str(latest_model)
                print(f"✅ Model loaded successfully!")
                self.ensemble['ppo'] = self.model
                return
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        else:
            print(f"❌ No models found in any search location")
        
        print(f"⚠️  Agent will act randomly (untrained)")
        self.model = None

    def decide_action(self, observation, portfolio_state=None):
        """
        Make trading decision based on observation
        Returns: 0 (HOLD), 1 (BUY), 2 (SELL)
        """
        if self.model is None:
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
        Process incoming bar and return trading action.
        Uses UnifiedFeatureEngine for O(1) consistency.
        """
        try:
            # 1. Update Stream & Get Features (O(1))
            # Returns 1D array of features (approx 19-20 floats)
            feat_vector = self.registry.update_stream(bar_dict)
            
            # 2. Append Account State (Must match TradingEnv!)
            # Expected: [Features..., Balance, Position, HoldingTime]
            if portfolio_state is None:
                portfolio_state = {}
                
            balance = float(portfolio_state.get('balance', 10000.0))
            position = float(portfolio_state.get('position', 0.0))
            steps = float(portfolio_state.get('steps_in_position', 0))
            
            # Normalize holding time as in Env (min(steps/100, 1.0))
            holding_norm = min(steps / 100.0, 1.0)
            
            # Construct full observation
            obs = np.concatenate([
                feat_vector,
                [balance, position, holding_norm]
            ])
            
            # Ensure float32
            obs = obs.astype(np.float32)

            return self.decide_action(obs, portfolio_state), 0.0 # turbulence placeholder

        except Exception as e:
            print(f"⚠️  RLAgent: Feature error - {e}")
            return 0, 0.0  # HOLD on error
