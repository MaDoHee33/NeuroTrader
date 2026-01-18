
import asyncio
import numpy as np
import pandas as pd
from collections import deque
from stable_baselines3 import PPO
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
        
        self._load_model()
    
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
    
    def _load_model(self):
        """Smart model loading with auto-discovery"""
        print(f"🔍 RLAgent: Looking for model at {self.model_path}")
        
        if os.path.exists(self.model_path):
            try:
                print(f"✅ RLAgent: Loading model from {self.model_path}")
                self.model = PPO.load(self.model_path)
                print(f"🧠 RLAgent: Model loaded successfully!")
                return
            except Exception as e:
                print(f"❌ RLAgent: Error loading model: {e}")
        
        # Model not found - try smart discovery
        print(f"⚠️  RLAgent: Model not found at {self.model_path}")
        
        # Search in checkpoints directory
        model_dir = Path(self.model_path).parent
        available_models = self._find_available_models(model_dir)
        
        if available_models:
            print(f"\n💡 Found {len(available_models)} model(s) in {model_dir}:")
            for i, model in enumerate(available_models[:5], 1):  # Show max 5
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"   {i}. {model.name} ({size_mb:.1f} MB)")
            
            # Auto-load latest
            latest_model = available_models[0]
            print(f"\n🤖 Auto-loading latest model: {latest_model.name}")
            try:
                self.model = PPO.load(str(latest_model))
                self.model_path = str(latest_model)  # Update path
                print(f"✅ Model loaded successfully!")
                return
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        else:
            print(f"❌ No models found in {model_dir}")
            print(f"💡 Tip: Train a model first with: python -m src.brain.train")
        
        print(f"⚠️  Agent will act randomly (untrained)")
        self.model = None

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
                
                # Create observation vector (matching training env)
                obs = latest[['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 
                             'ema_20', 'atr', 'log_return']].values.astype(np.float32)
                
                # Add portfolio state if provided
                if portfolio_state:
                    balance = portfolio_state.get('balance', 10000.0)
                    position = portfolio_state.get('position', 0.0)
                    obs = np.append(obs, [balance / 10000.0, position])  # Normalize
                
                return self.decide_action(obs, portfolio_state)
            else:
                return 0  # HOLD if feature engineering fails
        except Exception as e:
            print(f"❌ RLAgent: Error processing bar: {e}")
            return 0  # HOLD on error
