"""
Hybrid Trading Agent
=====================
Combines PPO baseline with Self-Evolving AI components.

This is the integration point between:
1. Traditional PPO model (for immediate trading decisions)
2. Curiosity Module (for intrinsic motivation)
3. Experience Buffer (for lifelong learning)
4. Curriculum Manager (for progressive difficulty)

Architecture:
```
                    ┌─────────────────────┐
                    │   Market Data       │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   TradingEnv        │
                    └─────────┬───────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  PPO Agent    │   │ Curiosity Module│   │Experience Buffer│
│  (Baseline)   │   │ (Exploration)   │   │  (Memory)       │
└───────┬───────┘   └────────┬────────┘   └────────┬────────┘
        │                    │                     │
        └────────────────────┼─────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Hybrid Agent   │
                    │  (This Module)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Trading Action │
                    └─────────────────┘
```
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import json

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from .curiosity import CuriosityModule, IntrinsicReward
from .experience_buffer import ExperienceBuffer, TradeExperience
from .difficulty_scaler import CurriculumManager, DifficultyLevel


class HybridTradingAgent:
    """
    Hybrid Trading Agent combining PPO with Self-Evolving capabilities.
    
    This agent:
    1. Uses PPO for action decisions (proven baseline)
    2. Adds curiosity rewards for exploration
    3. Stores experiences for continuous learning
    4. Adapts difficulty based on performance
    
    Usage:
        agent = HybridTradingAgent(ppo_model_path='models/scalper.zip')
        obs = env.reset()
        
        while not done:
            action, info = agent.get_action(obs)
            next_obs, reward, done, _, env_info = env.step(action)
            agent.store_experience(obs, action, reward, next_obs, env_info)
            obs = next_obs
        
        agent.end_episode(total_return=...)
    """
    
    def __init__(
        self,
        ppo_model_path: Optional[str] = None,
        use_curiosity: bool = True,
        use_experience_buffer: bool = True,
        use_curriculum: bool = True,
        curiosity_weight: float = 0.1,  # How much curiosity affects decisions
        data_dir: Optional[Path] = None,
        agent_name: str = "hybrid_agent"
    ):
        """
        Initialize Hybrid Trading Agent.
        
        Args:
            ppo_model_path: Path to trained PPO model (None = random agent)
            use_curiosity: Enable curiosity-driven exploration
            use_experience_buffer: Enable experience storage
            use_curriculum: Enable curriculum learning
            curiosity_weight: Weight for curiosity in reward (0-1)
            data_dir: Directory for persistence
            agent_name: Name for logging and persistence
        """
        self.agent_name = agent_name
        self.curiosity_weight = curiosity_weight
        
        # Setup directories
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path("data/hybrid_agent")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load PPO model
        self.ppo_model = None
        self.lstm_states = None
        if ppo_model_path and Path(ppo_model_path).exists():
            try:
                # Try RecurrentPPO first (LSTM)
                self.ppo_model = RecurrentPPO.load(ppo_model_path)
                self.is_recurrent = True
                print(f"[HYBRID] Loaded RecurrentPPO model from {ppo_model_path}")
            except Exception:
                try:
                    self.ppo_model = PPO.load(ppo_model_path)
                    self.is_recurrent = False
                    print(f"[HYBRID] Loaded PPO model from {ppo_model_path}")
                except Exception as e:
                    print(f"[HYBRID] Warning: Could not load model: {e}")
                    self.is_recurrent = False
        else:
            self.is_recurrent = False
            print(f"[HYBRID] No model loaded - using random actions")
        
        # Initialize components
        self.curiosity = None
        if use_curiosity:
            self.curiosity = CuriosityModule(
                novelty_weight=0.3,
                prediction_weight=0.4,
                pattern_weight=0.3
            )
            print("[HYBRID] Curiosity Module enabled")
        
        self.experience_buffer = None
        if use_experience_buffer:
            self.experience_buffer = ExperienceBuffer(
                max_size=50000,
                save_path=self.data_dir / f"{agent_name}_experiences.json",
                auto_save_interval=500
            )
            print(f"[HYBRID] Experience Buffer enabled ({len(self.experience_buffer.experiences)} loaded)")
        
        self.curriculum = None
        if use_curriculum:
            self.curriculum = CurriculumManager(
                start_level=DifficultyLevel.EASY,
                allow_regression=True
            )
            # Load saved curriculum state
            curriculum_path = self.data_dir / f"{agent_name}_curriculum.json"
            if curriculum_path.exists():
                with open(curriculum_path, 'r') as f:
                    self.curriculum.load_state(json.load(f))
            print(f"[HYBRID] Curriculum enabled (Level: {self.curriculum.get_current_level().name})")
        
        # Episode tracking
        self.episode_id = ""
        self.episode_step = 0
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_trades = []
        
        # Statistics
        self.stats = {
            'total_steps': 0,
            'total_episodes': 0,
            'total_intrinsic_reward': 0.0,
            'action_distribution': {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL
        }
    
    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, Dict]:
        """
        Get trading action for current observation.
        
        Args:
            observation: Current market observation
            deterministic: Use deterministic policy (no exploration)
            
        Returns:
            Tuple of (action, info_dict)
        """
        info = {
            'source': 'random',
            'confidence': 0.0,
            'curiosity_score': 0.0
        }
        
        # Get action from PPO model
        if self.ppo_model is not None:
            if self.is_recurrent:
                action, self.lstm_states = self.ppo_model.predict(
                    observation,
                    state=self.lstm_states,
                    deterministic=deterministic
                )
            else:
                action, _ = self.ppo_model.predict(observation, deterministic=deterministic)
            
            info['source'] = 'ppo'
            
            # Get action probabilities for confidence
            # Note: This is simplified - full implementation would use policy distribution
            info['confidence'] = 0.7 if deterministic else 0.5
        else:
            # Random action
            action = np.random.randint(0, 3)
            info['confidence'] = 0.33
        
        # Add curiosity influence on exploration
        if self.curiosity and not deterministic:
            curiosity_score = self.curiosity.get_curiosity_score()
            info['curiosity_score'] = curiosity_score
            
            # Higher curiosity = more likely to try different actions
            if np.random.random() < curiosity_score * self.curiosity_weight:
                # Explore: try a different action
                action = np.random.randint(0, 3)
                info['source'] = 'curiosity_exploration'
        
        # Track action
        self.episode_actions.append(action)
        self.stats['action_distribution'][int(action)] += 1
        
        return int(action), info
    
    def store_experience(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        env_info: Optional[Dict] = None
    ) -> Optional[IntrinsicReward]:
        """
        Store experience and compute intrinsic reward.
        
        Args:
            observation: State before action
            action: Action taken
            reward: Extrinsic reward from environment
            next_observation: State after action
            env_info: Additional info from environment
            
        Returns:
            IntrinsicReward if curiosity is enabled, else None
        """
        self.episode_step += 1
        self.stats['total_steps'] += 1
        
        env_info = env_info or {}
        intrinsic_reward = None
        
        # Compute intrinsic reward if curiosity enabled
        if self.curiosity:
            intrinsic_reward = self.curiosity.compute_intrinsic_reward(
                observation, action, next_observation, reward
            )
            self.stats['total_intrinsic_reward'] += intrinsic_reward.total
        
        # Store in experience buffer
        if self.experience_buffer:
            # Extract trade info
            trade_info = env_info.get('last_trade', {})
            pnl = env_info.get('pnl', 0.0)
            holding_time = env_info.get('holding_time', 0)
            
            # Determine market regime (simplified)
            market_regime = self._detect_regime(observation)
            
            # Generate lesson tags
            lesson_tags = []
            if trade_info:
                if trade_info.get('action') == 'SELL' and pnl > 0:
                    lesson_tags.append('profitable_exit')
                if holding_time < 12:  # < 1 hour on M5
                    lesson_tags.append('quick_trade')
            
            self.experience_buffer.add(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                pnl=pnl,
                holding_time=holding_time,
                market_regime=market_regime,
                confidence=0.5,  # Could be from model
                episode_id=self.episode_id,
                lesson_tags=lesson_tags
            )
            
            # Track trades for episode summary
            if trade_info:
                self.episode_trades.append({
                    'step': self.episode_step,
                    'action': trade_info.get('action'),
                    'pnl': pnl
                })
        
        self.episode_rewards.append(reward)
        
        return intrinsic_reward
    
    def _detect_regime(self, observation: np.ndarray) -> str:
        """
        Simple market regime detection from observation.
        
        This is a placeholder - could be replaced with ML-based detection.
        """
        # Assuming observation contains RSI, trend indicators, etc.
        # This is a simplified heuristic
        if len(observation) < 5:
            return 'unknown'
        
        # Use observation features as proxy
        # Typical observation: [rsi, macd, bb_width, atr, ...]
        # Higher values in certain positions suggest volatile/trending
        
        volatility_proxy = abs(observation[3]) if len(observation) > 3 else 0.5
        trend_proxy = observation[0] if len(observation) > 0 else 0.5
        
        if volatility_proxy > 0.7:
            return 'volatile'
        elif trend_proxy > 0.6:
            return 'bull'
        elif trend_proxy < 0.4:
            return 'bear'
        else:
            return 'sideways'
    
    def start_episode(self):
        """Start a new trading episode."""
        import uuid
        self.episode_id = str(uuid.uuid4())[:8]
        self.episode_step = 0
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_trades = []
        
        # Reset LSTM states
        self.lstm_states = None
        
        # Reset curiosity per-episode tracking
        if self.curiosity:
            self.curiosity.reset_episode()
    
    def end_episode(
        self,
        total_return: float,
        win_rate: float = 0.0,
        num_trades: int = 0
    ) -> Dict:
        """
        End current episode and update curriculum.
        
        Args:
            total_return: Episode total return (%)
            win_rate: Episode win rate (0-1)
            num_trades: Number of trades
            
        Returns:
            Episode summary dict
        """
        self.stats['total_episodes'] += 1
        
        summary = {
            'episode_id': self.episode_id,
            'total_steps': self.episode_step,
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'action_counts': {
                'hold': self.episode_actions.count(0),
                'buy': self.episode_actions.count(1),
                'sell': self.episode_actions.count(2)
            }
        }
        
        # Update curriculum
        if self.curriculum:
            curriculum_result = self.curriculum.record_episode(
                total_return=total_return,
                win_rate=win_rate,
                num_trades=num_trades
            )
            summary['curriculum'] = curriculum_result
            
            # Save curriculum state
            curriculum_path = self.data_dir / f"{self.agent_name}_curriculum.json"
            with open(curriculum_path, 'w') as f:
                json.dump(self.curriculum.save_state(), f)
        
        # Add curiosity stats
        if self.curiosity:
            summary['curiosity'] = self.curiosity.get_stats()
        
        # Save experience buffer periodically
        if self.experience_buffer:
            self.experience_buffer.save()
            summary['buffer_size'] = len(self.experience_buffer.experiences)
        
        return summary
    
    def get_similar_past_experiences(
        self,
        observation: np.ndarray,
        top_k: int = 5
    ) -> list:
        """
        Get similar past experiences for few-shot learning.
        
        Useful for: "What did I do in similar situations?"
        """
        if not self.experience_buffer:
            return []
        
        return self.experience_buffer.get_similar_experiences(observation, top_k)
    
    def get_stats(self) -> Dict:
        """Get comprehensive agent statistics."""
        stats = {**self.stats}
        
        if self.curiosity:
            stats['curiosity'] = self.curiosity.get_stats()
        
        if self.experience_buffer:
            stats['experience_buffer'] = self.experience_buffer.get_stats()
        
        if self.curriculum:
            stats['curriculum'] = self.curriculum.get_progress_report()
        
        return stats
    
    def save(self):
        """Save all agent state."""
        # Save experience buffer
        if self.experience_buffer:
            self.experience_buffer.save()
        
        # Save curiosity state
        if self.curiosity:
            curiosity_path = self.data_dir / f"{self.agent_name}_curiosity.json"
            with open(curiosity_path, 'w') as f:
                json.dump(self.curiosity.save_state(), f)
        
        # Save curriculum state
        if self.curriculum:
            curriculum_path = self.data_dir / f"{self.agent_name}_curriculum.json"
            with open(curriculum_path, 'w') as f:
                json.dump(self.curriculum.save_state(), f)
        
        # Save stats
        stats_path = self.data_dir / f"{self.agent_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f)
        
        print(f"[HYBRID] Agent state saved to {self.data_dir}")
    
    def load(self):
        """Load all agent state."""
        # Load curiosity state
        if self.curiosity:
            curiosity_path = self.data_dir / f"{self.agent_name}_curiosity.json"
            if curiosity_path.exists():
                with open(curiosity_path, 'r') as f:
                    self.curiosity.load_state(json.load(f))
        
        # Load curriculum state
        if self.curriculum:
            curriculum_path = self.data_dir / f"{self.agent_name}_curriculum.json"
            if curriculum_path.exists():
                with open(curriculum_path, 'r') as f:
                    self.curriculum.load_state(json.load(f))
        
        # Load stats
        stats_path = self.data_dir / f"{self.agent_name}_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.stats.update(json.load(f))
        
        print(f"[HYBRID] Agent state loaded from {self.data_dir}")
