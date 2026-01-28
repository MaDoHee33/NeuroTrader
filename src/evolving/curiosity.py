"""
Curiosity-Driven Exploration Module
====================================
Implements Intrinsic Curiosity Module (ICM) for self-motivated learning.

This module provides intrinsic rewards based on:
1. Novelty: How new/unexpected is the current state?
2. Prediction Error: How well can we predict the next state?
3. Pattern Discovery: Did we find a new profitable pattern?

Reference: Pathak et al. "Curiosity-driven Exploration by Self-Supervised Prediction"
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque
from dataclasses import dataclass
import hashlib


@dataclass
class IntrinsicReward:
    """Container for intrinsic reward components."""
    novelty_bonus: float = 0.0
    prediction_error: float = 0.0
    pattern_discovery: float = 0.0
    
    @property
    def total(self) -> float:
        """Calculate total intrinsic reward."""
        return self.novelty_bonus + self.prediction_error + self.pattern_discovery
    
    def to_dict(self) -> Dict:
        return {
            'novelty_bonus': self.novelty_bonus,
            'prediction_error': self.prediction_error,
            'pattern_discovery': self.pattern_discovery,
            'total': self.total
        }


class StateEncoder:
    """
    Encodes market states into a compact representation for novelty detection.
    Uses a simple feature-based encoding (can be upgraded to neural network later).
    """
    
    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
        self.feature_ranges = {}
    
    def encode(self, observation: np.ndarray) -> str:
        """Encode observation into a hashable string representation."""
        # Discretize continuous features into bins
        discretized = []
        for i, val in enumerate(observation):
            if np.isnan(val) or np.isinf(val):
                discretized.append(0)
            else:
                # Simple binning: normalize to 0-num_bins range
                bin_idx = int(np.clip(val * self.num_bins, 0, self.num_bins - 1))
                discretized.append(bin_idx)
        
        # Create hash for efficient lookup
        state_str = ",".join(map(str, discretized))
        return hashlib.md5(state_str.encode()).hexdigest()[:16]
    
    def similarity(self, obs1: np.ndarray, obs2: np.ndarray) -> float:
        """Calculate similarity between two observations (0 to 1)."""
        if len(obs1) != len(obs2):
            return 0.0
        
        # Cosine similarity
        norm1 = np.linalg.norm(obs1)
        norm2 = np.linalg.norm(obs2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(obs1, obs2) / (norm1 * norm2)


class CuriosityModule:
    """
    Curiosity-Driven Exploration Module
    
    Provides intrinsic rewards to encourage exploration of novel states
    and discovery of new trading patterns.
    
    Features:
    - State novelty detection (count-based)
    - State prediction model (simple linear for low resource usage)
    - Pattern discovery tracking
    
    Low Resource Design:
    - Uses hash-based state counting (O(1) lookup)
    - Simple linear prediction (no neural networks by default)
    - Fixed-size memory buffers
    """
    
    def __init__(
        self,
        novelty_weight: float = 0.3,
        prediction_weight: float = 0.4,
        pattern_weight: float = 0.3,
        memory_size: int = 10000,
        novelty_threshold: float = 0.1,
        prediction_lr: float = 0.01
    ):
        """
        Initialize Curiosity Module.
        
        Args:
            novelty_weight: Weight for novelty bonus in total reward
            prediction_weight: Weight for prediction error bonus
            pattern_weight: Weight for pattern discovery bonus
            memory_size: Max number of states to remember
            novelty_threshold: Threshold for considering state as "novel"
            prediction_lr: Learning rate for prediction model
        """
        self.novelty_weight = novelty_weight
        self.prediction_weight = prediction_weight
        self.pattern_weight = pattern_weight
        self.memory_size = memory_size
        self.novelty_threshold = novelty_threshold
        self.prediction_lr = prediction_lr
        
        # State memory (count-based novelty)
        self.state_counts: Dict[str, int] = {}
        self.total_states_seen = 0
        
        # State encoder
        self.encoder = StateEncoder()
        
        # Recent states buffer for prediction
        self.recent_states = deque(maxlen=100)
        self.recent_actions = deque(maxlen=100)
        
        # Pattern discovery tracking
        self.discovered_patterns: Dict[str, Dict] = {}
        self.pattern_rewards: List[float] = []
        
        # Simple linear prediction model weights
        self.prediction_weights: Optional[np.ndarray] = None
        self.input_dim: Optional[int] = None
        
        # Statistics
        self.stats = {
            'total_intrinsic_reward': 0.0,
            'avg_novelty': 0.0,
            'unique_states': 0,
            'patterns_found': 0
        }
    
    def compute_intrinsic_reward(
        self,
        observation: np.ndarray,
        action: int,
        next_observation: np.ndarray,
        extrinsic_reward: float
    ) -> IntrinsicReward:
        """
        Compute intrinsic reward for a state transition.
        
        Args:
            observation: Current state
            action: Action taken
            next_observation: Resulting state
            extrinsic_reward: External reward from environment
            
        Returns:
            IntrinsicReward object with all components
        """
        # 1. Novelty Bonus
        novelty = self._compute_novelty(next_observation)
        
        # 2. Prediction Error Bonus
        prediction_error = self._compute_prediction_error(
            observation, action, next_observation
        )
        
        # 3. Pattern Discovery Bonus
        pattern = self._check_pattern_discovery(
            observation, action, next_observation, extrinsic_reward
        )
        
        # Create reward object
        reward = IntrinsicReward(
            novelty_bonus=novelty * self.novelty_weight,
            prediction_error=prediction_error * self.prediction_weight,
            pattern_discovery=pattern * self.pattern_weight
        )
        
        # Update statistics
        self.stats['total_intrinsic_reward'] += reward.total
        
        # Store for future prediction
        self.recent_states.append(observation.copy())
        self.recent_actions.append(action)
        
        # Update prediction model
        self._update_prediction_model(observation, action, next_observation)
        
        return reward
    
    def _compute_novelty(self, observation: np.ndarray) -> float:
        """
        Compute novelty bonus based on state visit counts.
        
        Uses inverse count: bonus = 1 / sqrt(count + 1)
        This encourages visiting less-seen states.
        """
        state_hash = self.encoder.encode(observation)
        
        # Get current count
        count = self.state_counts.get(state_hash, 0)
        
        # Update count (with memory limit)
        if len(self.state_counts) >= self.memory_size:
            # Remove least visited state to make room
            if self.state_counts:
                min_key = min(self.state_counts, key=self.state_counts.get)
                del self.state_counts[min_key]
        
        self.state_counts[state_hash] = count + 1
        self.total_states_seen += 1
        
        # Compute bonus: inverse square root of count
        novelty = 1.0 / np.sqrt(count + 1)
        
        # Track unique states
        self.stats['unique_states'] = len(self.state_counts)
        self.stats['avg_novelty'] = (
            self.stats['avg_novelty'] * 0.99 + novelty * 0.01
        )
        
        return novelty if novelty > self.novelty_threshold else 0.0
    
    def _compute_prediction_error(
        self,
        observation: np.ndarray,
        action: int,
        next_observation: np.ndarray
    ) -> float:
        """
        Compute prediction error bonus.
        
        High prediction error = surprising transition = high bonus.
        """
        # Initialize prediction model if needed
        if self.prediction_weights is None:
            self.input_dim = len(observation) + 1  # +1 for action
            output_dim = len(observation)
            # Simple linear model: W * [obs, action]
            self.prediction_weights = np.random.randn(
                output_dim, self.input_dim
            ) * 0.01
        
        # Create input: concatenate observation and action
        input_vec = np.append(observation, action)
        
        # Predict next state
        predicted_next = np.dot(self.prediction_weights, input_vec)
        
        # Compute prediction error (normalized MSE)
        error = np.mean((predicted_next - next_observation) ** 2)
        
        # Normalize error to reasonable range (0 to 1)
        normalized_error = np.tanh(error)
        
        return normalized_error
    
    def _update_prediction_model(
        self,
        observation: np.ndarray,
        action: int,
        next_observation: np.ndarray
    ):
        """Update prediction model using simple gradient descent."""
        if self.prediction_weights is None:
            return
        
        input_vec = np.append(observation, action)
        predicted_next = np.dot(self.prediction_weights, input_vec)
        
        # Gradient: dL/dW = (predicted - actual) * input
        error = predicted_next - next_observation
        gradient = np.outer(error, input_vec)
        
        # Update weights
        self.prediction_weights -= self.prediction_lr * gradient
    
    def _check_pattern_discovery(
        self,
        observation: np.ndarray,
        action: int,
        next_observation: np.ndarray,
        extrinsic_reward: float
    ) -> float:
        """
        Check if this transition represents a new profitable pattern.
        
        A pattern is considered "discovered" if:
        1. It's a profitable trade (extrinsic_reward > threshold)
        2. The state-action combination hasn't been seen often
        """
        if extrinsic_reward <= 0.01:  # Only track profitable transitions
            return 0.0
        
        # Create pattern key from state-action pair
        state_hash = self.encoder.encode(observation)
        pattern_key = f"{state_hash}_{action}"
        
        # Check if this is a new pattern
        if pattern_key not in self.discovered_patterns:
            self.discovered_patterns[pattern_key] = {
                'count': 0,
                'total_reward': 0.0,
                'avg_reward': 0.0
            }
            self.stats['patterns_found'] += 1
        
        pattern = self.discovered_patterns[pattern_key]
        pattern['count'] += 1
        pattern['total_reward'] += extrinsic_reward
        pattern['avg_reward'] = pattern['total_reward'] / pattern['count']
        
        # Bonus for discovering new profitable patterns
        if pattern['count'] <= 3:  # First few discoveries get bonus
            discovery_bonus = 0.5 * extrinsic_reward
        else:
            discovery_bonus = 0.0
        
        return discovery_bonus
    
    def get_curiosity_score(self) -> float:
        """
        Get overall curiosity score (0 to 1).
        
        High score = agent is exploring well.
        Low score = agent is stuck in familiar states.
        """
        if self.total_states_seen == 0:
            return 0.0
        
        # Ratio of unique states to total states seen
        exploration_ratio = len(self.state_counts) / min(
            self.total_states_seen, self.memory_size
        )
        
        return np.clip(exploration_ratio, 0, 1)
    
    def get_stats(self) -> Dict:
        """Get curiosity module statistics."""
        return {
            **self.stats,
            'curiosity_score': self.get_curiosity_score(),
            'memory_usage': len(self.state_counts) / self.memory_size
        }
    
    def reset_episode(self):
        """Reset per-episode tracking (keep long-term memory)."""
        self.recent_states.clear()
        self.recent_actions.clear()
    
    def save_state(self) -> Dict:
        """Save module state for persistence."""
        return {
            'state_counts': dict(self.state_counts),
            'discovered_patterns': self.discovered_patterns,
            'stats': self.stats,
            'prediction_weights': (
                self.prediction_weights.tolist() 
                if self.prediction_weights is not None else None
            )
        }
    
    def load_state(self, state: Dict):
        """Load module state from persistence."""
        self.state_counts = state.get('state_counts', {})
        self.discovered_patterns = state.get('discovered_patterns', {})
        self.stats = state.get('stats', self.stats)
        
        weights = state.get('prediction_weights')
        if weights is not None:
            self.prediction_weights = np.array(weights)
