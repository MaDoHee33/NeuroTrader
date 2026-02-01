"""
Experience Buffer Module
=========================
Implements a persistent experience storage system for lifelong learning.

This module stores trading experiences as structured "stories" that can be:
1. Replayed for offline learning
2. Analyzed for pattern discovery
3. Used for few-shot learning on new market conditions

Design Principles:
- Low memory footprint (fixed-size buffer with smart eviction)
- Prioritized replay (important experiences kept longer)
- Structured storage (context, action, outcome, lesson)
"""

import json
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import hashlib


@dataclass
class TradeExperience:
    """
    A single trading experience - the fundamental unit of learning.
    
    Structured as a "story" with context, action, outcome, and lesson.
    """
    # Context: What was the market situation?
    timestamp: str
    market_state: List[float]  # Observation vector
    market_regime: str  # bull, bear, sideways, volatile
    
    # Action: What did we decide?
    action: int  # 0=HOLD, 1=BUY, 2=SELL
    confidence: float  # Agent's confidence (0-1)
    
    # Outcome: What happened?
    reward: float
    next_state: List[float]
    pnl: float  # Profit/Loss
    holding_time: int  # Steps held
    
    # Lesson: What can we learn?
    was_profitable: bool
    lesson_tags: List[str]  # e.g., ['quick_profit', 'trend_follow']
    
    # Metadata
    experience_id: str = ""
    episode_id: str = ""
    priority: float = 1.0  # Higher = more important to keep
    
    def __post_init__(self):
        if not self.experience_id:
            self.experience_id = hashlib.md5(
                f"{self.timestamp}_{np.random.rand()}".encode()
            ).hexdigest()[:12]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeExperience':
        return cls(**data)


class ExperienceBuffer:
    """
    Persistent Experience Buffer for Lifelong Learning
    
    Features:
    - Fixed-size memory with priority-based eviction
    - Categorized storage (profitable/unprofitable, by regime)
    - Batch sampling for training
    - JSON persistence for cross-session learning
    
    This is the "memory" of the Self-Evolving AI.
    """
    
    def __init__(
        self,
        max_size: int = 50000,
        min_priority: float = 0.1,
        save_path: Optional[Path] = None,
        auto_save_interval: int = 1000
    ):
        """
        Initialize Experience Buffer.
        
        Args:
            max_size: Maximum number of experiences to store
            min_priority: Minimum priority to keep experience
            save_path: Path for persistence (None = no persistence)
            auto_save_interval: Auto-save every N additions
        """
        self.max_size = max_size
        self.min_priority = min_priority
        self.save_path = Path(save_path) if save_path else None
        self.auto_save_interval = auto_save_interval
        
        # Main storage
        self.experiences: Dict[str, TradeExperience] = {}
        
        # Indexed storage for fast access
        self.by_regime: Dict[str, List[str]] = {
            'bull': [], 'bear': [], 'sideways': [], 'volatile': [], 'unknown': []
        }
        self.profitable_ids: List[str] = []
        self.unprofitable_ids: List[str] = []
        
        # Priority queue for eviction
        self.priority_order: List[Tuple[float, str]] = []
        
        # Statistics
        self.stats = {
            'total_added': 0,
            'total_evicted': 0,
            'profitable_ratio': 0.0,
            'avg_pnl': 0.0,
            'by_regime_count': {k: 0 for k in self.by_regime}
        }
        
        # Addition counter for auto-save
        self._addition_counter = 0
        
        # Load existing data if available
        if self.save_path and self.save_path.exists():
            self.load()
    
    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        pnl: float = 0.0,
        holding_time: int = 0,
        market_regime: str = 'unknown',
        confidence: float = 0.5,
        episode_id: str = "",
        lesson_tags: Optional[List[str]] = None
    ) -> TradeExperience:
        """
        Add a new experience to the buffer.
        
        Returns the created TradeExperience object.
        """
        # Create experience
        experience = TradeExperience(
            timestamp=datetime.now().isoformat(),
            market_state=observation.tolist(),
            market_regime=market_regime,
            action=action,
            confidence=confidence,
            reward=reward,
            next_state=next_observation.tolist(),
            pnl=pnl,
            holding_time=holding_time,
            was_profitable=pnl > 0,
            lesson_tags=lesson_tags or [],
            episode_id=episode_id,
            priority=self._compute_priority(reward, pnl, holding_time)
        )
        
        # Check if we need to evict
        if len(self.experiences) >= self.max_size:
            self._evict_lowest_priority()
        
        # Add to storage
        self.experiences[experience.experience_id] = experience
        
        # Update indices
        regime = market_regime if market_regime in self.by_regime else 'unknown'
        self.by_regime[regime].append(experience.experience_id)
        
        if experience.was_profitable:
            self.profitable_ids.append(experience.experience_id)
        else:
            self.unprofitable_ids.append(experience.experience_id)
        
        # Update priority order
        self.priority_order.append((experience.priority, experience.experience_id))
        self.priority_order.sort(key=lambda x: x[0])
        
        # Update statistics
        self._update_stats(experience)
        
        # Auto-save check
        self._addition_counter += 1
        if (self.save_path and 
            self._addition_counter % self.auto_save_interval == 0):
            self.save()
        
        return experience
    
    def _compute_priority(
        self,
        reward: float,
        pnl: float,
        holding_time: int
    ) -> float:
        """
        Compute priority score for experience.
        
        Higher priority = more likely to be kept.
        
        Factors:
        - Absolute PnL (extreme wins/losses are interesting)
        - Reward magnitude
        - Holding time (quick trades are interesting for Scalper)
        """
        # Base priority from absolute PnL
        pnl_factor = np.tanh(abs(pnl) * 10) * 0.4
        
        # Reward factor
        reward_factor = np.tanh(abs(reward) * 5) * 0.3
        
        # Speed factor (shorter holding time = higher priority for scalping)
        speed_factor = 0.3 * (1.0 / (1.0 + holding_time * 0.1))
        
        # Bonus for profitable experiences
        profit_bonus = 0.2 if pnl > 0 else 0.0
        
        return pnl_factor + reward_factor + speed_factor + profit_bonus
    
    def _evict_lowest_priority(self):
        """Remove the lowest priority experience."""
        if not self.priority_order:
            return
        
        # Get lowest priority experience
        _, exp_id = self.priority_order.pop(0)
        
        if exp_id not in self.experiences:
            return
        
        experience = self.experiences[exp_id]
        
        # Remove from indices
        regime = experience.market_regime
        if regime in self.by_regime and exp_id in self.by_regime[regime]:
            self.by_regime[regime].remove(exp_id)
        
        if exp_id in self.profitable_ids:
            self.profitable_ids.remove(exp_id)
        if exp_id in self.unprofitable_ids:
            self.unprofitable_ids.remove(exp_id)
        
        # Remove from main storage
        del self.experiences[exp_id]
        
        self.stats['total_evicted'] += 1
    
    def _update_stats(self, experience: TradeExperience):
        """Update running statistics."""
        self.stats['total_added'] += 1
        
        # Update profitable ratio
        n = len(self.experiences)
        if n > 0:
            self.stats['profitable_ratio'] = len(self.profitable_ids) / n
        
        # Update regime counts
        regime = experience.market_regime
        if regime in self.stats['by_regime_count']:
            self.stats['by_regime_count'][regime] += 1
        
        # Update average PnL (running average)
        alpha = 0.01
        self.stats['avg_pnl'] = (
            self.stats['avg_pnl'] * (1 - alpha) + experience.pnl * alpha
        )
    
    def sample_batch(
        self,
        batch_size: int = 32,
        filter_regime: Optional[str] = None,
        only_profitable: bool = False,
        prioritized: bool = True
    ) -> List[TradeExperience]:
        """
        Sample a batch of experiences for training.
        
        Args:
            batch_size: Number of experiences to sample
            filter_regime: Only sample from specific regime
            only_profitable: Only sample profitable experiences
            prioritized: Use priority weighting (higher priority = more likely)
            
        Returns:
            List of TradeExperience objects
        """
        # Get candidate pool
        if filter_regime and filter_regime in self.by_regime:
            candidates = self.by_regime[filter_regime]
        elif only_profitable:
            candidates = self.profitable_ids
        else:
            candidates = list(self.experiences.keys())
        
        if not candidates:
            return []
        
        # Sample
        n_samples = min(batch_size, len(candidates))
        
        if prioritized and n_samples < len(candidates):
            # Weighted sampling by priority
            priorities = [
                self.experiences[cid].priority for cid in candidates
            ]
            priorities = np.array(priorities)
            priorities = priorities / priorities.sum()
            
            indices = np.random.choice(
                len(candidates),
                size=n_samples,
                replace=False,
                p=priorities
            )
            sampled_ids = [candidates[i] for i in indices]
        else:
            # Uniform sampling
            sampled_ids = np.random.choice(
                candidates,
                size=n_samples,
                replace=False
            ).tolist()
        
        return [self.experiences[eid] for eid in sampled_ids]
    
    def get_lessons_for_regime(self, regime: str) -> List[TradeExperience]:
        """Get all profitable experiences for a specific market regime."""
        if regime not in self.by_regime:
            return []
        
        return [
            self.experiences[eid] 
            for eid in self.by_regime[regime]
            if eid in self.experiences and self.experiences[eid].was_profitable
        ]
    
    def get_similar_experiences(
        self,
        observation: np.ndarray,
        top_k: int = 5
    ) -> List[TradeExperience]:
        """
        Find experiences with similar market states.
        
        Useful for few-shot learning: "What did I do in similar situations?"
        """
        if not self.experiences:
            return []
        
        # Calculate similarities
        similarities = []
        for exp_id, exp in self.experiences.items():
            state = np.array(exp.market_state)
            if len(state) != len(observation):
                continue
            
            # Cosine similarity
            norm1 = np.linalg.norm(observation)
            norm2 = np.linalg.norm(state)
            
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(observation, state) / (norm1 * norm2)
                similarities.append((sim, exp_id))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k
        return [
            self.experiences[eid] 
            for _, eid in similarities[:top_k]
        ]
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            **self.stats,
            'current_size': len(self.experiences),
            'capacity': self.max_size,
            'utilization': len(self.experiences) / self.max_size
        }
    
    def save(self):
        """Save buffer to disk."""
        if self.save_path is None:
            return
        
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'experiences': {
                eid: exp.to_dict() 
                for eid, exp in self.experiences.items()
            },
            'stats': self.stats,
            'version': '1.0'
        }
        
        with open(self.save_path, 'w') as f:
            json.dump(data, f)
    
    def load(self):
        """Load buffer from disk."""
        if self.save_path is None or not self.save_path.exists():
            return
        
        with open(self.save_path, 'r') as f:
            data = json.load(f)
        
        # Restore experiences
        for eid, exp_dict in data.get('experiences', {}).items():
            exp = TradeExperience.from_dict(exp_dict)
            self.experiences[eid] = exp
            
            # Rebuild indices
            regime = exp.market_regime
            if regime in self.by_regime:
                self.by_regime[regime].append(eid)
            
            if exp.was_profitable:
                self.profitable_ids.append(eid)
            else:
                self.unprofitable_ids.append(eid)
            
            self.priority_order.append((exp.priority, eid))
        
        # Sort priority order
        self.priority_order.sort(key=lambda x: x[0])
        
        # Restore stats
        self.stats.update(data.get('stats', {}))
    
    def clear(self):
        """Clear all experiences (useful for fresh start)."""
        self.experiences.clear()
        for regime in self.by_regime:
            self.by_regime[regime].clear()
        self.profitable_ids.clear()
        self.unprofitable_ids.clear()
        self.priority_order.clear()
        self.stats = {
            'total_added': 0,
            'total_evicted': 0,
            'profitable_ratio': 0.0,
            'avg_pnl': 0.0,
            'by_regime_count': {k: 0 for k in self.by_regime}
        }
