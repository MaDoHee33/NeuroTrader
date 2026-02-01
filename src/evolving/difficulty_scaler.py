"""
Difficulty Scaler & Curriculum Manager
=======================================
Implements Progressive Difficulty Scaling for curriculum learning.

This module gradually increases trading difficulty as the agent improves:
1. Start with easy markets (low volatility, clear trends)
2. Progress to harder markets (high volatility, choppy)
3. Eventually handle all market conditions

Design for Low Resources:
- Simple rule-based difficulty assessment (no ML overhead)
- Environment wrapper approach (minimal code changes)
- Configurable progression thresholds
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import IntEnum


class DifficultyLevel(IntEnum):
    """Difficulty levels for curriculum learning."""
    EASY = 1        # Clear trends, low volatility
    MEDIUM = 2      # Some noise, moderate volatility
    HARD = 3        # High noise, high volatility
    EXPERT = 4      # Black swan events, extreme conditions


@dataclass
class DifficultyConfig:
    """Configuration for each difficulty level."""
    level: DifficultyLevel
    name: str
    description: str
    
    # Market characteristics
    max_volatility: float  # ATR threshold
    min_trend_strength: float  # Min trend clarity
    allow_choppy: bool  # Allow sideways markets
    allow_news_events: bool  # Allow high-impact news periods
    
    # Performance thresholds to advance
    min_win_rate: float  # Min win rate to advance
    min_sharpe: float  # Min Sharpe ratio to advance
    min_episodes: int  # Min episodes at this level


# Default curriculum configurations
DEFAULT_CURRICULUM = {
    DifficultyLevel.EASY: DifficultyConfig(
        level=DifficultyLevel.EASY,
        name="Training Wheels",
        description="Clear trends, low volatility - perfect for learning basics",
        max_volatility=0.3,
        min_trend_strength=0.6,
        allow_choppy=False,
        allow_news_events=False,
        min_win_rate=0.4,
        min_sharpe=0.0,
        min_episodes=10
    ),
    DifficultyLevel.MEDIUM: DifficultyConfig(
        level=DifficultyLevel.MEDIUM,
        name="Real World",
        description="Normal market conditions with some noise",
        max_volatility=0.6,
        min_trend_strength=0.3,
        allow_choppy=True,
        allow_news_events=False,
        min_win_rate=0.45,
        min_sharpe=0.3,
        min_episodes=20
    ),
    DifficultyLevel.HARD: DifficultyConfig(
        level=DifficultyLevel.HARD,
        name="Volatile Markets",
        description="High volatility and unpredictable movements",
        max_volatility=1.0,
        min_trend_strength=0.0,
        allow_choppy=True,
        allow_news_events=True,
        min_win_rate=0.5,
        min_sharpe=0.5,
        min_episodes=30
    ),
    DifficultyLevel.EXPERT: DifficultyConfig(
        level=DifficultyLevel.EXPERT,
        name="Master Level",
        description="All conditions including black swans",
        max_volatility=float('inf'),
        min_trend_strength=0.0,
        allow_choppy=True,
        allow_news_events=True,
        min_win_rate=0.55,
        min_sharpe=0.7,
        min_episodes=50
    )
}


class DifficultyScaler:
    """
    Evaluates market conditions and assigns difficulty scores.
    
    Used to filter training data by difficulty level.
    """
    
    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50
    ):
        """
        Initialize Difficulty Scaler.
        
        Args:
            volatility_window: Lookback for volatility calculation
            trend_window: Lookback for trend strength calculation
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
    
    def assess_difficulty(
        self,
        prices: np.ndarray,
        atr: Optional[np.ndarray] = None
    ) -> Tuple[DifficultyLevel, Dict]:
        """
        Assess difficulty of given market data.
        
        Args:
            prices: Array of close prices
            atr: Optional pre-computed ATR values
            
        Returns:
            Tuple of (DifficultyLevel, metrics_dict)
        """
        metrics = {}
        
        # 1. Calculate volatility (normalized ATR)
        if atr is not None and len(atr) > 0:
            volatility = np.mean(atr[-self.volatility_window:]) / np.mean(prices[-self.volatility_window:])
        else:
            # Calculate from price returns
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns[-self.volatility_window:]) if len(returns) >= self.volatility_window else 0.01
        
        metrics['volatility'] = volatility
        
        # 2. Calculate trend strength (R-squared of linear regression)
        if len(prices) >= self.trend_window:
            recent_prices = prices[-self.trend_window:]
            x = np.arange(len(recent_prices))
            
            # Linear regression
            slope, intercept = np.polyfit(x, recent_prices, 1)
            predicted = slope * x + intercept
            
            # R-squared
            ss_res = np.sum((recent_prices - predicted) ** 2)
            ss_tot = np.sum((recent_prices - np.mean(recent_prices)) ** 2)
            trend_strength = 1 - (ss_res / (ss_tot + 1e-10))
        else:
            trend_strength = 0.5
        
        metrics['trend_strength'] = trend_strength
        
        # 3. Check for choppiness (price crosses moving average often)
        if len(prices) >= 20:
            ma = np.convolve(prices, np.ones(20)/20, mode='valid')
            recent_prices = prices[-len(ma):]
            crosses = np.sum(np.abs(np.diff(np.sign(recent_prices - ma)))) / 2
            choppiness = crosses / len(ma)
        else:
            choppiness = 0.0
        
        metrics['choppiness'] = choppiness
        
        # 4. Determine difficulty level
        if volatility < 0.3 and trend_strength > 0.6:
            level = DifficultyLevel.EASY
        elif volatility < 0.6 and trend_strength > 0.3:
            level = DifficultyLevel.MEDIUM
        elif volatility < 1.0:
            level = DifficultyLevel.HARD
        else:
            level = DifficultyLevel.EXPERT
        
        metrics['difficulty_level'] = int(level)
        
        return level, metrics
    
    def filter_by_difficulty(
        self,
        data_segments: List[np.ndarray],
        max_difficulty: DifficultyLevel
    ) -> List[np.ndarray]:
        """
        Filter data segments to only include those at or below difficulty level.
        
        Args:
            data_segments: List of price arrays
            max_difficulty: Maximum allowed difficulty
            
        Returns:
            Filtered list of segments
        """
        filtered = []
        for segment in data_segments:
            level, _ = self.assess_difficulty(segment)
            if level <= max_difficulty:
                filtered.append(segment)
        
        return filtered


class CurriculumManager:
    """
    Manages the curriculum learning progression.
    
    Tracks agent performance at each level and decides when to advance.
    """
    
    def __init__(
        self,
        curriculum: Optional[Dict[DifficultyLevel, DifficultyConfig]] = None,
        start_level: DifficultyLevel = DifficultyLevel.EASY,
        allow_regression: bool = True
    ):
        """
        Initialize Curriculum Manager.
        
        Args:
            curriculum: Custom curriculum config (default uses DEFAULT_CURRICULUM)
            start_level: Starting difficulty level
            allow_regression: Allow going back to easier levels on poor performance
        """
        self.curriculum = curriculum or DEFAULT_CURRICULUM
        self.current_level = start_level
        self.allow_regression = allow_regression
        
        # Performance tracking per level
        self.level_stats: Dict[DifficultyLevel, Dict] = {
            level: {
                'episodes': 0,
                'wins': 0,
                'total_return': 0.0,
                'returns': [],
                'sharpe': 0.0,
                'consecutive_failures': 0
            }
            for level in DifficultyLevel
        }
        
        # History
        self.level_history: List[Tuple[int, DifficultyLevel]] = []
        self.total_episodes = 0
    
    def get_current_config(self) -> DifficultyConfig:
        """Get current difficulty configuration."""
        return self.curriculum[self.current_level]
    
    def get_current_level(self) -> DifficultyLevel:
        """Get current difficulty level."""
        return self.current_level
    
    def record_episode(
        self,
        total_return: float,
        win_rate: float,
        num_trades: int
    ) -> Dict:
        """
        Record episode performance and check for level advancement.
        
        Args:
            total_return: Episode return (%)
            win_rate: Episode win rate (0-1)
            num_trades: Number of trades in episode
            
        Returns:
            Dict with 'advanced', 'regressed', 'new_level' keys
        """
        self.total_episodes += 1
        stats = self.level_stats[self.current_level]
        
        # Update stats
        stats['episodes'] += 1
        stats['returns'].append(total_return)
        stats['total_return'] += total_return
        
        # Track wins
        is_win = total_return > 0
        if is_win:
            stats['wins'] += 1
            stats['consecutive_failures'] = 0
        else:
            stats['consecutive_failures'] += 1
        
        # Calculate Sharpe (simplified)
        if len(stats['returns']) >= 10:
            returns = np.array(stats['returns'][-30:])  # Last 30 episodes
            mean_return = np.mean(returns)
            std_return = np.std(returns) + 1e-10
            stats['sharpe'] = mean_return / std_return
        
        # Check advancement
        result = {'advanced': False, 'regressed': False, 'new_level': self.current_level}
        
        config = self.curriculum[self.current_level]
        current_win_rate = stats['wins'] / max(1, stats['episodes'])
        
        # Check for advancement
        if (stats['episodes'] >= config.min_episodes and
            current_win_rate >= config.min_win_rate and
            stats['sharpe'] >= config.min_sharpe and
            self.current_level < DifficultyLevel.EXPERT):
            
            # Advance!
            new_level = DifficultyLevel(self.current_level + 1)
            self.current_level = new_level
            self.level_history.append((self.total_episodes, new_level))
            result['advanced'] = True
            result['new_level'] = new_level
            print(f"[CURRICULUM] 🎉 Advanced to {self.curriculum[new_level].name}!")
        
        # Check for regression
        elif (self.allow_regression and
              stats['consecutive_failures'] >= 5 and
              self.current_level > DifficultyLevel.EASY):
            
            # Regress
            new_level = DifficultyLevel(self.current_level - 1)
            self.current_level = new_level
            self.level_history.append((self.total_episodes, new_level))
            result['regressed'] = True
            result['new_level'] = new_level
            stats['consecutive_failures'] = 0  # Reset
            print(f"[CURRICULUM] ⬇️ Regressed to {self.curriculum[new_level].name}")
        
        return result
    
    def should_skip_segment(
        self,
        segment_difficulty: DifficultyLevel
    ) -> bool:
        """Check if a data segment should be skipped based on current level."""
        return segment_difficulty > self.current_level
    
    def get_progress_report(self) -> Dict:
        """Get detailed progress report."""
        return {
            'current_level': self.current_level.name,
            'current_config': self.curriculum[self.current_level].description,
            'total_episodes': self.total_episodes,
            'level_stats': {
                level.name: {
                    'episodes': stats['episodes'],
                    'win_rate': stats['wins'] / max(1, stats['episodes']),
                    'sharpe': stats['sharpe'],
                    'avg_return': stats['total_return'] / max(1, stats['episodes'])
                }
                for level, stats in self.level_stats.items()
                if stats['episodes'] > 0
            },
            'level_history': [
                {'episode': ep, 'level': level.name}
                for ep, level in self.level_history
            ]
        }
    
    def reset_level(self, level: DifficultyLevel):
        """Reset to a specific level (useful for fresh start)."""
        self.current_level = level
        self.level_history.append((self.total_episodes, level))
    
    def save_state(self) -> Dict:
        """Save curriculum state for persistence."""
        return {
            'current_level': int(self.current_level),
            'total_episodes': self.total_episodes,
            'level_stats': {
                int(level): {
                    **stats,
                    'returns': stats['returns'][-100:] if stats['returns'] else []
                }
                for level, stats in self.level_stats.items()
            },
            'level_history': [
                (ep, int(level)) for ep, level in self.level_history
            ]
        }
    
    def load_state(self, state: Dict):
        """Load curriculum state from persistence."""
        self.current_level = DifficultyLevel(state.get('current_level', 1))
        self.total_episodes = state.get('total_episodes', 0)
        
        for level_int, stats in state.get('level_stats', {}).items():
            level = DifficultyLevel(int(level_int))
            if level in self.level_stats:
                self.level_stats[level].update(stats)
        
        self.level_history = [
            (ep, DifficultyLevel(level))
            for ep, level in state.get('level_history', [])
        ]
