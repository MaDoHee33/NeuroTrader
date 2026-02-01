"""
Unit Tests for Self-Evolving AI Modules
=========================================
Lightweight tests that don't require loading real data or models.

Run with: python -m pytest tests/test_evolving.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json


class TestCuriosityModule:
    """Tests for CuriosityModule class."""
    
    def test_import(self):
        """Test that module can be imported."""
        from src.evolving.curiosity import CuriosityModule, IntrinsicReward
        assert CuriosityModule is not None
        assert IntrinsicReward is not None
    
    def test_initialization(self):
        """Test CuriosityModule initialization with default params."""
        from src.evolving.curiosity import CuriosityModule
        
        cm = CuriosityModule()
        assert cm.novelty_weight == 0.3
        assert cm.prediction_weight == 0.4
        assert cm.pattern_weight == 0.3
        assert cm.memory_size == 10000
    
    def test_initialization_custom(self):
        """Test CuriosityModule initialization with custom params."""
        from src.evolving.curiosity import CuriosityModule
        
        cm = CuriosityModule(
            novelty_weight=0.5,
            prediction_weight=0.3,
            pattern_weight=0.2,
            memory_size=1000
        )
        assert cm.novelty_weight == 0.5
        assert cm.memory_size == 1000
    
    def test_intrinsic_reward_dataclass(self):
        """Test IntrinsicReward dataclass."""
        from src.evolving.curiosity import IntrinsicReward
        
        reward = IntrinsicReward(
            novelty_bonus=0.1,
            prediction_error=0.2,
            pattern_discovery=0.3
        )
        assert reward.total == 0.6
        
        d = reward.to_dict()
        assert 'total' in d
        assert d['novelty_bonus'] == 0.1
    
    def test_compute_intrinsic_reward(self):
        """Test computing intrinsic reward."""
        from src.evolving.curiosity import CuriosityModule
        
        cm = CuriosityModule()
        
        obs = np.random.randn(10).astype(np.float32)
        next_obs = np.random.randn(10).astype(np.float32)
        action = 1
        ext_reward = 0.5
        
        reward = cm.compute_intrinsic_reward(obs, action, next_obs, ext_reward)
        
        assert reward.total >= 0
        assert hasattr(reward, 'novelty_bonus')
        assert hasattr(reward, 'prediction_error')
    
    def test_novelty_detection(self):
        """Test that novelty decreases for repeated states."""
        from src.evolving.curiosity import CuriosityModule
        
        cm = CuriosityModule()
        
        # Same observation repeated
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        next_obs = np.array([1.1, 2.1, 3.1], dtype=np.float32)
        
        # First time seeing state
        r1 = cm.compute_intrinsic_reward(obs, 1, next_obs, 0.0)
        
        # Second time (should have lower novelty)
        r2 = cm.compute_intrinsic_reward(obs, 1, next_obs, 0.0)
        
        # Novelty should decrease
        assert r2.novelty_bonus <= r1.novelty_bonus
    
    def test_curiosity_score(self):
        """Test curiosity score calculation."""
        from src.evolving.curiosity import CuriosityModule
        
        cm = CuriosityModule()
        
        # Initially zero (no states seen)
        assert cm.get_curiosity_score() == 0.0
        
        # After seeing some states
        for i in range(10):
            obs = np.random.randn(5).astype(np.float32)
            next_obs = np.random.randn(5).astype(np.float32)
            cm.compute_intrinsic_reward(obs, 1, next_obs, 0.0)
        
        score = cm.get_curiosity_score()
        assert 0 <= score <= 1
    
    def test_stats(self):
        """Test statistics tracking."""
        from src.evolving.curiosity import CuriosityModule
        
        cm = CuriosityModule()
        
        for _ in range(5):
            obs = np.random.randn(5).astype(np.float32)
            next_obs = np.random.randn(5).astype(np.float32)
            cm.compute_intrinsic_reward(obs, 1, next_obs, 0.0)
        
        stats = cm.get_stats()
        assert 'total_intrinsic_reward' in stats
        assert 'unique_states' in stats
        assert 'curiosity_score' in stats
    
    def test_save_load_state(self):
        """Test persistence."""
        from src.evolving.curiosity import CuriosityModule
        
        cm1 = CuriosityModule()
        
        # Generate some state
        for _ in range(5):
            obs = np.random.randn(5).astype(np.float32)
            next_obs = np.random.randn(5).astype(np.float32)
            cm1.compute_intrinsic_reward(obs, 1, next_obs, 0.1)
        
        # Save state
        state = cm1.save_state()
        
        # Load into new module
        cm2 = CuriosityModule()
        cm2.load_state(state)
        
        assert len(cm2.state_counts) == len(cm1.state_counts)


class TestExperienceBuffer:
    """Tests for ExperienceBuffer class."""
    
    def test_import(self):
        """Test that module can be imported."""
        from src.evolving.experience_buffer import ExperienceBuffer, TradeExperience
        assert ExperienceBuffer is not None
        assert TradeExperience is not None
    
    def test_initialization(self):
        """Test ExperienceBuffer initialization."""
        from src.evolving.experience_buffer import ExperienceBuffer
        
        eb = ExperienceBuffer(max_size=100)
        assert eb.max_size == 100
        assert len(eb.experiences) == 0
    
    def test_add_experience(self):
        """Test adding experiences."""
        from src.evolving.experience_buffer import ExperienceBuffer
        
        eb = ExperienceBuffer(max_size=100)
        
        obs = np.random.randn(10).astype(np.float32)
        next_obs = np.random.randn(10).astype(np.float32)
        
        exp = eb.add(
            observation=obs,
            action=1,
            reward=0.5,
            next_observation=next_obs,
            pnl=0.01,
            holding_time=5,
            market_regime='bull'
        )
        
        assert len(eb.experiences) == 1
        assert exp.experience_id in eb.experiences
        assert exp.was_profitable == True
    
    def test_eviction(self):
        """Test priority-based eviction when buffer is full."""
        from src.evolving.experience_buffer import ExperienceBuffer
        
        eb = ExperienceBuffer(max_size=5)
        
        # Add more than max
        for i in range(10):
            obs = np.random.randn(5).astype(np.float32)
            next_obs = np.random.randn(5).astype(np.float32)
            eb.add(
                observation=obs,
                action=1,
                reward=0.1 * i,
                next_observation=next_obs,
                pnl=0.001 * i
            )
        
        # Should be capped at max_size
        assert len(eb.experiences) == 5
        assert eb.stats['total_evicted'] == 5
    
    def test_sample_batch(self):
        """Test batch sampling."""
        from src.evolving.experience_buffer import ExperienceBuffer
        
        eb = ExperienceBuffer(max_size=100)
        
        # Add experiences
        for i in range(20):
            obs = np.random.randn(5).astype(np.float32)
            next_obs = np.random.randn(5).astype(np.float32)
            eb.add(
                observation=obs,
                action=i % 3,
                reward=0.1,
                next_observation=next_obs,
                pnl=0.01 if i % 2 == 0 else -0.01
            )
        
        batch = eb.sample_batch(batch_size=5)
        assert len(batch) == 5
        
        # Test filtered sampling
        profitable = eb.sample_batch(batch_size=10, only_profitable=True)
        assert all(exp.was_profitable for exp in profitable)
    
    def test_regime_filtering(self):
        """Test filtering by regime."""
        from src.evolving.experience_buffer import ExperienceBuffer
        
        eb = ExperienceBuffer(max_size=100)
        
        # Add experiences with different regimes
        for regime in ['bull', 'bear', 'sideways']:
            for _ in range(3):
                obs = np.random.randn(5).astype(np.float32)
                next_obs = np.random.randn(5).astype(np.float32)
                eb.add(
                    observation=obs,
                    action=1,
                    reward=0.1,
                    next_observation=next_obs,
                    market_regime=regime
                )
        
        bull_samples = eb.sample_batch(batch_size=10, filter_regime='bull')
        assert len(bull_samples) == 3
    
    def test_save_load(self):
        """Test persistence."""
        from src.evolving.experience_buffer import ExperienceBuffer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_buffer.json"
            
            # Create and populate
            eb1 = ExperienceBuffer(max_size=100, save_path=save_path)
            for _ in range(5):
                obs = np.random.randn(5).astype(np.float32)
                next_obs = np.random.randn(5).astype(np.float32)
                eb1.add(observation=obs, action=1, reward=0.1, next_observation=next_obs)
            eb1.save()
            
            # Load into new buffer
            eb2 = ExperienceBuffer(max_size=100, save_path=save_path)
            
            assert len(eb2.experiences) == 5


class TestCurriculumManager:
    """Tests for CurriculumManager class."""
    
    def test_import(self):
        """Test that module can be imported."""
        from src.evolving.difficulty_scaler import CurriculumManager, DifficultyLevel
        assert CurriculumManager is not None
        assert DifficultyLevel is not None
    
    def test_initialization(self):
        """Test CurriculumManager initialization."""
        from src.evolving.difficulty_scaler import CurriculumManager, DifficultyLevel
        
        cm = CurriculumManager(start_level=DifficultyLevel.EASY)
        assert cm.get_current_level() == DifficultyLevel.EASY
    
    def test_record_episode(self):
        """Test recording episode results."""
        from src.evolving.difficulty_scaler import CurriculumManager, DifficultyLevel
        
        cm = CurriculumManager()
        
        result = cm.record_episode(
            total_return=0.05,
            win_rate=0.6,
            num_trades=10
        )
        
        assert 'advanced' in result
        assert 'regressed' in result
        assert cm.total_episodes == 1
    
    def test_advancement(self):
        """Test level advancement after good performance."""
        from src.evolving.difficulty_scaler import CurriculumManager, DifficultyLevel
        
        cm = CurriculumManager(start_level=DifficultyLevel.EASY)
        
        # Simulate many successful episodes
        for _ in range(20):
            cm.record_episode(total_return=0.1, win_rate=0.6, num_trades=10)
        
        # Should have advanced from EASY
        assert cm.get_current_level() >= DifficultyLevel.MEDIUM
    
    def test_regression(self):
        """Test level regression after poor performance."""
        from src.evolving.difficulty_scaler import CurriculumManager, DifficultyLevel
        
        cm = CurriculumManager(start_level=DifficultyLevel.MEDIUM, allow_regression=True)
        
        # Force some episodes to get past minimum
        for _ in range(5):
            cm.record_episode(total_return=0.05, win_rate=0.5, num_trades=5)
        
        # Simulate consecutive failures
        for _ in range(6):
            cm.record_episode(total_return=-0.1, win_rate=0.2, num_trades=5)
        
        # Should have regressed
        assert cm.get_current_level() == DifficultyLevel.EASY
    
    def test_progress_report(self):
        """Test progress report generation."""
        from src.evolving.difficulty_scaler import CurriculumManager
        
        cm = CurriculumManager()
        
        for _ in range(3):
            cm.record_episode(total_return=0.05, win_rate=0.5, num_trades=5)
        
        report = cm.get_progress_report()
        assert 'current_level' in report
        assert 'total_episodes' in report
        assert 'level_stats' in report


class TestMarketRegimeDetector:
    """Tests for MarketRegimeDetector class."""
    
    def test_import(self):
        """Test that module can be imported."""
        from src.evolving.regime_detector import MarketRegimeDetector, MarketRegime
        assert MarketRegimeDetector is not None
        assert MarketRegime is not None
    
    def test_initialization(self):
        """Test MarketRegimeDetector initialization."""
        from src.evolving.regime_detector import MarketRegimeDetector
        
        detector = MarketRegimeDetector(trend_period=20)
        assert detector.trend_period == 20
    
    def test_update_single(self):
        """Test single price update."""
        from src.evolving.regime_detector import MarketRegimeDetector, MarketRegime
        
        detector = MarketRegimeDetector()
        
        regime, metrics = detector.update(100.0)
        
        # Not enough data yet
        assert regime == MarketRegime.UNKNOWN
    
    def test_bull_detection(self):
        """Test bull market detection."""
        from src.evolving.regime_detector import MarketRegimeDetector, MarketRegime
        
        detector = MarketRegimeDetector(trend_period=10)
        
        # Simulate uptrend
        for i in range(30):
            price = 100 + i * 0.5  # Steady increase
            regime, metrics = detector.update(price)
        
        assert regime == MarketRegime.BULL
        assert metrics.trend_strength > 0
    
    def test_bear_detection(self):
        """Test bear market detection."""
        from src.evolving.regime_detector import MarketRegimeDetector, MarketRegime
        
        detector = MarketRegimeDetector(trend_period=10)
        
        # Simulate downtrend
        for i in range(30):
            price = 100 - i * 0.5  # Steady decrease
            regime, metrics = detector.update(price)
        
        assert regime == MarketRegime.BEAR
        assert metrics.trend_strength < 0
    
    def test_sideways_detection(self):
        """Test sideways market detection."""
        from src.evolving.regime_detector import MarketRegimeDetector, MarketRegime
        
        detector = MarketRegimeDetector(trend_period=10)
        
        # Simulate range-bound price
        for i in range(30):
            price = 100 + 2 * np.sin(i * 0.5)  # Oscillating
            regime, metrics = detector.update(price)
        
        # Should be sideways or similar
        assert regime in [MarketRegime.SIDEWAYS, MarketRegime.VOLATILE]
    
    def test_regime_probabilities(self):
        """Test regime probability distribution."""
        from src.evolving.regime_detector import MarketRegimeDetector
        
        detector = MarketRegimeDetector()
        
        for i in range(30):
            price = 100 + i * 0.3
            regime, metrics = detector.update(price)
        
        probs = detector.get_regime_probabilities(metrics)
        
        # Should be valid probabilities
        assert sum(probs.values()) > 0
        assert all(0 <= p <= 1 for p in probs.values())
    
    def test_stats(self):
        """Test statistics tracking."""
        from src.evolving.regime_detector import MarketRegimeDetector
        
        detector = MarketRegimeDetector()
        
        for i in range(20):
            detector.update(100 + np.random.randn())
        
        stats = detector.get_stats()
        assert 'current_regime' in stats
        assert 'total_steps' in stats
        assert stats['total_steps'] == 20


class TestDifficultyScaler:
    """Tests for DifficultyScaler class."""
    
    def test_import(self):
        """Test that module can be imported."""
        from src.evolving.difficulty_scaler import DifficultyScaler, DifficultyLevel
        assert DifficultyScaler is not None
    
    def test_assess_difficulty(self):
        """Test difficulty assessment."""
        from src.evolving.difficulty_scaler import DifficultyScaler, DifficultyLevel
        
        scaler = DifficultyScaler()
        
        # Easy: low volatility uptrend
        easy_prices = np.linspace(100, 110, 50)  # Smooth trend
        level, metrics = scaler.assess_difficulty(easy_prices)
        
        assert level in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM]
        assert 'volatility' in metrics
        assert 'trend_strength' in metrics


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
