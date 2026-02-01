# NeuroTrader Self-Evolving AI Module
# =====================================
# This package contains modules for building a self-evolving trading AI
# that learns continuously from experience.

from .curiosity import CuriosityModule, IntrinsicReward
from .experience_buffer import ExperienceBuffer, TradeExperience
from .difficulty_scaler import DifficultyScaler, CurriculumManager, DifficultyLevel
from .regime_detector import MarketRegimeDetector, MarketRegime, RegimeMetrics
from .hybrid_agent import HybridTradingAgent

__all__ = [
    # Core Modules
    'CuriosityModule',
    'IntrinsicReward', 
    'ExperienceBuffer',
    'TradeExperience',
    'DifficultyScaler',
    'CurriculumManager',
    'DifficultyLevel',
    # Regime Detection
    'MarketRegimeDetector',
    'MarketRegime',
    'RegimeMetrics',
    # Integration
    'HybridTradingAgent'
]
