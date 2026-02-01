"""
Market Regime Detection Module
================================
Detects current market conditions for adaptive strategy selection.

Market Regimes:
1. BULL - Strong uptrend
2. BEAR - Strong downtrend
3. SIDEWAYS - Range-bound/consolidation
4. VOLATILE - High volatility, direction unclear
5. BREAKOUT - Potential trend change

This module provides:
- Real-time regime classification
- Regime transition detection
- Historical regime analysis

Design: Uses only numpy/pandas for low resource usage.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from collections import deque


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


@dataclass
class RegimeMetrics:
    """Metrics used for regime detection."""
    trend_strength: float      # -1 (strong bear) to +1 (strong bull)
    volatility: float          # 0 to 1 (normalized ATR)
    momentum: float            # RSI-based, 0 to 1
    range_bound: float         # 0 to 1 (how range-bound is price)
    breakout_score: float      # 0 to 1 (likelihood of breakout)
    
    def to_dict(self) -> Dict:
        return {
            'trend_strength': round(self.trend_strength, 4),
            'volatility': round(self.volatility, 4),
            'momentum': round(self.momentum, 4),
            'range_bound': round(self.range_bound, 4),
            'breakout_score': round(self.breakout_score, 4)
        }


class MarketRegimeDetector:
    """
    Detects market regime from price data.
    
    Uses multiple indicators:
    - Trend: EMA slope, price position relative to MAs
    - Volatility: ATR, Bollinger Band width
    - Momentum: RSI, MACD histogram
    - Range: Support/Resistance touches
    
    Low Resource Design:
    - Pure numpy operations (no sklearn/tensorflow)
    - Rolling calculations (no full history needed)
    - Configurable lookback periods
    """
    
    def __init__(
        self,
        trend_period: int = 20,
        volatility_period: int = 14,
        momentum_period: int = 14,
        regime_threshold: float = 0.6
    ):
        """
        Initialize Regime Detector.
        
        Args:
            trend_period: Lookback for trend calculation
            volatility_period: Lookback for volatility (ATR)
            momentum_period: Lookback for momentum (RSI)
            regime_threshold: Confidence threshold for regime classification
        """
        self.trend_period = trend_period
        self.volatility_period = volatility_period
        self.momentum_period = momentum_period
        self.regime_threshold = regime_threshold
        
        # History buffers
        self.price_history = deque(maxlen=max(trend_period, 50) * 2)
        self.high_history = deque(maxlen=volatility_period * 2)
        self.low_history = deque(maxlen=volatility_period * 2)
        
        # Regime tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_start_step = 0
        self.regime_history: List[Tuple[int, MarketRegime]] = []
        self.step_count = 0
        
        # Volatility normalization
        self.volatility_mean = 0.0
        self.volatility_std = 1.0
    
    def update(
        self,
        close: float,
        high: Optional[float] = None,
        low: Optional[float] = None
    ) -> Tuple[MarketRegime, RegimeMetrics]:
        """
        Update with new price data and detect current regime.
        
        Args:
            close: Close price
            high: High price (optional, uses close if not provided)
            low: Low price (optional, uses close if not provided)
            
        Returns:
            Tuple of (current_regime, metrics)
        """
        self.step_count += 1
        
        # Store prices
        self.price_history.append(close)
        self.high_history.append(high if high is not None else close)
        self.low_history.append(low if low is not None else close)
        
        # Need minimum data
        if len(self.price_history) < self.trend_period:
            return MarketRegime.UNKNOWN, RegimeMetrics(0, 0, 0.5, 0, 0)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Classify regime
        new_regime = self._classify_regime(metrics)
        
        # Track regime changes
        if new_regime != self.current_regime:
            self.regime_history.append((self.step_count, new_regime))
            self.regime_start_step = self.step_count
            self.current_regime = new_regime
        
        return self.current_regime, metrics
    
    def _calculate_metrics(self) -> RegimeMetrics:
        """Calculate all regime metrics from price history."""
        prices = np.array(self.price_history)
        highs = np.array(self.high_history)
        lows = np.array(self.low_history)
        
        # 1. Trend Strength (-1 to +1)
        trend_strength = self._calculate_trend_strength(prices)
        
        # 2. Volatility (0 to 1, normalized)
        volatility = self._calculate_volatility(prices, highs, lows)
        
        # 3. Momentum (0 to 1)
        momentum = self._calculate_momentum(prices)
        
        # 4. Range-Bound Score (0 to 1)
        range_bound = self._calculate_range_bound(prices)
        
        # 5. Breakout Score (0 to 1)
        breakout_score = self._calculate_breakout_score(prices, volatility)
        
        return RegimeMetrics(
            trend_strength=trend_strength,
            volatility=volatility,
            momentum=momentum,
            range_bound=range_bound,
            breakout_score=breakout_score
        )
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """
        Calculate trend strength using linear regression slope.
        
        Returns: -1 (strong downtrend) to +1 (strong uptrend)
        """
        n = min(len(prices), self.trend_period)
        recent = prices[-n:]
        
        # Linear regression slope
        x = np.arange(n)
        slope, _ = np.polyfit(x, recent, 1)
        
        # Normalize by price range
        price_range = np.ptp(recent) + 1e-10
        normalized_slope = slope * n / price_range
        
        # Clip to -1 to +1
        return np.clip(normalized_slope, -1, 1)
    
    def _calculate_volatility(
        self,
        prices: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> float:
        """
        Calculate normalized volatility using ATR.
        
        Returns: 0 to 1 (normalized)
        """
        n = min(len(prices), self.volatility_period)
        
        if n < 2:
            return 0.5
        
        # True Range
        tr = np.maximum(
            highs[-n:] - lows[-n:],
            np.maximum(
                np.abs(highs[-n:] - np.roll(prices[-n:], 1)),
                np.abs(lows[-n:] - np.roll(prices[-n:], 1))
            )
        )
        tr[0] = highs[-n] - lows[-n]  # First element fix
        
        # ATR
        atr = np.mean(tr)
        
        # Normalize by price
        normalized_atr = atr / (np.mean(prices[-n:]) + 1e-10)
        
        # Update running stats for normalization
        alpha = 0.01
        self.volatility_mean = self.volatility_mean * (1 - alpha) + normalized_atr * alpha
        self.volatility_std = max(self.volatility_std * (1 - alpha) + 
                                   abs(normalized_atr - self.volatility_mean) * alpha, 1e-10)
        
        # Z-score normalization, then sigmoid to 0-1
        z_score = (normalized_atr - self.volatility_mean) / self.volatility_std
        volatility = 1 / (1 + np.exp(-z_score))  # Sigmoid
        
        return volatility
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """
        Calculate momentum using RSI-like indicator.
        
        Returns: 0 to 1 (0.5 = neutral)
        """
        n = min(len(prices), self.momentum_period)
        
        if n < 2:
            return 0.5
        
        # Price changes
        changes = np.diff(prices[-n:])
        
        # Separate gains and losses
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        # RSI
        if avg_loss == 0:
            rsi = 1.0 if avg_gain > 0 else 0.5
        else:
            rs = avg_gain / avg_loss
            rsi = 1 - (1 / (1 + rs))
        
        return rsi
    
    def _calculate_range_bound(self, prices: np.ndarray) -> float:
        """
        Calculate how range-bound the price is.
        
        Returns: 0 (trending) to 1 (range-bound)
        """
        n = min(len(prices), self.trend_period)
        recent = prices[-n:]
        
        # Calculate price oscillation around mean
        mean_price = np.mean(recent)
        deviations = recent - mean_price
        
        # Count zero-crossings (more = more range-bound)
        sign_changes = np.sum(np.abs(np.diff(np.sign(deviations)))) / 2
        crossing_ratio = sign_changes / (n - 1)
        
        # High-low range vs actual price path
        price_range = np.ptp(recent)
        total_movement = np.sum(np.abs(np.diff(recent)))
        
        efficiency = price_range / (total_movement + 1e-10)
        
        # Low efficiency + high crossings = range-bound
        range_score = (1 - efficiency) * crossing_ratio
        
        return np.clip(range_score * 2, 0, 1)  # Scale up
    
    def _calculate_breakout_score(self, prices: np.ndarray, volatility: float) -> float:
        """
        Calculate breakout likelihood.
        
        Returns: 0 to 1 (1 = likely breakout)
        """
        n = min(len(prices), self.trend_period)
        recent = prices[-n:]
        
        # Bollinger Band squeeze detection
        if n < 10:
            return 0.0
        
        # Calculate BB width
        ma = np.mean(recent)
        std = np.std(recent)
        bb_width = 2 * std / (ma + 1e-10)
        
        # Squeeze = narrow bands = potential breakout
        # Compare to recent average BB width
        if len(prices) >= n * 2:
            older = prices[-n*2:-n]
            older_std = np.std(older)
            older_bb_width = 2 * older_std / (np.mean(older) + 1e-10)
            squeeze_ratio = bb_width / (older_bb_width + 1e-10)
        else:
            squeeze_ratio = 1.0
        
        # Squeeze + expanding volatility = breakout
        squeeze_score = np.clip(1 - squeeze_ratio, 0, 1)
        
        # Recent price near bands
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std
        current_price = recent[-1]
        
        band_touch = max(
            1 - abs(current_price - upper_band) / (std + 1e-10),
            1 - abs(current_price - lower_band) / (std + 1e-10)
        )
        band_touch = np.clip(band_touch, 0, 1)
        
        breakout_score = (squeeze_score * 0.5 + band_touch * 0.3 + volatility * 0.2)
        
        return np.clip(breakout_score, 0, 1)
    
    def _classify_regime(self, metrics: RegimeMetrics) -> MarketRegime:
        """Classify market regime from metrics."""
        
        # Breakout detection (highest priority)
        if metrics.breakout_score > self.regime_threshold:
            return MarketRegime.BREAKOUT
        
        # Volatile regime
        if metrics.volatility > 0.7:
            return MarketRegime.VOLATILE
        
        # Trending regimes
        if metrics.trend_strength > 0.3 and metrics.range_bound < 0.4:
            return MarketRegime.BULL
        elif metrics.trend_strength < -0.3 and metrics.range_bound < 0.4:
            return MarketRegime.BEAR
        
        # Range-bound
        if metrics.range_bound > 0.5:
            return MarketRegime.SIDEWAYS
        
        # Mixed/unclear
        if abs(metrics.trend_strength) < 0.2:
            return MarketRegime.SIDEWAYS
        
        # Weak trend with momentum confirmation
        if metrics.trend_strength > 0.1 and metrics.momentum > 0.55:
            return MarketRegime.BULL
        elif metrics.trend_strength < -0.1 and metrics.momentum < 0.45:
            return MarketRegime.BEAR
        
        return MarketRegime.SIDEWAYS
    
    def get_regime_duration(self) -> int:
        """Get number of steps in current regime."""
        return self.step_count - self.regime_start_step
    
    def get_regime_probabilities(self, metrics: RegimeMetrics) -> Dict[str, float]:
        """
        Get probability distribution over regimes.
        
        Useful for soft regime classification.
        """
        scores = {}
        
        # Bull score
        scores[MarketRegime.BULL.value] = max(0, metrics.trend_strength) * (1 - metrics.range_bound)
        
        # Bear score
        scores[MarketRegime.BEAR.value] = max(0, -metrics.trend_strength) * (1 - metrics.range_bound)
        
        # Sideways score
        scores[MarketRegime.SIDEWAYS.value] = metrics.range_bound * (1 - metrics.volatility)
        
        # Volatile score
        scores[MarketRegime.VOLATILE.value] = metrics.volatility * (1 - metrics.range_bound)
        
        # Breakout score
        scores[MarketRegime.BREAKOUT.value] = metrics.breakout_score
        
        # Normalize to probabilities
        total = sum(scores.values()) + 1e-10
        probs = {k: v / total for k, v in scores.items()}
        
        return probs
    
    def get_stats(self) -> Dict:
        """Get detector statistics."""
        regime_counts = {}
        for _, regime in self.regime_history:
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
        
        return {
            'current_regime': self.current_regime.value,
            'regime_duration': self.get_regime_duration(),
            'total_steps': self.step_count,
            'regime_changes': len(self.regime_history),
            'regime_counts': regime_counts
        }
    
    def reset(self):
        """Reset detector state."""
        self.price_history.clear()
        self.high_history.clear()
        self.low_history.clear()
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_start_step = 0
        self.regime_history.clear()
        self.step_count = 0


def detect_regime_batch(
    prices: np.ndarray,
    highs: Optional[np.ndarray] = None,
    lows: Optional[np.ndarray] = None,
    **kwargs
) -> List[Tuple[int, MarketRegime, RegimeMetrics]]:
    """
    Detect regimes for a batch of prices (convenience function).
    
    Args:
        prices: Array of close prices
        highs: Optional array of high prices
        lows: Optional array of low prices
        **kwargs: Arguments for MarketRegimeDetector
        
    Returns:
        List of (step, regime, metrics) tuples
    """
    detector = MarketRegimeDetector(**kwargs)
    results = []
    
    for i, price in enumerate(prices):
        high = highs[i] if highs is not None else None
        low = lows[i] if lows is not None else None
        
        regime, metrics = detector.update(price, high, low)
        results.append((i, regime, metrics))
    
    return results
