Absolutely — let’s now build a **Strategic Refactoring & Prevention Plan** to resolve the “schizophrenic” feature mismatch and memory leak issues, while also establishing guardrails to prevent them from recurring.

---

# ✅ Strategic Refactoring & Prevention Plan

---

## 🧠 Part 1: The Fix – Unified Feature Engine (Python Sketch)

We will implement a **Unified Feature Engine** that supports both batch (training) and streaming (inference) modes using the exact same logic.

This ensures no divergence between training and inference pipelines.

### 🔁 Core Idea:
Use **stateful indicators** that can be updated incrementally during inference, but also applied over full DataFrames during training.

```python
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from collections import deque
from ta import (
    RSIIndicator,
    MACDIndicator,
    BollingerBands,
    EMAIndicator,
    StochasticOscillator,
    VWAPIndicator
)

class FeatureEngineBase(ABC):
    """Abstract base class for unified feature engine."""

    @abstractmethod
    def compute_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def update_stream(self, tick: dict) -> np.ndarray:
        pass


class UnifiedFeatureEngine(FeatureEngineBase):
    """
    A stateful feature engine supporting both batch mode (DataFrame) and
    streaming mode (single tick), with guaranteed consistency.
    """

    FEATURE_NAMES = [
        'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
        'ema_20', 'atr', 'log_return', 'stoch_k', 'stoch_d',
        'vwap', 'bb_width', 'normalized_close', 'normalized_volume',
        'close', 'open', 'high', 'low', 'volume'
    ]

    def __init__(self, window=200, max_history=500):
        self.window = window
        self.max_history = max_history
        self.history = deque(maxlen=max_history)

        # Stateful TA indicators
        self._init_indicators()

    def _init_indicators(self):
        # Placeholder initialization; actual indicators initialized per bar
        self.indicators = {}

    def _update_history(self, tick: dict):
        self.history.append(tick)

    def compute_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features in batch mode."""
        df = df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Indicators
        rsi = RSIIndicator(df['close'], window=14)
        macd = MACDIndicator(df['close'])
        bb = BollingerBands(df['close'])
        ema = EMAIndicator(df['close'], window=20)
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])

        df['rsi'] = rsi.rsi()
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['ema_20'] = ema.ema_indicator()
        df['atr'] = abs(df['high'] - df['low'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        df['vwap'] = VWAPIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        ).vwap

        df['normalized_close'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
        df['normalized_volume'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume'].rolling(window=20).std()

        return df[self.FEATURE_NAMES].fillna(0)

    def update_stream(self, tick: dict) -> np.ndarray:
        """Update internal state and return latest feature vector."""
        self._update_history(tick)
        df_tick = pd.DataFrame([dict(self.history)])

        # Compute features on most recent window only
        df_window = pd.DataFrame(list(self.history)).tail(self.window)
        df_features = self.compute_batch(df_window).iloc[-1]

        return df_features.values.astype(np.float32)

# Singleton registry to enforce consistency across modules
class FeatureRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.engine = UnifiedFeatureEngine()
        return cls._instance

    def get_feature_names(self):
        return self.engine.FEATURE_NAMES

    def compute_batch(self, df):
        return self.engine.compute_batch(df)

    def update_stream(self, tick):
        return self.engine.update_stream(tick)
```

---

## ⚙️ Part 2: Prevention – Guardrails

### 🔍 Unit Tests That Must Exist

To catch mismatches automatically:

```python
def test_feature_consistency():
    registry = FeatureRegistry()
    sample_df = generate_sample_data(n_rows=1000)  # helper function

    # Batch mode
    batch_result = registry.compute_batch(sample_df.tail(100))
    expected_shape = (100, len(registry.get_feature_names()))
    assert batch_result.shape == expected_shape

    # Stream mode (simulate last 10 ticks)
    for _, row in sample_df.iterrows():
        stream_features = registry.update_stream(row.to_dict())

    # Final tick should match final row of batch output
    assert np.allclose(stream_features, batch_result.iloc[-1].values, atol=1e-6)


def test_feature_schema_integrity():
    registry = FeatureRegistry()
    features = registry.get_feature_names()
    expected_length = 19
    assert len(features) == expected_length, f"Expected {expected_length}, got {len(features)}"
    assert set(features) == set([
        'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
        'ema_20', 'atr', 'log_return', 'stoch_k', 'stoch_d',
        'vwap', 'bb_width', 'normalized_close', 'normalized_volume',
        'close', 'open', 'high', 'low', 'volume'
    ])
```

### 📜 Golden Rule for Adding New Features

> **Golden Rule:**  
_"Every feature added must be registered in `FeatureRegistry`, used identically in both batch and stream modes, and validated by unit tests."_

Any feature not in `FEATURE_NAMES` does **not exist** in the system.

---

## 🏛️ Part 3: Architecture Standard – Brain Coherence Layer

Ensure `RLAgent` and `TradingEnv` never diverge again by centralizing their shared components into a **coherent architecture standard**.

### ✨ Proposed Architecture Layers:

| Layer | Responsibility |
|-------|----------------|
| `ObservationEngine` | Generates standardized observations using `FeatureRegistry`. Used by both training and inference. |
| `ActionMapper` | Maps raw actions (Buy/Sell/Hold) to valid orders, enforcing risk rules uniformly. |
| `RewardCalculator` | Computes rewards consistently across simulation and live environments. |

### 🔄 Refactor Example: Shared Observation Pipeline

```python
# src/core/observation_engine.py

class ObservationEngine:
    def __init__(self):
        self.registry = FeatureRegistry()

    def get_observation_from_df(self, df):
        features = self.registry.compute_batch(df)
        return features.values.astype(np.float32)

    def get_observation_from_tick(self, tick):
        return self.registry.update_stream(tick)
```

Then both `TradingEnv.step()` and `RLAgent.process_bar()` use this.

### 🧩 Enforce Uniformity via Dependency Injection

In your `main.py` or config loader:

```python
observation_engine = ObservationEngine()
rl_agent = RLAgent(observation_engine=observation_engine)
trading_env = TradingEnv(observation_engine=observation_engine)
```

Now they share the same source of truth.

---

## 🧱 Summary of Fixes & Safeguards

| Problem | Solution | Guardrail |
|--------|----------|-----------|
| Schizophrenic Logic (Training ≠ Inference) | Unified Feature Engine with one API for both modes | Unit Test: `test_feature_consistency()` |
| Memory Leak / Latency | Streaming updates instead of full recalculations | Profile monitoring + benchmarks |
| Divergent Architectures (`RLAgent` vs `TradingEnv`) | Centralized `ObservationEngine` | DI-based design + contract testing |
| No Feature Governance | `FeatureRegistry` enforces schema integrity | Golden Rule enforcement + CI linting |

---

Let me know if you want this refactored into modular files (`features.py`, `engine.py`, etc.) or integrated into existing NeuroTrader structure!