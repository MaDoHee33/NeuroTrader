# 🧠 NeuroTrader V4 (AutoPilot Edition)
**Advanced Agentic Trading System with Automated Training Pipeline**

NeuroTrader V4 evolves into a fully **Automated Training System** with Model Versioning, Auto-Evaluation, and Config-Driven Pipelines.

---

## 🏗️ Architecture: The Trinity System

We deploy 3 specialized agents, each with unique reward functions and data horizons:

| Agent Role | Timeframe | Strategy | Reward Logic |
| :--- | :--- | :--- | :--- |
| **⚔️ Scalper** | M5 | Hit & Run | High PnL + **Time Penalty** (Force Short Holding) |
| **🛡️ Swing** | H1 | Trend Waves | Hybrid (PnL + Trend Following) |
| **👑 Trend** | D1 | Wealth Gen | Buy & Hold (Sharpe Ratio + Drawdown Penalty) |

---

## ⚡ Key Features (V4)

### 1. AutoPilot Training System (`src/skills/`)
-   **🗂️ Model Registry**: Versioned model storage with auto-promotion
-   **📊 Auto-Evaluator**: Automatic post-training evaluation and comparison
-   **🎛️ Training Orchestrator**: Config-driven training with auto-resume
-   **🔔 Notifier**: Telegram/Discord alerts for training events

### 2. Autonomous Skills
-   **📰 News Watcher**: Blocks trades 30 mins before high-impact news
-   **📝 Auto Reporter**: Generates Markdown/PDF performance reports

### 3. Hyperparameter Tuning (`scripts/tune_trinity.py`)
-   Powered by **Optuna**
-   Includes **Behavioral Penalties** (e.g., punishing Scalper for holding > 1 hour)

---

## 🧬 V6: Self-Evolving AI (NEW!)

### Hybrid Development Approach
Combining PPO baseline with Self-Evolving AI capabilities for continuous learning.

### New Modules (`src/evolving/`)
| Module | Description |
|--------|-------------|
| **🔍 CuriosityModule** | Intrinsic rewards for exploration (ICM-based) |
| **📚 ExperienceBuffer** | Lifelong learning memory with priority eviction |
| **📈 CurriculumManager** | Progressive difficulty scaling |
| **🌡️ MarketRegimeDetector** | Real-time market condition classification |
| **🤖 HybridTradingAgent** | Integration of PPO + Self-Evolving components |

### Usage Example
```python
from src.evolving import HybridTradingAgent

agent = HybridTradingAgent(
    ppo_model_path='models/scalper/best.zip',
    use_curiosity=True,
    use_curriculum=True
)

obs = env.reset()
while not done:
    action, info = agent.get_action(obs)
    next_obs, reward, done, _, env_info = env.step(action)
    agent.store_experience(obs, action, reward, next_obs, env_info)
    obs = next_obs
```

See [HYBRID_DEVELOPMENT_PLAN.md](docs/HYBRID_DEVELOPMENT_PLAN.md) for full roadmap.

---

## 🚀 Quick Start

### Using AutoPilot CLI (Recommended)
```powershell
# Train all roles with auto-resume
python scripts/autopilot.py train --all

# Quick train single role
python scripts/autopilot.py quick --role scalper

# Resume interrupted training
python scripts/autopilot.py resume

# Show model status
python scripts/autopilot.py status

# Compare model versions
python scripts/autopilot.py compare --role scalper
```

### Direct Scripts
```powershell
# Train with checkpoints
python scripts/train_trinity.py --role scalper --data data/raw/XAUUSDm_M5_raw.parquet --steps 500000

# Hyperparameter tuning
python scripts/tune_trinity.py --role scalper --data data/processed/XAUUSD_M5_processed.parquet --trials 20

# Backtest with reporting
python scripts/backtest_trinity.py --role scalper --data data/processed/XAUUSD_M5_processed.parquet --model models/scalper/best/model.zip
```

---

## 📁 Project Structure
```
NeuroTrader/
├── src/
│   ├── brain/          # RL Agents, Environment, Features
│   ├── body/           # MT5 Driver
│   ├── skills/         # AutoPilot Components
│   ├── evolving/       # 🆕 Self-Evolving AI Modules
│   └── utils/          # Utilities
├── scripts/            # CLI Scripts
├── config/             # YAML Configurations
├── models/             # Trained Models (Versioned)
├── data/               # Processed Data
├── tests/              # Unit Tests
├── examples/           # Usage Examples
└── docs/               # Documentation
```

---

## 📚 Documentation
-   [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Complete system documentation (Thai)
-   [HYBRID_DEVELOPMENT_PLAN.md](docs/HYBRID_DEVELOPMENT_PLAN.md) - Self-Evolving AI Roadmap
-   [RL_ALGORITHMS_TH.md](docs/RL_ALGORITHMS_TH.md) - RL algorithms comparison (Thai)
-   [DEV_JOURNAL.md](DEV_JOURNAL.md) - Development history

---

**Status:** ✅ V2.7 Scalper Training | 🧬 Self-Evolving AI Phase 1 | **OS:** Windows Native

## ⚡ Version History

### V2.7 Scalper (Current Training)
- Steeper time decay (0.04/step from 4 bars)
- Higher entry bonus (0.08)
- Speed bonus for exits < 12 steps

### V2.1 Unified Engine
**Unified Feature Engine**: Solved training/inference divergence.
-   **One Logic Rule**: `src/brain/features.py` handles ALL indicators.
-   **Qwen-Validated**: Architecture approved by Qwen-3 Coder (Cloud) audit.

### V5 Optimization
1.  **Fast Trainer (`src/brain/experimental/train_fast.py`)**: Parallel environment training (4-8x faster).
2.  **Unified Config (`config/hyperparameters.yaml`)**: Centralized settings for easy experimentation.
3.  **Robust Features (`src/brain/features.py`)**: Consolidated feature engineering with better NaN handling.

### V6 Self-Evolving AI (In Progress)
Hybrid approach combining PPO with curiosity-driven exploration and lifelong learning.

