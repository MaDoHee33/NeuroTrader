# 🗺️ NeuroTrader Project Roadmap & Protocol

> **🛑 AGENT INSTRUCTION: READ THIS FIRST**
> Every agent working on this project **MUST** read this file at the start of the session.
> 1.  **Understand the Vision**: Know where we are going.
> 2.  **Check Status**: Look at the [Current Status](#-current-status-checklist) section to see what is currently "In Progress".
> 3.  **Update Progress**: Before you finish your session, you **MUST** update the checklist to reflect completed tasks.

---

## 🏛️ Project Vision
**NeuroTrader** is a fully autonomous, hybrid algorithmic trading system that combines:
1.  **Traditional Quant**: Robust execution and backtesting via `NautilusTrader`.
2.  **AI/ML**: Reinforcement Learning (RL) and LLM-based decision making.
3.  **Self-Evolving AI**: Curiosity-driven exploration with lifelong learning.
4.  **Automation**: Self-correcting pipelines that run 24/7 on Hybrid Linux/Windows Systemd.

### 🏗️ Architecture (The "3-Tier Automation")
*   **Tier 1: Always-On Core** (`neuro-trader.service`) - The execution engine running `src/main.py`.
*   **Tier 2: Data Stream** (`neuro-data.service`) - Continuous market data fetching via `scripts/start_data.sh`.
*   **Tier 3: Maintenance Loop** (`scripts/weekly_maintenance.sh`) - Weekly retraining and model evolution.

---

## 🚀 Development Roadmap

### ✅ Phase 1: Foundation (Completed)
- [x] Set up Project Structure (`src/body`, `src/brain`, `src/data`).
- [x] Integrate `NautilusTrader` for Backtesting.
- [x] Integrate `MetaTrader 5` (via `mt5windows` and `mt5linux`) for Market Connectivity.
- [x] Implement Basic Data Pipeline (`scripts/update_data.py`).

### ✅ Phase 2: Automation & Infrastructure (Completed)
- [x] Design 3-Tier Automation Architecture.
- [x] Create Systemd Service Files (`autosystem/systemd/`).
- [x] Implement Auto-Restart Scripts (`scripts/start_node.sh`).
- [x] Create Watchdog/Health-Check Scripts (`scripts/check_health.py`).

### ✅ Phase 3: Intelligence & Strategy (Completed)
- [x] **Data Quality**: Ensure `scripts/update_data.py` handles gap-filling perfectly.
- [x] **Strategy Logic**: Connect `Brain` (AI) signals to `NeuroBridgeStrategy` in Nautilus.
- [x] **Risk Management**: Implement Turbulence Index (FinRL) for crash protection.
- [x] **Multi-Agent Architecture**: Upgrade Agent to support Ensemble models.
- [x] **Model Iteration**: Improve the RL Agent (PPO/LSTM) in `src/brain/`.

### 🚧 Phase 4: Trinity System & Optimization (Current)
- [x] **Trinity Architecture**: 3 specialized agents (Scalper, Swing, Trend)
- [x] **Unified Feature Engine**: `src/brain/features.py` with FeatureRegistry
- [x] **Model Registry**: Versioned model storage with auto-promotion
- [x] **Hyperparameter Tuning**: Optuna integration with behavioral penalties
- [/] **V2.7 Scalper Training**: Steeper time decay for faster trades (In Progress)
- [ ] **Paper Trading**: Run system in mock mode for 1 week uninterrupted

### 🧬 Phase 5: Self-Evolving AI (NEW - In Progress)
- [x] **Curiosity Module**: Intrinsic rewards for exploration (`src/evolving/curiosity.py`)
- [x] **Experience Buffer**: Lifelong learning memory (`src/evolving/experience_buffer.py`)
- [x] **Curriculum Manager**: Progressive difficulty scaling (`src/evolving/difficulty_scaler.py`)
- [x] **Market Regime Detector**: Real-time condition classification (`src/evolving/regime_detector.py`)
- [x] **Hybrid Agent**: Integration wrapper (`src/evolving/hybrid_agent.py`)
- [x] **Unit Tests**: Test suite (`tests/test_evolving.py`)
- [ ] **Integration Test**: Run with V2.7 PPO model
- [ ] **Adaptive Learning**: Meta-learning for market regime adaptation
- [ ] **Production Deployment**: Live paper trading with Self-Evolving capabilities

### 🔮 Phase 6: Interface & Scaling (Future)
- [ ] **Dashboard**: A Web UI (Streamlit/FastAPI) to view live stats (reading `data/status.json`).
- [ ] **Notifications**: Enhanced Discord alerts (Trade execution, Daily PnL).
- [ ] **Multi-Asset**: Scale from XAUUSD to multiple pairs.

---

## 📝 Current Status Checklist
*Updated: 2026-01-28*

### 🟢 Completed Today
- [x] **Hybrid Development Plan**: Created `docs/HYBRID_DEVELOPMENT_PLAN.md`
- [x] **Self-Evolving Modules**: 5 new modules in `src/evolving/`
- [x] **V2.7 Scalper Config**: Updated reward function for faster trading
- [x] **Unit Tests**: Created `tests/test_evolving.py`
- [x] **Usage Examples**: Created `examples/evolving_usage.py`

### 🟡 In Progress
- [/] **V2.7 Scalper Training**: 500k steps (~25% complete)
- [/] **Documentation Update**: README, ROADMAP, DEV_JOURNAL

### 🔵 Next Up
- [ ] **Backtest V2.7**: Evaluate holding time metrics
- [ ] **Integration Test**: HybridTradingAgent with V2.7 model
- [ ] **Market Regime Integration**: Connect detector to HybridAgent

---

## 💾 Technical Context
*   **OS**: Windows (Primary), Linux (Future Deployment)
*   **Python**: 3.10+ (Virtualenv `.venv`)
*   **Key Libs**: `stable_baselines3`, `sb3-contrib`, `gymnasium`, `pandas`, `numpy`
*   **Config**: `config/training_config.yaml`
*   **Self-Evolving**: `src/evolving/` (Low resource, numpy-only design)

