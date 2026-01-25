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
3.  **Automation**: Self-correcting pipelines that run 24/7 on Hybrid Linux/Windows Systemd.

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

### 🚧 Phase 3: Intelligence & Strategy (Current Focus)
- [ ] **Data Quality**: Ensure `scripts/update_data.py` handles gap-filling perfectly.
- [x] **Strategy Logic**: Connect `Brain` (AI) signals to `NeuroBridgeStrategy` in Nautilus.
- [x] **Risk Management**: Implement Turbulence Index (FinRL) for crash protection.
- [x] **Multi-Agent Architecture**: Upgrade Agent to support Ensemble models.
- [ ] **Model Iteration**: Improve the RL Agent (PPO/LSTM) in `src/brain/`.
- [ ] **Live Test**: Run the system in "Paper Mode" (Mock Money) for 1 week uninterrupted.

### 🔮 Phase 4: Interface & Scaling (Future)
- [ ] **Dashboard**: A Web UI (Streamlit/FastAPI) to view live stats (reading `data/status.json`).
- [ ] **Notifications**: Enhanced Discord alerts (Trade execution, Daily PnL).
- [ ] **Multi-Asset**: Scale from XAUUSD to multiple pairs.

---

## 📝 Current Status Checklist
*Updated: 2026-01-19*

### 🔴 Immediate Next Priorities
- [ ] **Fix Data Gap Filling**: `update_data.py` needs to be robust against network drops.
- [ ] **Verify Strategy Connection**: Ensure `runner.py` actually executes trades based on `brain` output.
- [ ] **Deploy Services**: User needs to run `sudo systemctl enable` commands.

### 🟡 In Progress (V2.1 Refactor)
- [x] **DeepSeek Audit**: Identify "Schizophrenic Logic" bug.
- [x] **Unified Feature Engine**: Implement shared logic for Train/Trade.
- [x] **System Cleanup**: Wipe old incompatible models.
- [ ] **Train V2.1**: Retrain Scalper on clean architecture.

---

## 💾 Technical Context
*   **OS**: Linux (Ubuntu/Debian)
*   **Python**: 3.10+ (Virtualenv `.venv`)
*   **Key Libs**: `nautilus_trader`, `mt5linux`, `stable_baselines3`, `pandas`
*   **Config**: `config/main_config.yaml`
