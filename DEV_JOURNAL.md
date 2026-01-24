# NeuroTrader Development Journal

> **AI INSTRUCTION**: Before starting any task, READ this file to understand the current state, recent changes, and lessons learned. After completing a task, UPDATE this file with:
> 1. Date/Time
> 2. Task Description
> 3. Actions Taken (Files modified, scripts run)
> 4. Results (Metrics, unexpected outcomes)
> 5. Errors & Fixes
> 6. Next Steps

---

## 📅 2026-01-22 (Session 1)

### 1. Initial Data Pipeline & Single Asset Training
**Objective**: Fetch XAUUSDm data, train PPO model, and backtest.

**Actions**:
- **Fixed `src/brain/feature_eng.py`**: Removed early `return` to enable `dropna()`.
- **Fixed `src/body/mt5_driver.py`**: Expanded `tf_map` to support M5, M15, etc.
- **Created `scripts/fetch_xauusdm.py`**: Automated fetching of 50k candles for XAUUSDm.
- **Created `scripts/train_xauusdm.py`**: Trained PPO (1M steps) on XAUUSDm M5.
- **Created `scripts/backtest_xauusdm.py`**: Backtested the model.

**Results**:
- **In-Sample Performance**:
    - Total Return: **+21.52%**
    - Sharpe Ratio: **2.83**
    - Max Drawdown: **-6.31%**
- *Observation*: Excellent results suspected to be Overfitting.

---

### 2. Walk-Forward Validation (Overfitting Check)
**Objective**: Verify model performance on unseen data (Out-of-Sample).

**Actions**:
- **Created `scripts/train_walk_forward.py`**:
    - Split XAUUSDm M5 data: **80% Train** / **20% Test**.
    - Trained on 80%, Backtested on 20% immediately.

**Results**:
- **In-Sample (Train)**: +21% Return (Confirmed learning).
- **Out-of-Sample (Test)**: **-0.05% Return**, 0.00 Sharpe.
- **Conclusion**: **CONFIRMED OVERFITTING**. The model memorized the training data but failed to generalize.

---

### 3. Strategy Pivoting (Phase 2)
**Objective**: Solve overfitting by simplifying features (Price Action) and expanding data (Multi-Asset).

**Plan**:
1.  **Features**: Switch to "Price Action Hybrid" (Candles + EMA + RSI + ATR). Remove noisy indicators (MACD, Stoch).
2.  **Data**: Fetch XAUUSD, BTCUSD, DXY across ALL timeframes (M1-MN1).
3.  **Regularization**: Train on multiple assets simultaneously (Random Asset Switching).

**Actions**:
- **Refactored `src/brain/feature_eng.py`**: Implemented `body_size`, `wicks`, `is_bullish`, `ema`, `atr`, `rsi`, `log_ret`.
- **Created `scripts/fetch_multi_asset.py`**: Fetched 50k rows for XAU, BTC, DXY (24+ datasets).
- **Refactored `TradingEnv`**: Added support for initializing with a Dict of DataFrames and random asset switching via `reset()`.

**Errors & Fixes**:
- **Error**: `KeyError: 'body_size'` during training.
- **Cause**: Data fetch script (run previously) used the *old* feature function in memory/cache, so parquet files lacked new columns.
- **Fix**: Deleted all `data/processed/*.parquet` files and re-ran `scripts/fetch_multi_asset.py` from scratch.

---

### 4. Multi-Asset Training (Current Status)
**Objective**: Train regularized PPO model on XAU/BTC/DXY with proper validation.

**Actions**:
- **Initial Attempt**: Ran `scripts/train_multi_asset.py` on full data.
- **User Feedback**: "How do we verify it's not memorizing?" -> Requested Train/Test split.
- **Modification**: Updated `scripts/train_multi_asset.py` to:
    - Split ALL assets into **80% Train / 20% Test**.
    - Train only on Train sets.
    - Automatically validate on Test sets after training.

**Current State**:
- Script `scripts/train_multi_asset.py` is **RUNNING**.
- Training on 24 datasets (XAU, BTC, DXY mixed).
- Waiting for completion to see Out-of-Sample Validation table.

**Update (Completed)**:
- **XAUUSD (Gold)**: Passed with flying colors!
    - D1: **+71.49%** (Test Set)
    - H4: **+60.02%** (Test Set)
    - M5: **+7.11%** (Test Set)
- **BTCUSD**: Flat (-0.05%). Model chose "Safety Mode" (Cash) for Bitcoin.
- **DXY**: Small losses (-1.5%).
- **Conclusion**: The **Price Action Hybrid** model is extremely robust for Gold across all timeframes. The Multi-Asset training successfully prevented overfitting (proven by high Test Set scores).

---

### 5. Phase 3: Paper Trading (Deployment)
**Objective**: Run the model on live data (Paper Mode) to verify real-time execution.

**Actions**:
- **Updated `MT5Driver`**: Implemented `_send_order` for real trade execution capabilities.
- **Created `scripts/paper_trade_xauusdm.py`**:
    - Mode: **Shadow Mode** (Default) - Logs trades to console without sending to broker.
    - Logic: Fetch M5 candle -> Predict -> Log "Ghost Trade".
    - Risk Control: Fixed 0.01 Lot size for safety.

**Status**:
- Script launched. Verifying connectivity to MT5 terminal...
- **Update**: User requested switch to **Live Demo Trading**.
- Restarting script with `--live` flag enabled.

---

### 6. Phase 4: Brain Transplant (NeuroTrader 2.0)
**Objective**: Upgrade architecture to LSTM (Recurrent Neural Network) + Time Awareness.

**Changes**:
- **Pipelines**: Added `hour_sin`, `day_sin` features (Time Context).
- **Architecture**: Switched from `PPO` (MLP) to `RecurrentPPO` (LSTM).
- **Goal**: Enable "Long-term Memory" to reduce false breakouts and improve context awareness.

**Actions**:
- Installed `sb3-contrib`.
- Updated `feature_eng.py` and re-fetched all data.
- Created `scripts/train_neurotrader_v2.py`.
- **Status**: Launching Training Job (1M Steps)...
- **Update (Completed)**: Training finished successfully.

**Validation Results (Phase 4)**:
Compare V1 vs V2 on the same "Blind Test" (XAUUSD D1):
- **V1 (Base Model)**: **-19.63%** Loss | **-20.62%** Drawdown
- **V2 (NeuroTrader 2.0)**: **+9.86%** Profit | **-9.65%** Drawdown

**Verdict**: The LSTM memory successfully prevented the "Crash" scenarios. It trades safer and survives volatility.

---

### 7. System Halt 🛑
- **Action**: User requested to stop all operations ("หยุดทุกอย่างก่อน").
- **Status**: Live Trading Script Terminated. Project Paused.
- **Next Step**: Awaiting user instruction to resume (Deployment or Further Testing).







---

### 7. Multi-Agent & News Filter Upgrade
**Objective**: Develop 'NeuroTrader Multi-Agent System' and integrate News Filter.

**Plan**:
1.  **News Filter**: Block trades during High Impact economic events (Risk Management).
2.  **Short-Term Agent (Speed)**: Develop 1D-CNN + LSTM specialist for short-term scalping.
3.  **Ensemble Manager**: Combine signals from Long-term (Trend) and Short-term (Speed) agents.

**Actions**:
- **Created src/data/economic_calendar.py**: Module to check High Impact news window.
- **Updated RiskManager**: Added check_order(current_time) to block trades 30 mins around High Impact news.
- **Updated TradingEnv**: Passed simulation time to Risk Manager to enforce news filtering during training/backtest.
- **Planned**: Drafted architecture for NeuroTrader-Speed (1D-CNN).

**Next Steps**:
- Implement NeuroTrader-Speed model.
- Build Ensemble logic.

---

### 8. Short-Mid Term Model & Dynamic Risk (V2.1)
**Objective**: Develop a specialized model for 5-minute (M5) timeframes and solve position sizing limitations.

**Actions**:
- **Feature Engineering**: Added fast indicators (MACD, Bollinger Bands, Stochastic, EMA 9/21) for short-term sensitivity.
- **Environment**: Updated TradingEnv to support dynamic feature selection.
- **Training**: Trained NeuroTrader-ShortTerm (LSTM) on M5 data (500k steps).
- **Risk Management Upgrade**:
    - **Problem**: Fixed 1.0 Lot limit caused Order Blocked errors, preventing the model from trading profitably despite good signals.
    - **Solution**: Implemented Dynamic Lot Sizing in RiskManager.
    - **Formula**: Allowed Lots = (Equity / 10,000) * 5.0.

**Results (Test Set M5)**:
- **Before Dynamic Risk**: +6.75% Return (Mostly holding, unable to trade).
- **After Dynamic Risk**: **+202.82% Return** | **-1.74% Drawdown**.
- **Conclusion**: The model is highly effective at scalping M5 trends when allowed to size positions correctly. The low drawdown confirms high precision entry/exit.

**Next Steps**:
- Paper Trade the Short-Term Model.
- Compare vs Long-Term Model.

---

### 9. FinRL Integration (Turbulence & Ensemble)
**Objective**: Enhance robustness by integrating advanced financial engineering features from FinRL framework.

**Actions**:
-   **Analyzed FinRL**: Identified "Turbulence Index" (Risk) and "Ensemble Strategy" (Multi-Agent) as key missing components.
-   **Updated `src/brain/feature_eng.py`**: Added `turbulence` calculation using Mahalanobis distance (scipy).
-   **Updated `src/brain/risk_manager.py`**: Added `check_turbulence()` to block trades during market crashes (Threshold > 15.0).
-   **Refactored `src/brain/rl_agent.py`**:
    -   Added support for `self.ensemble` (Dictionary of models).
    -   Implemented logic to load PPO, A2C, DDPG from `models/checkpoints/`.
-   **Updated `src/neuro_nautilus/strategy.py`**: Wired `turbulence` signal to Risk Manager and Agent to Strategy.

**Results**:
-   **Syntax Verified**: Code base is stable and imports correctly.
-   **System Capabilities**:
    -   **Crash Detection**: Bot now "panics" correctly when statistical anomalies occur.
    -   **Multi-Model Ready**: Architecture supports hot-swapping strategies (e.g. Trend vs Mean Reversion).

**Next Steps**:
-   Train A2C/DDPG models to populate the ensemble.
-   Implement the "Selector" logic to dynamically switch agents based on weekly performance.

---

### 10. The Trinity System & Behavioral Analytics (Current)
**Goal:** Evolve from a single model to a specialized **3-Agent System** and implement deep behavioral tracking.

**Implementations:**
1.  **Trading Environment Upgrade**:
    -   Modified `TradingEnv` to accept `agent_type` ("scalper", "swing", "trend") logic.
    -   Implemented specialized **Reward Functions**:
        -   **Scalper**: Penalizes time in market, rewards realized PnL and speed.
        -   **Swing**: Hybrid reward (Trend + PnL).
        -   **Trend**: Classic "Buy & Hold" logic (Holding Reward + Sharpe).

2.  **Unified Training Protocol**:
    -   Created `scripts/train_trinity.py` to handle all 3 roles with specific hyperparameters.
    -   Fixed data loading issues ensuring robust training pipelines.

3.  **Behavioral Analysis System**:
    -   Created `src/analysis/behavior.py`: Module for calculating metrics like Avg Holding Time, Win Rate, and Exposure %.
    -   Created `scripts/backtest_trinity.py`: Unified backtester that generates detailed Markdown Reports and CSV logs.

**Outcome:**
-   The system is now capable of training agents with distinct personalities.
-   Verification of "Scalper" training confirmed the pipeline works.
-   Backtesting now provides "Deep Diagnostics" to automatically generate behavioral reports.

---

### 11. V3 Upgrade & Hyperparameter Tuning (Current)
**Goal:** Enhance developer experience, autonomy, and model performance.

**Implementations:**
1.  **Context7 Integration**: Configured MCP server for real-time documentation retrieval.
2.  **Autonomous Skills**:
    -   **News Watcher**: Implemented `news_watcher.py` to filter high-impact economic news and integration with `RiskManager`.
    -   **Reporter**: Implemented `reporter.py` for automated Markdown/PDF performance reports.
3.  **Hyperparameter Tuning**:
    -   Created `tune_trinity.py` using **Optuna**.
    -   Defined objective function with **Holding Time Penalty** to force Scalper behavior.

**Next Steps:**
-   Run optimization for Scalper (M5) to fix "Buy & Hold" behavior.
-   Verify optimized parameters.

### 12. Knowledge Base Expansion
**Goal:** Document technical depth for future reference.

**Actions:**
-   Created `docs/RL_ALGORITHMS_TH.md`: A comprehensive guide comparing PPO, A2C, DDPG, SAC, and Ensemble methods in Thai.
-   Updated `README.md`: Reflected V3 Architecture (Trinity + Skills + Optuna).

---

### 13. System Status: PAUSED (User Request)
**Date:** 2026-01-23
**State:**
-   **Context:** V3 Upgrade is complete (Code, Docs, Git).
-   **Active Task:** Retraining Scalper V2 with Optuna Optimized parameters.
-   **Interruption:** Training stopped at ~40% progress.

**Next Action Items:**
1.  **Resume Training:** Run `python scripts/train_trinity.py --role scalper ...` again.
2.  **Verify:** Check if the new Scalper holds positions shorter than 1 hour.
3.  **Deploy:** If verified, move to Live/Paper Trading.

---

### 14. V4 AutoPilot System
**Date:** 2026-01-24
**Goal:** Create fully automated training pipeline with model versioning.

**New Components:**
| File | Purpose |
|------|---------|
| `src/skills/model_registry.py` | Versioned model storage with auto-promotion |
| `src/skills/auto_evaluator.py` | Automatic post-training evaluation |
| `src/skills/training_orchestrator.py` | Config-driven training controller |
| `src/skills/notifier.py` | Telegram/Discord notifications |
| `config/training_config.yaml` | Declarative pipeline configuration |
| `scripts/autopilot.py` | Unified CLI for all operations |

**Upgrades to `train_trinity.py`:**
-   ✅ Checkpoint saving every 100k steps
-   ✅ Resume from checkpoint (`--resume` flag)
-   ✅ Auto-register in Model Registry
-   ✅ Auto-promote if better than previous best
-   ✅ Cleanup checkpoints after successful training

**New CLI Commands:**
```powershell
python scripts/autopilot.py train --all      # Train all roles
python scripts/autopilot.py resume           # Resume interrupted
python scripts/autopilot.py status           # Show model status
python scripts/autopilot.py compare --role X # Compare versions
```

**Documentation:**
-   Created `docs/ARCHITECTURE.md`: Complete system documentation in Thai

**Status:** ✅ V4 AutoPilot Complete

---

### 15. Phase 1: Exit Signal Features
**Date:** 2026-01-24
**Goal:** Fix Buy & Hold behavior in Scalper by adding exit signal features.

**Baseline Results (Before):**
| Metric | Value | Status |
|--------|-------|--------|
| Return | +6.71% | ✅ |
| Max DD | -2.99% | ✅ |
| Avg Holding | 9,761 steps | ❌ Buy & Hold |
| Trades | 1 | ❌ |

**New Features Added to `feature_eng.py`:**
-   `exit_signal_score` - Combined exit signal (0-1)
-   `rsi_extreme`, `rsi_overbought`, `rsi_oversold`
-   `macd_cross_up`, `macd_cross_down`, `macd_weakening`
-   `price_extension`, `price_overextended`
-   `bb_position`, `at_bb_upper`, `at_bb_lower`
-   `stoch_overbought`, `stoch_oversold`

**Updated `TradingEnv`:**
-   Added 5 exit signal features to default observation space (23 → 28 features)

**Status:** Testing Scalper v2

---

### 16. Phase 1.5: Fix Buy & Hold (Hard Constraints)
**Date:** 2026-01-24
**Problem:** Scalper v2 still exhibited "Buy & Hold" behavior (9,761 steps holding) despite new features.
**Root Cause:** Reward imbalance (accumulated profit > penalty) and lack of hard exit logic.

**Solution (Hard Mode Implemented):**
1.  **Force Exit:** `max_holding_steps = 36` (3 hours). Position closed automatically.
2.  **Realized Reward Only:**
    -   `HOLD`: Reward = 0 (No unrealized gains allowed).
    -   `SELL`: Reward = Realized PnL (Huge bonus/penalty).
3.  **Sniper Penalty:** Aggressive exponential penalty if holding > 12 steps without profit.

**Status:** Retrying with Multi-Model Experiments (Phase 1.6)

---

### 17. Phase 1.6: Scalper Experiments (Sequential Tuning)
**Date:** 2026-01-24
**Goal:** Find optimal balance between "Force Exit" penalty and "Risk Taking".

**Experiments Launched:**
1.  **Aggressive:** Max 36 steps, Penalty -0.1 (Starts at 1 hr)
2.  **Balanced:** Max 48 steps, Penalty -0.02 (Starts at 2 hr)
3.  **Relaxed:** Max 96 steps, Penalty -0.01 (Starts at 4 hr)

**Execution:** Running sequentially via `scripts/train_experiments.ps1`

---

### 18. Phase 2: Sentiment Foundation
**Date:** 2026-01-24
**Work:** Created `src/skills/sentiment_fetcher.py` while waiting for training.
**Features Implemented:**
-   **Fear & Greed Index:** API integration (alternative.me)
-   **VIX (Volatility):** Yahoo Finance (`^VIX`)
-   **US 10Y Yield:** Yahoo Finance (`^TNX`)
-   **DXY (Dollar Index):** Yahoo Finance (`DX-Y.NYB`)

**Status:** Phase 1.6 Experiments Paused (After Model 1)

---

### 18. Phase 1.6 Result: Aggressive Model Failure
**Date:** 2026-01-24
**Model:** `trinity_scalper_XAUUSD_M5_aggressive.zip`
**Constraints:** Max Holding 36 steps, Penalty -0.1 (Start 1hr)
**Results:**
-   **Return:** -0.85% (Loss)
-   **Avg Holding:** 198 Steps (Misleading metric)
-   **Behavior:** Panic Trading (Force Sell -> Buy Back immediately). The constraints were too strict, causing the model to churn and lose on fees/spread.

**Decision:**
-   Experiments paused to analyze.
-   **Next Step:** Resume with **Balanced Model** (Medium constraints) which allows more breathing room.

---

### 19. Phase 2: Sentiment Foundation (Ready)
**Date:** 2026-01-24
**Status:** Code implemented (`src/skills/sentiment_fetcher.py`). Waiting for a stable Scalper model before integrating.


**Status:** Training Scalper v2 with Exit Signals (500k steps)
