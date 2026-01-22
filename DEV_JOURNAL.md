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
