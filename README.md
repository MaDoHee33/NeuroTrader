# 🧠 NeuroTrader V3 (The Trinity System)
**Advanced Agentic Trading System with Behavioral Analysis & Autonomous Skills**

NeuroTrader V3 evolves beyond a single model into a **Multi-Agent Trinity System**, capable of adapting to different market phases (Scalping, Swinging, Trending). It is augmented with **Autonomous Skills** (News Watching, Reporting) and **Hyperparameter Tuning**.

---

## 🏗️ Architecture: The Trinity System
We deploy 3 specialized agents, each with unique reward functions and data horizons:

| Agent Role | Timeframe | Strategy | Reward Logic |
| :--- | :--- | :--- | :--- |
| **⚔️ Scalper** | M5 | Hit & Run | High PnL + **Time Penalty** (Force Short Holding) |
| **🛡️ Swing** | H1 | Trend Waves | Hybrid (PnL + Trend Following) |
| **👑 Trend** | D1 | Wealth Gen | Buy & Hold (Sharpe Ratio + Drawdown Penalty) |

---

## ⚡ Key Features (V3)
### 1. Autonomous Skills (`src/skills/`)
-   **📰 News Watcher**: Automatically connects to Economic Calendars to detect high-impact events (e.g., FOMC, Non-Farm). **Blocks trades** 30 mins before critical news.
-   **📝 Auto Reporter**: Generates professional **Markdown/PDF Reports** after every backtest, analyzing Win Rate, Holding Time, and Market Exposure.

### 2. Hyperparameter Tuning (`scripts/tune_trinity.py`)
-   Powered by **Optuna**.
-   Optimizes `Gamma`, `Learning Rate`, and `Batch Size` to find the perfect balance between Profit and Behavior.
-   Includes **Behavioral Penalties** in the objective function (e.g., punishing a Scalper for holding > 1 hour).

### 3. Developer Experience (`.cursor/mcp.json`)
-   Integrated **Context7 MCP Server**: Allows the AI Developer to fetch up-to-date documentation for libraries (Pandas, TA-Lib) in real-time.

---

## 🚀 Quick Start

### 1. Train the Trinity (Batch)
Train all models sequentially:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_trinity_full.ps1
```
*Individual training:* `python scripts/train_trinity.py --role scalper --data data/processed/XAUUSD_M5_processed.parquet`

### 2. Hyperparameter Tuning (Optimize Behavior)
Fix "Buy & Hold" behavior for Scalpers:
```bash
python scripts/tune_trinity.py --role scalper --data data/processed/XAUUSD_M5_processed.parquet --trials 50
```

### 3. Verification & Reporting
Run backtests with auto-reporting:
```bash
python scripts/backtest_trinity.py --role scalper --data data/processed/XAUUSD_M5_processed.parquet --model models/trinity_scalper_best.zip
```
*Reports saved to `reports/`*

---

## 📁 Project Structure
-   `src/brain/`: Core RL Agents (PPO/LSTM) & Reward Functions
-   `src/skills/`: Autonomous Capabilities (News, Report)
-   `scripts/`: Automation Scripts (Train, Backtest, Tune)
-   `data/`: Processed Parquet Data

---
**Status:** ✅ V3 Upgrade Complete (Jan 2026) | **OS:** Windows Native
