---
name: NeuroTrader Skill
description: A complete AI trading skill for Gold (XAUUSD) and Crypto. Includes data pipeline, backtesting, and live execution using PPO/LSTM agents.
---

# NeuroTrader Skill

This skill allows the agent to manage the entire lifecycle of an algorithmic trading strategy, from fetching data to executing live trades on MetaTrader 5 (MT5).

## Capabilities

### 1. Update Data (ETL Pipeline)
Fetches the latest candles from MT5, merges them with the existing dataset, and re-calculates technical indicators (Level 2 Features).
- **Script**: `scripts/update_data.py`
- **Usage**: `python skills/neuro_trader/scripts/update_data.py --level 2`

### 2. Backtest Strategy
Evaluates the performance of a trained model against historical data.
- **Script**: `scripts/backtest.py`
- **Usage**: `!MPLBACKEND=Agg python skills/neuro_trader/scripts/backtest.py --level 2`
- **Note**: Use `MPLBACKEND=Agg` when running in headless environments (e.g., Colab, VPS).

### 3. Live Trading (Inference)
Connects the trained model to a live MT5 terminal to execute trades in real-time.
- **Script**: `scripts/trade.py`
- **Usage**: `python skills/neuro_trader/scripts/trade.py --model path/to/model.zip`
- **Safety**: Defaults to 0.01 lots for safety.

## Configuration
- **Models**: Place trained `.zip` models in `skills/neuro_trader/assets/models/`.
- **Data**: Processed Parquet files are typically stored in `data/processed/`.

## Dependencies
- `metatrader5` (Windows) or `mt5linux` (Linux)
- `stable-baselines3`, `sb3-contrib`
- `pandas`, `numpy`, `ta`
