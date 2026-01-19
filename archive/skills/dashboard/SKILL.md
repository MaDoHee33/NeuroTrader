---
name: NeuroTrader DashboardSkill
description: A real-time command center for NeuroTrader. Visualizes market data, AI signals, and portfolio performance using Streamlit.
---

# NeuroTrader Dashboard Skill

This skill provides a web-based "Cockpit" for the trading system. It aggregates data from multiple sources (MT5, Sentiment Analyst, Logs) into a single view.

## Capabilities

### 1. Market Overview
- Displays real-time chart for XAUUSD (Gold).
- Shows key technical indicators (EMA, RSI, MACD).

### 2. Sentiment Intelligence
- Visualizes the "Fear & Greed" score from the Sentiment Analyst skill.
- Displays the latest headlines used for analysis.

### 3. AI Diagnostics
- Shows recent trade logs and decision probabilities.
- (Future) Real-time visualization of the LSTM hidden states.

## Usage
Run the dashboard locally:
```bash
streamlit run skills/dashboard/app.py
```
This will open a browser window at `http://localhost:8501`.

## Dependencies
- `streamlit`
- `plotly`
- `pandas`
- `watchdog` (for auto-updates)
