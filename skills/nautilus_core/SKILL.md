---
name: Nautilus Core Skill
description: The high-performance execution engine for NeuroTrader v3. Handles data storage (Parquet), custom data types (Sentiment), and trading execution.
---

# Nautilus Core Skill 🐚

This skill acts as the "Heart" of the system, replacing the old `live_trader.py` loop. It utilizes `nautilus_trader` (Rust) for micro-second precision.

## Components
- **Data Catalog**: Manages historical data in Parquet format.
- **SentimentData**: Custom data type for News/Sentiment scores.
- **Conversion Scripts**: Tools to migrate legacy data to Nautilus format.

## Usage
### 1. Convert Data
Migrate your augmented data to the new catalog:
```bash
python skills/nautilus_core/scripts/convert_data.py
```

### 2. Run Backtest
(Coming in Phase 2)
```bash
python skills/nautilus_core/scripts/backtest_nautilus.py
```
