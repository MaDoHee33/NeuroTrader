# AI Consultation Log

> **Purpose**: Record all strategic consultations with external Large Language Models (DeepSeek via MCP).
> **Format**: Problem -> Prompt/Inquiry -> Key Advice -> Action Taken.

---

## 1. Zero Trades Issue (Scalper Aggressive)
**Date**: 2026-01-27
**Model**: DeepSeek (via Ollama)

### 🔴 Problem
The "Aggressive Scalper" (V2.2) trained for 1M steps resulted in **0 Trades**.
- It learned to "Hold Cash" perfectly to get a Reward of 0.
- Entering a trade immediately incurred a negative reward (Spread/Commission penalty), which the model couldn't overcome with the promise of future profits.

### ❓ Inquiry (Prompt Summary)
"Aggressive rewards failed (0 Trades). The issue seems to be the 'Barrier to Entry' (Spread Cost). How do we fix this?"
- Proposed options: Entry Bonus, Action Masking, Curriculum Learning.

### 💡 AI Advice
1. **Entry Bonus is effective**: A small positive reward (+0.05 to +0.1) just for entering a trade can zero-out the spread fear. It must be carefully tuned to avoid spam.
2. **Curriculum Learning is best**: Ideally, start with 0 cost and ramp it up.
3. **Action Masking**: Good for control but harder to implement quickly.

### ✅ Action Taken
**Implemented Entry Bonus (+0.05)** in `src/brain/env/trading_env.py` (V2.3 Experiment).
- Logic: `if action == BUY: reward += 0.05`
- Logic: `if action == BUY: reward += 0.05`
- Rationale: Counteract the immediate spread penalty so the net reward of entering is roughly 0 (neutral), allowing exploration.

---

## 2. Reducing Holding Time (Scalper V2.4)
**Date**: 2026-01-27
**Model**: DeepSeek (via Ollama)

### 🔴 Problem
Entry Bonus worked (V2.3: 51 trades, +4% return). However, **Avg Holding Time is 16 hours** (194 steps). This is Swing Trading behavior, not Scalping. The model holds profitable positions too long because the trend reward outweighs the weak time penalty.

### ❓ Inquiry
"How to force reducing holding time to < 2 hours without killing profitability?"
- Options: Harsher Time Penalty vs Velocity Reward (`profit / steps`).

### 💡 AI Advice
1. **Time Penalty must be exponential**: Linear decay (-0.01) is too weak against a Trend Reward (+20.0). Use `decay = (steps-12)^1.5`.
2. **Velocity Reward**: Reward "Profit per Minute". `reward += log_return / steps`. This strongly incentives quick profits.
3. **Combination**: Use both to squeeze the model.

### ✅ Action Taken
**Implemented "Velocity Mode" (V2.4)** in `trading_env.py`:
1. **Velocity Bonus**: `if profit > 0: reward += log_return * 50.0 * (1 / steps)`
2. **Exponential Decay**: `decay = 0.005 * (steps - 12) ** 1.5`
- This should aggressively punish holding > 1 hour and hugely reward quick snipes.

