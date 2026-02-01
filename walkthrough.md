# 📝 Walkthrough: Scalper V2.6 Implementation

> **วันที่:** 2026-01-27
> **สถานะ:** Quick Test (50k) สำเร็จ ✅

---

## สิ่งที่ทำ

### 1. แก้ไข Hyperparameters
**File:** `config/training_config.yaml`

| Parameter | Before | After | เหตุผล |
|-----------|--------|-------|--------|
| `n_steps` | 256 | **64** | ลด horizon จาก 21h → 5h |
| `ent_coef` | 0.01 | **0.02** | เพิ่มการ explore |

render_diffs(file:///C:/Users/pp/.gemini/antigravity/scratch/NeuroTrader/config/training_config.yaml)

---

### 2. ปรับ Reward Function
**File:** `src/brain/env/trading_env.py`

**การเปลี่ยนแปลง:**
- Time Decay เริ่มเร็วขึ้น: **6 bars** (30 นาที) → ก่อน 12 bars
- Decay อ่อนลง: **0.02** → ก่อน 0.05
- เพิ่ม **Speed Bonus** สำหรับปิดเร็ว + กำไร (max +0.1)

render_diffs(file:///C:/Users/pp/.gemini/antigravity/scratch/NeuroTrader/src/brain/env/trading_env.py)

---

### 3. แก้ไขปัญหา Data Loading
**File:** `scripts/train_trinity.py`

**ปัญหา:** MT5 ใช้ `tick_volume` แทน `volume`
**แก้ไข:** เพิ่ม rename อัตโนมัติใน `load_data()`

render_diffs(file:///C:/Users/pp/.gemini/antigravity/scratch/NeuroTrader/scripts/train_trinity.py)

---

## ผลการทดสอบ

### Quick Training (50k steps)
```
✅ Training Complete!
📁 Model: models/trinity_scalper_XAUUSDm_M5.zip
⏱️ Duration: ~6 minutes
```

---

## ขั้นตอนถัดไป

1. **รัน Full Training (1M steps)**
   ```powershell
   cd C:\Users\pp\.gemini\antigravity\scratch\NeuroTrader
   python scripts/train_trinity.py --role scalper --data data/raw/XAUUSDm_M5_raw.parquet --steps 1000000
   ```

2. **Evaluate หลังเทรนเสร็จ**
   ```powershell
   python scripts/autopilot.py evaluate --model models/trinity_scalper_XAUUSDm_M5.zip --data data/raw/XAUUSDm_M5_raw.parquet --role scalper
   ```

3. **เปรียบเทียบกับ V2.3 Baseline:**
   - Target: Avg Holding < 48 steps (< 4h)
   - V2.3: 194 steps (~16h), +4.15% return
