# 🧠 แผนพัฒนาแบบ Hybrid: PPO + Self-Evolving AI

> **วัตถุประสงค์:** พัฒนา NeuroTrader ให้ใช้งานได้จริง (Production-ready) โดยใช้ทรัพยากรน้อยที่สุด  
> **แนวทาง:** ใช้ PPO + LSTM เดิมเป็น baseline + พัฒนา Self-Evolving AI ควบคู่  
> **สร้างโดย:** DeepSeek Consultation via MCP | วันที่ 2026-01-28

---

## 📋 สารบัญ

1. [ภาพรวมแผนงาน](#-ภาพรวมแผนงาน)
2. [Roadmap แบบละเอียด](#-roadmap-แบบละเอียด-6-เฟส)
3. [สิ่งที่ใช้ต่อได้ vs ต้องสร้างใหม่](#-สิ่งที่ใช้ต่อได้-vs-ต้องสร้างใหม่)
4. [เริ่มต้นเฟส 1: Step-by-Step](#-เริ่มต้นเฟส-1-step-by-step)
5. [Resource Requirements](#-resource-requirements)
6. [Risk Mitigation](#-risk-mitigation)
7. [Success Metrics](#-success-metrics)
8. [Budget Breakdown](#-budget-breakdown)

---

## 🎯 ภาพรวมแผนงาน

### หลักการ 3 ข้อ:
1. **ใช้งานได้จริง** - ทุกเฟสต้องมี Production-ready output
2. **ประหยัดทรัพยากร** - Hardware & Budget น้อยที่สุด
3. **Quick Wins** - เห็นผลลัพธ์เร็ว ไม่ต้องรอนาน

### แนวทาง Hybrid:
```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Architecture                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   PPO + LSTM    │    │     Self-Evolving AI            │ │
│  │   (Baseline)    │───►│  + Curiosity Module             │ │
│  │                 │    │  + Experience Accumulation      │ │
│  │  ✅ ทำกำไรได้    │    │  + Adaptive Learning            │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
│           │                           │                      │
│           └───────────┬───────────────┘                      │
│                       ▼                                      │
│              ┌─────────────────┐                             │
│              │  Hybrid Trading │                             │
│              │     System      │                             │
│              └─────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗓️ Roadmap แบบละเอียด (6 เฟส)

| เฟส | ชื่อ | ระยะเวลา | เป้าหมาย | Milestone |
|:---:|------|:--------:|----------|-----------|
| **1** | Baseline Stabilization | 2 สัปดาห์ | ปรับ PPO ให้ scalping จริง | Scalper ถือ < 1 ชม. |
| **2** | Data & Env Optimization | 2 สัปดาห์ | ลด resource usage | ประมวลผลเร็วขึ้น 50% |
| **3** | Infant Self-Evolving | 1 เดือน | สร้าง Curiosity Agent | Agent เรียนรู้ได้ |
| **4** | Hybrid Deployment | 1 เดือน | รวม PPO + Self-Evolving | ระบบ 2 Agent ทำงานคู่ |
| **5** | Adolescent Learning | 2 เดือน | Adaptive + Regime Detection | ปรับกลยุทธ์อัตโนมัติ |
| **6** | Mature Intelligence | 2+ เดือน | Full Autonomous | 24/7 self-improving |

**รวมระยะเวลา:** ~6-8 เดือน (เร็วกว่า Blueprint เดิม 40%)

---

## 🔍 สิ่งที่ใช้ต่อได้ vs ต้องสร้างใหม่

### ✅ ใช้ได้เลย (ไม่ต้องแก้)

| โมดูล | ไฟล์ | หมายเหตุ |
|-------|------|----------|
| Trading Environment | `src/brain/env/trading_env.py` | รองรับ multi-agent แล้ว |
| Risk Manager | `src/brain/risk_manager.py` | มี Circuit Breaker พร้อม |
| Model Registry | `src/skills/model_registry.py` | รองรับ versioning |
| Feature Registry | `src/brain/features.py` | Unified feature engine |

### ⚠️ ต้องแก้ไขบางส่วน

| โมดูล | ไฟล์ | สิ่งที่ต้องแก้ |
|-------|------|--------------|
| Training Script | `scripts/train_trinity.py` | เพิ่ม curiosity reward, meta-learning |
| Config | `config/training_config.yaml` | เพิ่ม params สำหรับ curiosity, regime |
| Features | `src/brain/features.py` | เพิ่ม regime labels, volatility clusters |
| Market Regime | - | ใช้ HMM หรือ LSTM classifier |

### ❌ ต้องสร้างใหม่ทั้งหมด

| โมดูล | เทคโนโลยี | ลำดับความสำคัญ |
|-------|-----------|:-------------:|
| **Curiosity Module** | PyTorch, ICM (Intrinsic Curiosity Module) | 🔴 สูง |
| **Experience Buffer** | Redis/MongoDB, Custom ReplayBuffer | 🔴 สูง |
| **Progressive Difficulty** | Gymnasium Wrapper, Curriculum Learning | 🟡 กลาง |
| **Market Regime Detection** | HMM (hmmlearn), LSTM, Scikit-learn | 🟡 กลาง |
| **Meta-Learning** | Learn2Learn, MAML, Reptile | 🟢 ต่ำ (ทำทีหลัง) |

---

## 🚀 เริ่มต้นเฟส 1: Step-by-Step

### เป้าหมาย: ทำให้ Scalper ถือ Position < 1 ชั่วโมง

### Step 1: ปรับ Hyperparameters
**ไฟล์:** `config/training_config.yaml`
```yaml
roles:
  scalper:
    hyperparams:
      n_steps: 64          # ลดจาก 256 → 64 (มองสั้นลง)
      gamma: 0.85          # ลดจาก 0.95 (ให้น้ำหนัก short-term)
      ent_coef: 0.03       # เพิ่มเล็กน้อย (exploration)
```

### Step 2: ปรับ Reward Function
**ไฟล์:** `src/brain/env/trading_env.py`
```python
# แก้ไขใน Scalper reward section (ประมาณบรรทัด 286-312)
if self.agent_type == 'scalper':
    # V2.7: Aggressive Time Decay + Speed Bonus
    reward = log_return * 25.0  # เพิ่มจาก 20 → 25
    
    # Entry Bonus (สำคัญมาก - ป้องกัน passivity)
    if trade_info and trade_info.get('action') == 'BUY':
        reward += 0.08  # เพิ่มจาก 0.05 → 0.08
    
    # Steeper Time Decay (เริ่มที่ 4 bars = 20min)
    if self.position > 0 and self.steps_in_position > 4:
        excess = self.steps_in_position - 4
        decay = excess * 0.04  # เพิ่มจาก 0.02 → 0.04
        reward -= decay
    
    # Speed Bonus: ปิดเร็ว = โบนัสสูง
    if trade_info and trade_info.get('action') == 'SELL':
        if log_return > 0 and self.steps_in_position < 12:  # < 1 ชม.
            speed_bonus = 0.15 * (12 - self.steps_in_position) / 12.0
            reward += speed_bonus
```

### Step 3: ลด Force Exit Time
**ไฟล์:** `src/brain/env/trading_env.py`
```python
# แก้ไขบรรทัด 49
self.time_limit = self.reward_config.get('max_holding_steps', 24)  # ลดจาก 36 → 24 (2 ชม.)
```

### Step 4: เทรนใหม่
```powershell
# รัน training ใหม่
cd C:\Users\pp\.gemini\antigravity\scratch\NeuroTrader
python scripts/train_trinity.py --role scalper --data data/processed/XAUUSD_M5_processed.parquet --steps 500000
```

### Step 5: ทดสอบ
```powershell
# Backtest บน Test Set
python scripts/backtest_trinity.py --model models/trinity_scalper_XAUUSD_M5.zip --data data/processed/XAUUSD_M5_processed.parquet
```

### ✅ Checklist เฟส 1
- [ ] ปรับ `n_steps` เป็น 64
- [ ] ปรับ Reward Function (Entry Bonus 0.08, Decay 0.04)
- [ ] ลด Force Exit เป็น 24 steps
- [ ] Train 500k steps
- [ ] Backtest และวัด Avg Holding Time
- [ ] ถ้า Holding < 60 steps (5 ชม.) → ผ่าน!

---

## 💻 Resource Requirements

### ต่ำสุดที่แนะนำ (เฟส 1-3):
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| GPU | ❌ ไม่จำเป็น | Optional |
| Storage | 50 GB | 100 GB |
| ค่าไฟ/เดือน | ~100 บาท | ~200 บาท |

### สำหรับเฟส 4-6:
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 8 cores | 16 cores |
| RAM | 16 GB | 32 GB |
| GPU | ❌ ไม่จำเป็น | RTX 3060+ (ถ้ามี) |
| Storage | 200 GB | 500 GB |
| ค่าไฟ/เดือน | ~300 บาท | ~500 บาท |

---

## ⚠️ Risk Mitigation

| ปัญหาที่อาจเจอ | ทางแก้ไข |
|---------------|---------|
| **Scalper ยังถือนาน** | เพิ่ม penalty, ลด n_steps ต่อไป |
| **Agent ไม่เรียนรู้** | ใช้ curriculum learning, เริ่มจาก env ง่ายๆ |
| **Overfitting** | ใช้ validation set, early stopping |
| **ใช้ RAM เกิน** | ลด batch size, ใช้ streaming data |
| **ระบบล่มระหว่างเทรด** | แยก training กับ production, auto-restart |
| **Budget หมด** | ใช้ open-source tools ทั้งหมด |

---

## 📈 Success Metrics

| เฟส | Metric | เป้าหมาย |
|:---:|--------|:--------:|
| 1 | Avg Holding Time | < 60 steps (< 5 ชม.) |
| 1 | Profit (Test Set) | > 3% |
| 2 | Processing Time | ลด 50% |
| 2 | RAM Usage | < 12 GB |
| 3 | Curiosity Score | > 0.5 |
| 3 | Learning Episodes | < 10 ep ก่อนเรียนรู้ |
| 4 | Dual-Agent Profit | เพิ่ม 5% จาก baseline |
| 5 | Regime Adaptation | Switch strategy ได้เอง |
| 6 | Uptime | 99%+ (24/7) |

---

## 💰 Budget Breakdown

### แบบประหยัด (DIY - ทำเอง)

| เฟส | ระยะเวลา | ค่าใช้จ่าย | หมายเหตุ |
|:---:|:--------:|:---------:|----------|
| 1 | 2 สัปดาห์ | **0 บาท** | ใช้ hardware ที่มี |
| 2 | 2 สัปดาห์ | **0 บาท** | Optimize code |
| 3 | 1 เดือน | **0-500 บาท** | อาจใช้ cloud บ้าง |
| 4 | 1 เดือน | **500-1,000 บาท** | Cloud testing |
| 5 | 2 เดือน | **1,000-2,000 บาท** | เพิ่ม resource |
| 6 | 2+ เดือน | **2,000-5,000 บาท** | 24/7 operation |

**รวม (DIY):** ~5,000-10,000 บาท / 6-8 เดือน

### แบบจ้างทำ (Outsource)

| เฟส | ระยะเวลา | ค่าใช้จ่าย |
|:---:|:--------:|:---------:|
| 1-2 | 1 เดือน | 35,000-45,000 บาท |
| 3-4 | 2 เดือน | 90,000-110,000 บาท |
| 5-6 | 4 เดือน | 200,000-250,000 บาท |

**รวม (Outsource):** ~325,000-400,000 บาท

---

## 🔄 Next Steps

1. **ทันที:** เริ่มเฟส 1 - ปรับ hyperparameters และ reward function
2. **สัปดาห์นี้:** Train Scalper V2.7 และ backtest
3. **เดือนหน้า:** เริ่มสร้าง Curiosity Module
4. **ทุกเดือน:** Review progress และปรับแผน

---

## 📚 Resources & References

- **ICM (Intrinsic Curiosity Module):** [Paper](https://pathak22.github.io/noreward-rl/)
- **Learn2Learn (Meta-Learning):** [GitHub](https://github.com/learnables/learn2learn)
- **HMM for Regime Detection:** [hmmlearn](https://hmmlearn.readthedocs.io/)
- **Curriculum Learning:** [Paper](https://arxiv.org/abs/2003.04960)

---

> 💡 **คำแนะนำสุดท้าย:** เริ่มเล็ก → ค่อยๆ ขยาย → อย่ารีบ → Document ทุกอย่าง!
