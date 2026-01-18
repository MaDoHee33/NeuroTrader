# NeuroNautilus Training Guide

## 🎯 Overview
NeuroNautilus supports training on both **Local** and **Google Colab** environments with automatic detection and path configuration.

---

## 🏠 Local Training

### Quick Start
```bash
# Basic training (1M steps)
python -m src.brain.train

# Custom configuration
python -m src.brain.train \
  --timesteps 10000000 \
  --model-name ppo_production \
  --learning-rate 0.0003
```

### Parameters
- `--timesteps`: Total training steps (default: 1,000,000)
- `--bar-type`: Data bar type (default: XAUUSD.SIM-5-MINUTE-LAST-EXTERNAL)
- `--model-name`: Output model filename (default: ppo_neurotrader)
- `--learning-rate`: PPO learning rate (default: 0.0003)
- `--data-dir`: Override data directory path

### Time Estimates (Local - CPU)
- **100k steps**: ~3-4 minutes
- **1M steps**: ~30-40 minutes
- **10M steps**: ~5-7 hours ⚠️

---

## ☁️ Google Colab Training

### Quick Start
1. Open the notebook: `notebooks/NeuroNautilus_Training.ipynb`
2. Or use this direct link: [Open in Colab](https://colab.research.google.com/github/MaDoHee33/NeuroTrader/blob/neuronautilus-v1/notebooks/NeuroNautilus_Training.ipynb)
3. **Runtime → Change runtime type → GPU (T4)**
4. Run all cells sequentially

### Prerequisites
Upload your Parquet data files to Google Drive:
```
MyDrive/
└── NeuroTrader_Workspace/
    └── data/
        └── nautilus_store/
            └── data/
                └── bar/
                    └── XAUUSD.SIM-5-MINUTE-LAST-EXTERNAL/
                        ├── data.parquet
                        └── metadata.parquet
```

### Time Estimates (Colab - T4 GPU)
- **100k steps**: ~5 minutes
- **1M steps**: ~30-45 minutes
- **10M steps**: ~6-8 hours

### Advanced: Colab Pro
For faster training, upgrade to Colab Pro:
- **GPU**: V100 or A100 (5-10x faster)
- **Time**: 10M steps in 1-2 hours
- **Cost**: ~$10/month

---

## 📊 Model Outputs

### Saved Files
After training completes, you'll find:

**Models:**
- `models/checkpoints/{model_name}.zip` - Final trained model
- `models/checkpoints/ppo_checkpoint_*_steps.zip` - Intermediate checkpoints (every 10k steps)

**Logs:**
- `logs/PPO_{run_id}/` - TensorBoard logs

### View TensorBoard

**Local:**
```bash
tensorboard --logdir logs/
```

**Colab:**
```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/NeuroTrader_Workspace/logs
```

---

## 🚀 Recommended Training Strategy

1. **Test Run (100k steps)**: Verify setup works (~5 min)
2. **Baseline (1M steps)**: Get a functional model (~45 min)
3. **Production (10M steps)**: High-quality model (overnight)

### Strategy by Environment

| Environment | Recommended Steps | Reason |
|------------|-------------------|---------|
| **Local (CPU)** | 100k - 1M | CPU is slow for RL; use for testing |
| **Colab Free (T4)** | 1M - 5M | Free GPU accelerates training 5-10x |
| **Colab Pro (V100/A100)** | 10M+ | Premium GPUs can handle deep training |

---

## 🛠️ Troubleshooting

### Issue: "No bars loaded"
**Fix**: Ensure data exists at `data/nautilus_store/data/bar/{BAR_TYPE}/`

### Issue: Colab disconnects during training
**Fix**: 
1. Use Colab Pro for longer sessions
2. Add keep-alive script:
```javascript
function KeepClicking(){
   console.log("Clicking");
   document.querySelector("colab-connect-button").click()
}
setInterval(KeepClicking,60000)
```

### Issue: "CUDA out of memory"
**Fix**: Reduce batch size in `train.py`:
```python
model = PPO(..., batch_size=32)  # Default: 64
```

---

## 📈 Next Steps

After training:
1. Download the model (`.zip` file)
2. Place it in `models/checkpoints/` on your local machine
3. Run backtest: `python -m src.neuro_nautilus.runner`
4. Compare performance vs. random baseline

Happy Training! 🧠⚡
