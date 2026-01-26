
# Trinity Training Batch Script
Write-Host "Starting Trinity Training Protocol..."
Write-Host "Estimated Time: ~45-60 mins per model"

# 1. Train Scalper (Short-Term)
Write-Host "Step 1/2: Training Scalper (M5)..."
python scripts/train_trinity.py --role scalper --data data/processed/XAUUSD_M5_processed.parquet --steps 500000

# 2. Train Swing (Mid-Term)
Write-Host "Step 2/2: Training Swing Trader (H1)..."
python scripts/train_trinity.py --role swing --data data/processed/XAUUSD_H1_processed.parquet --steps 500000

Write-Host "All Training Complete!"
