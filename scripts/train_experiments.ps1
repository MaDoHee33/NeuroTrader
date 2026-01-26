# NeuroTrader Multi-Model Training Experiment
# Trains 3 variations of Scalper to find optimal constraints

$PYTHON = "python"
$SCRIPT = "scripts/train_trinity.py"
$DATA = "data/processed/XAUUSD_M5_processed.parquet"
$STEPS = 500000

Write-Host "🚀 STARTING SCALPER TUNING EXPERIMENTS" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. AGGRESSIVE (Hard Constraints - Like v002 but refined)
Write-Host "`n[1/3] Training AGGRESSIVE Model (Hard Limits)..." -ForegroundColor Yellow
& $PYTHON $SCRIPT --role scalper --data $DATA --steps $STEPS --suffix "aggressive" `
    --max_steps_holding 36 `
    --sniper_start 12 `
    --sniper_amt 0.1 `
    --force_exit_penalty 1.0

if ($LASTEXITCODE -ne 0) { Write-Host "❌ Aggressive Training Failed!" -ForegroundColor Red; exit }

# 2. BALANCED (Medium Constraints)
Write-Host "`n[2/3] Training BALANCED Model (Medium Limits)..." -ForegroundColor Yellow
& $PYTHON $SCRIPT --role scalper --data $DATA --steps $STEPS --suffix "balanced" `
    --max_steps_holding 48 `
    --sniper_start 20 `
    --sniper_amt 0.02 `
    --force_exit_penalty 0.5

if ($LASTEXITCODE -ne 0) { Write-Host "❌ Balanced Training Failed!" -ForegroundColor Red; exit }

# 3. RELAXED (Soft Constraints - Encouraging Entry)
Write-Host "`n[3/3] Training RELAXED Model (Soft Limits)..." -ForegroundColor Yellow
& $PYTHON $SCRIPT --role scalper --data $DATA --steps $STEPS --suffix "relaxed" `
    --max_steps_holding 96 `
    --sniper_start 48 `
    --sniper_amt 0.01 `
    --force_exit_penalty 0.1

if ($LASTEXITCODE -ne 0) { Write-Host "❌ Relaxed Training Failed!" -ForegroundColor Red; exit }

Write-Host "`n✅ ALL EXPERIMENTS COMPLETED!" -ForegroundColor Green
Write-Host "Check reports in reports/ directory to compare results."
