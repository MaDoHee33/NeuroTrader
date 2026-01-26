@echo off
:: NeuroTrader Weekly Maintenance

cd /d "%~dp0.."
if exist ".venv" call .venv\Scripts\activate

set "LOG_FILE=logs\maintenance_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log"

echo 🔧 Starting Maintenance %DATE% %TIME% >> "%LOG_FILE%"
echo 🔧 Starting Maintenance %DATE% %TIME%

:: 1. Deep Data Clean/Update
echo 📊 Extending History... >> "%LOG_FILE%"
echo 📊 Extending History...
python scripts\update_data.py >> "%LOG_FILE%" 2>&1

:: 2. Retrain Model
echo 🧠 Retraining Model... >> "%LOG_FILE%"
echo 🧠 Retraining Model...
if exist "src\brain\train.py" (
    python src\brain\train.py --mode=retrain >> "%LOG_FILE%" 2>&1
) else (
    echo ⚠️ Trainer not found! >> "%LOG_FILE%"
)

:: 3. Cleanup Logs (older than 30 days - approximate)
echo 🧹 Cleaning Logs... >> "%LOG_FILE%"
echo 🧹 Cleaning Logs...
forfiles /p "logs" /s /m *.log /d -30 /c "cmd /c del @path" 2>nul

echo ✅ Maintenance Complete. >> "%LOG_FILE%"
echo ✅ Maintenance Complete.
