@echo off
:: NeuroTrader - Data Pipeline Auto-Start Script
:: Usage: .\scripts\run_pipeline.bat

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

echo 🚀 Starting NeuroTrader Data Pipeline...
echo 📂 Project Root: %CD%

:: Check for Python Virtual Env
if exist ".venv" (
    echo 🐍 Activating .venv...
    call .venv\Scripts\activate
) else if exist "venv" (
    echo 🐍 Activating venv...
    call venv\Scripts\activate
) else (
    echo ⚠️ No virtual environment found! Running with system python...
)

:: Run Pipeline
echo ⏳ Running Pipeline...
python scripts\update_data.py

echo ✅ Done. You can close this window.
timeout /t 5
