@echo off
:: NeuroTrader Daemon Entrypoint

cd /d "%~dp0.."

echo [%DATE% %TIME%] Starting NeuroTrader Node...

:: Activate Venv
if exist ".venv" (
    call .venv\Scripts\activate
) else if exist "venv" (
    call venv\Scripts\activate
) else (
    echo ❌ No virtual environment found!
    exit /b 1
)

:: Run Core
python src\main.py
pause
