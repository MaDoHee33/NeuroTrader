@echo off
:: NeuroTrader Data Streamer
:: Runs update_data.py periodically

cd /d "%~dp0.."

:: Activate Venv
if exist ".venv" (
    call .venv\Scripts\activate
) else if exist "venv" (
    call venv\Scripts\activate
)

echo 🌊 Starting Data Stream Loop...

:loop
echo [%DATE% %TIME%] Running Data Update...
python scripts\update_data.py

if %ERRORLEVEL% NEQ 0 (
    echo ⚠️ Data Update Failed! Retrying in 60s...
) else (
    echo ✅ Data Update Success. Sleeping...
)

timeout /t 60
goto loop
