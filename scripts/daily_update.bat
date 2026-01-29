@echo off
:: NeuroTrader Daily Data Updater
:: Runs update_data.py once per execution (for Task Scheduler)
:: Auto-launches MT5 if not running

cd /d "%~dp0.."

:: MT5 Path
set MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe

echo ========================================
echo   NeuroTrader Daily Data Update
echo   %DATE% %TIME%
echo ========================================

:: Check if MT5 is already running
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] MT5 is already running.
) else (
    echo [*] Starting MetaTrader 5...
    start "" "%MT5_PATH%"
    echo [*] Waiting 30 seconds for MT5 to initialize...
    timeout /t 30 /nobreak >NUL
)

:: Activate Venv
if exist ".venv" (
    call .venv\Scripts\activate
) else if exist "venv" (
    call venv\Scripts\activate
)

echo [*] Running data update...
python scripts\update_data.py

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Data Update Failed!
    exit /b 1
) else (
    echo ✅ Data Update Complete!
    exit /b 0
)
