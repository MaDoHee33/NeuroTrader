@echo off
echo 🚀 Installing System Dependencies...
echo Requesting Administrative Privileges...

:: Check for permissions
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Success: Administrative permissions confirmed.
) else (
    echo Failure: Current permissions inadequate.
    echo Right-click this script and select "Run as Administrator".
    pause
    exit
)

echo 📦 Upgrading Visual C++ Redistributable...
winget install --id Microsoft.VCRedist.2015+.x64 --source winget --accept-package-agreements --accept-source-agreements --force
winget upgrade --id Microsoft.VCRedist.2015+.x64 --source winget --accept-package-agreements --accept-source-agreements

echo ✅ Installation Process Finished.
echo 🔄 Please restart your terminal/computer and try running the backtest again.
pause
