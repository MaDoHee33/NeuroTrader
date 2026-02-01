@echo off
echo 🚀 Setting up NeuroTrader Environment...

cd /d "%~dp0"

if not exist ".venv" (
    echo 🐍 Creating virtual environment...
    python -m venv .venv
) else (
    echo 🐍 .venv already exists.
)

echo 📦 Installing dependencies...
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ✅ Setup Complete!
pause
