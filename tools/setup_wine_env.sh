#!/bin/bash
# setup_wine_env.sh
# Automates the installation of Python 3.10 into the Wine environment

set -e

echo "🍷 Setting up Python Environment in Wine..."
echo "----------------------------------------"

# 1. Check Wine
if ! command -v wine &> /dev/null; then
    echo "❌ Error: 'wine' is not installed. Please install it first (sudo apt install wine64)."
    exit 1
fi

echo "✅ Wine detected."

# 2. Download Python Installer (Windows Version)
PYTHON_VER="3.10.11"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VER}/python-${PYTHON_VER}-amd64.exe"
INSTALLER="python-${PYTHON_VER}-amd64.exe"

if [ ! -f "$INSTALLER" ]; then
    echo "⬇️  Downloading Python ${PYTHON_VER} Installer..."
    wget -q --show-progress -O "$INSTALLER" "$PYTHON_URL"
else
    echo "ℹ️  Using existing installer: $INSTALLER"
fi

# 3. Install Python quietly
echo "📦 Installing Python into Wine (Silent Mode)..."
echo "⏳ This may take 1-2 minutes. Please wait..."

# Attempt installation
# /quiet = No GUI
# InstallAllUsers=1 = Install for all users (System wide in wine)
# PrependPath=1 = Add to PATH so 'wine python' works
wine "$INSTALLER" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

# Wait a bit for wine checking to settle
sleep 2

# 4. Verify
echo "🔍 Verifying Installation..."
if wine python --version &> /dev/null; then
    VERSION=$(wine python --version)
    echo "✅ Success! Installed: $VERSION"
    
    # 5. Clean up
    echo "🧹 Removing installer..."
    rm "$INSTALLER"
    
    echo "----------------------------------------"
    echo "🎉 Setup Complete."
    echo "👉 Next Step: Install dependencies inside Wine:"
    echo "   wine python -m pip install -r requirements.txt"
else
    echo "❌ Verification Failed."
    echo "It seems 'python' is not yet in the Wine PATH."
    echo "Try running: 'wine cmd' then type 'python'."
    echo "If that fails, run '$INSTALLER' manually with 'wine $INSTALLER' to see the GUI errors."
fi
