#!/bin/bash
# tools/fix_wine_dlls.sh
# Fixes 'ucrtbase.dll' errors by installing MSVC Redistributable and setting DLL overrides.

set -e

echo "🔧 Fixing Wine Runtime Libraries..."
echo "---------------------------------"

# 1. Download MSVC Redist 2015-2022
URL="https://aka.ms/vs/17/release/vc_redist.x64.exe"
FILE="vc_redist.x64.exe"

if [ ! -f "$FILE" ]; then
    echo "⬇️  Downloading VC Redistributable..."
    wget -q --show-progress -O "$FILE" "$URL"
fi

# 2. Install (Silent)
echo "📦 Installing VC Redistributable (Silent)..."
# /install /quiet /norestart
wine "$FILE" /install /quiet /norestart

echo "⏳ Waiting for installation to register..."
sleep 5

# 3. Force Wine to use the Native DLL for ucrtbase
# This is critical. Without this, Wine might still use its incomplete builtin version.
echo "⚙️  Configuring DLL Overrides..."
wine reg add "HKEY_CURRENT_USER\Software\Wine\DllOverrides" /v ucrtbase /t REG_SZ /d "native,builtin" /f > /dev/null

# Clean up
rm "$FILE"

echo "---------------------------------"
echo "✅ Fix Applied."
echo "👉 Please try running the bot again."
