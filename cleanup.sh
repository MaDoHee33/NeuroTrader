#!/bin/bash
set -e

echo "🧹 Starting Cleanup..."

# 1. Create Archive Structure
mkdir -p archive/{skills,workspace,logs,docs,notebooks}
mkdir -p models logs

# 2. Archive Legacy Skills
if [ -d "skills" ]; then
    echo "📦 Archiving skills..."
    mv skills archive/
fi

# 3. Consolidate Models (from workspace/models to root models/)
if [ -d "workspace/models" ]; then
    echo "📦 Consolidating models..."
    rsync -av workspace/models/ models/
fi

# 4. Consolidate Logs
if [ -d "data/logs" ]; then
    echo "📦 Consolidating logs (from data)..."
    rsync -av data/logs/ logs/
    rm -rf data/logs
fi
if [ -d "workspace/logs" ]; then
    echo "📦 Consolidating logs (from workspace)..."
    rsync -av workspace/logs/ logs/
fi

# Move root log files
mv *.log archive/logs/ 2>/dev/null || true

# 5. Archive Workspace (after moving useful stuff) & Legacy Files
echo "📦 Archiving workspace & legacy files..."
mv MANUAL_TH.md archive/docs/ 2>/dev/null || true
mv notebooks/NeuroTrader_Colab_Fast.ipynb archive/notebooks/ 2>/dev/null || true

if [ -d "workspace" ]; then
    mv workspace archive/
fi

# 6. Verify Structure
echo "✅ Cleanup structure created."
ls -F
