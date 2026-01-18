#!/bin/bash
# NeuroTrader - Data Pipeline Auto-Start Script
# Usage: ./scripts/run_pipeline.sh

# Get directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting NeuroTrader Data Pipeline..."
echo "📂 Project Root: $PROJECT_ROOT"

# Navigate to project root
cd "$PROJECT_ROOT"

# Check for Python Virtual Env
if [ -d ".venv" ]; then
    echo "🐍 Activating .venv..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "🐍 Activating venv..."
    source venv/bin/activate
fi

# Run Pipeline
echo "⏳ Running Pipeline..."
python tools/data_pipeline.py

echo "✅ Done. You can close this window."
sleep 5
