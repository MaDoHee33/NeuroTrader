#!/bin/bash
# tools/wine_runner.sh
# A wrapper to run Python scripts inside Wine using the full path to python.exe

# 1. Locate Python in Wine
# Common path for Python 3.10 installed via our script
PYTHON_EXE="$HOME/.wine/drive_c/users/$USER/AppData/Local/Programs/Python/Python310/python.exe"

if [ ! -f "$PYTHON_EXE" ]; then
    # Try generic find (slower)
    echo "🔍 Python not found in standard location, searching..."
    PYTHON_EXE=$(find "$HOME/.wine/drive_c" -name "python.exe" -print -quit)
fi

if [ -z "$PYTHON_EXE" ] || [ ! -f "$PYTHON_EXE" ]; then
    echo "❌ Error: Could not find python.exe in Wine directory."
    echo "Please run 'bash tools/setup_wine_env.sh' first."
    exit 1
fi

# echo "ℹ️  Using Wine Python: $PYTHON_EXE"

# 2. Run the command
# We use "$@" to pass all arguments to the python interpreter
wine "$PYTHON_EXE" "$@"
