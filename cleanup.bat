@echo off
setlocal EnableDelayedExpansion

echo 🧹 Starting Cleanup...

:: 1. Create Archive Structure
if not exist "archive\skills" mkdir "archive\skills"
if not exist "archive\workspace" mkdir "archive\workspace"
if not exist "archive\logs" mkdir "archive\logs"
if not exist "archive\docs" mkdir "archive\docs"
if not exist "archive\notebooks" mkdir "archive\notebooks"
if not exist "models" mkdir "models"
if not exist "logs" mkdir "logs"

:: 2. Archive Legacy Skills
if exist "skills" (
    echo 📦 Archiving skills...
    move "skills" "archive\"
)

:: 3. Consolidate Models
if exist "workspace\models" (
    echo 📦 Consolidating models...
    xcopy "workspace\models\*" "models\" /E /I /Y
)

:: 4. Consolidate Logs
if exist "data\logs" (
    echo 📦 Consolidating logs from data...
    xcopy "data\logs\*" "logs\" /E /I /Y
    rd /s /q "data\logs"
)
if exist "workspace\logs" (
    echo 📦 Consolidating logs from workspace...
    xcopy "workspace\logs\*" "logs\" /E /I /Y
)

:: Move root log files
if exist "*.log" move "*.log" "archive\logs\"

:: 5. Archive Workspace & Legacy Files
echo 📦 Archiving workspace ^& legacy files...
if exist "MANUAL_TH.md" move "MANUAL_TH.md" "archive\docs\"
if exist "notebooks\NeuroTrader_Colab_Fast.ipynb" move "notebooks\NeuroTrader_Colab_Fast.ipynb" "archive\notebooks\"

if exist "workspace" (
    move "workspace" "archive\"
)

:: 6. Verify Structure
echo ✅ Cleanup structure created.
dir /w
pause
