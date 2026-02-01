# NeuroTrader - Setup Daily Data Update Task
# Run this script as Administrator to create the scheduled task

$TaskName = "NeuroTrader_DailyDataUpdate"
$ScriptPath = "$PSScriptRoot\daily_update.bat"
$ProjectRoot = (Get-Item $PSScriptRoot).Parent.FullName

Write-Host "========================================"
Write-Host "  NeuroTrader Task Scheduler Setup"
Write-Host "========================================"
Write-Host ""

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "This script requires Administrator privileges!" -ForegroundColor Yellow
    Write-Host "Right-click and 'Run as Administrator'" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Remove existing task if exists
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Removing existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create action
$Action = New-ScheduledTaskAction -Execute $ScriptPath -WorkingDirectory $ProjectRoot

# Create trigger: At logon
$TriggerLogon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Settings
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Create the task
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $TriggerLogon -Settings $Settings -Description "NeuroTrader: Update XAUUSD/BTCUSD data from MT5 daily on login" -RunLevel Highest

Write-Host ""
Write-Host "Task '$TaskName' created successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Task Details:"
Write-Host "   - Trigger: At every logon"
Write-Host "   - Script:  $ScriptPath"
Write-Host "   - Symbols: XAUUSD, BTCUSD"
Write-Host ""
Write-Host "To remove: Unregister-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Yellow
Write-Host "To run now: Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Yellow
Write-Host ""

Read-Host "Press Enter to exit"
