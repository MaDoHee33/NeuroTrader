import sys
import os
import psutil
import asyncio
from pathlib import Path
from datetime import datetime

# Add root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.discord_bot import DiscordBot
from src.main import load_config

async def check_health():
    config = load_config()
    bot = DiscordBot(config)
    
    # 1. Check Process (NeuroTrader)
    running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info['cmdline']
            if cmd and 'python' in cmd[0] and 'src/main.py' in str(cmd):
                running = True
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
            
    if not running:
        print("❌ Core Process DOWN!")
        await bot.send("🚨 **CRITICAL**: NeuroTrader Core process is DOWN! Restarting Service recommended.")
        sys.exit(1)
    
    # 2. Check Data Freshness (Optional - Check last modified parquet)
    # TODO: Scan data/nautilus_catalog for recent updates
    
    print("✅ System Healthy.")

if __name__ == "__main__":
    asyncio.run(check_health())
