import logging
import asyncio
import yaml
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.thermal import ThermalGovernor
from src.utils.discord_bot import DiscordBot
from src.brain.coordinator import BrainCoordinator
from src.body.mt5_driver import MT5Driver
from src.body.stealth import StealthLayer
from src.memory.storage import StorageEngine


CONFIG_PATH = Path("config/main_config.yaml")
SECRETS_PATH = Path("config/secrets.yaml")

def load_config():
    config = {}
    
    # Load Main Config
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f) or {}

    # Load and Merge Secrets
    if SECRETS_PATH.exists():
        with open(SECRETS_PATH, 'r') as f:
            secrets = yaml.safe_load(f) or {}
            config['secrets'] = secrets
            
    return config

async def main_loop():
    # 1. Initialize Subsystems
    logger = get_logger("Main")
    config = load_config()
    
    # Initialize Storage First
    storage = StorageEngine()
    
    thermal = ThermalGovernor(max_temp=config.get('hardware', {}).get('max_cpu_temp', 80))
    stealth = StealthLayer(config)
    
    # Switch to Discord
    bot = DiscordBot(config)
    await bot.start() # Start background loop
    
    brain = BrainCoordinator(config)
    # Pass storage to execution driver for Shadow Recording
    execution = MT5Driver(config, storage=storage)
    
    logger.info("System Starting...")
    await bot.send("🚀 NeuroTrader Ultimate Started. Shadow Mode & DB Active.")

    # Initialize Execution Connection (MT5/Mock)
    if not await execution.initialize():
        logger.critical("Failed to initialize Execution Driver. Exiting.")
        return

    # Save Runtime Status for Dashboard
    import json
    status_path = "data/status.json"
    status_data = {
        "connection": "MOCK" if execution.is_mock else "REAL",
        "shadow_mode": execution.shadow_mode,
        "start_time": str(datetime.now()),
        "pid": os.getpid()
    }
    with open(status_path, "w") as f:
        json.dump(status_data, f)
    
    logger.info(f"System Status: {status_data['connection']} | Shadow: {status_data['shadow_mode']}")

    try:
        while True:
            # 2. Hardware Safety Check
            if not thermal.is_safe():
                logger.warning("Overheat! Pausing Brain...")
                await bot.send("🔥 System Overheating! Pausing.")
                await asyncio.sleep(60)
                continue

            # 3. Async Logic
            market_data = await execution.get_latest_data()
            
            if market_data:
                decision = await brain.process(market_data)
                
                if decision['action'] != 'HOLD':
                    await execution.execute_trade(decision)
            
            # 4. Stealth / Heartbeat
            await stealth.sleep_random(base_seconds=1.0)

    except asyncio.CancelledError:
        logger.info("Main loop cancelled.")
    except Exception as e:
        logger.error(f"Critical Loop Error: {e}", exc_info=True)
        await bot.send(f"⚠️ Error: {e}")
    finally:
        execution.shutdown()
        logger.info("System Shutdown.")

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        pass
