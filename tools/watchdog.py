import subprocess
import time
import sys
import os
from pathlib import Path

# Constants
RESTART_DELAY = 10  # Seconds
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_agent():
    """Starts the main agent process."""
    main_script = PROJECT_ROOT / "src" / "main.py"
    if not main_script.exists():
        print(f"❌ Critical Error: Cannot find {main_script}")
        print(f"Please run this from the project root or ensure src/main.py exists.")
        sys.exit(1)
        
    print(f"🚀 Launching NeuroTrader from {main_script}...")
    # Use the same python executable that called this script
    return subprocess.Popen([sys.executable, str(main_script)])

if __name__ == "__main__":
    print("🐶 Watchdog Active. Monitoring NeuroTrader...")
    print(f"📂 Project Root: {PROJECT_ROOT}")
    
    process = run_agent()
    
    try:
        while True:
            return_code = process.poll()
            
            if return_code is not None:
                print(f"⚠️ Agent Crashed (Code {return_code}). Restarting in {RESTART_DELAY}s...")
                time.sleep(RESTART_DELAY)
                process = run_agent()
            
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n🛑 Watchdog stopped by user. Killing agent...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("✅ Shutdown complete.")
         
