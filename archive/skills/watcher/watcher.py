
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import shutil
import sys

# Constants
CACHE_DIR = Path(__file__).parent / "dump"

class Watcher:
    def __init__(self):
        self._ensure_browser_installed()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def _ensure_browser_installed(self):
        """Checks if 'browser' command is available."""
        self.browser_cmd = shutil.which("browser")
        if not self.browser_cmd:
            # Check common install location
            potential_path = Path.home() / ".browser" / "bin" / "browser"
            if potential_path.exists():
                self.browser_cmd = str(potential_path)
            else:
                print("❌ 'browser' CLI not found. Please run 'install.sh' inside skills/watcher/ first.")
                # Fallback to local user bin if unrelated
                potential_local = Path.home() / ".local" / "bin" / "browser"
                if potential_local.exists():
                     self.browser_cmd = str(potential_local)
                else: 
                     sys.exit(1)
        print(f"✅ Watcher using: {self.browser_cmd}")

    def fetch(self, url: str) -> dict:
        """
        Uses browser CLI to extract content.
        Returns a dict with 'url', 'content', 'timestamp'.
        """
        print(f"👀 Watching: {url}")
        try:
            # 1. Open URL
            # Note: 'open' usually starts the browser if not running, or adds a tab.
            subprocess.run([self.browser_cmd, "open", url], check=True, stdout=subprocess.DEVNULL)
            
            # 2. Wait for Load (Naive)
            # Future: Use 'browser wait <body>'
            time.sleep(5) 
            
            # 3. Extract Text
            start_time = time.time()
            result = subprocess.run([self.browser_cmd, "text"], capture_output=True, text=True, timeout=60)
            duration = time.time() - start_time
            
            # 4. Cleanup (Close Tab)
            subprocess.run([self.browser_cmd, "close"], check=False, stdout=subprocess.DEVNULL)
            
            if result.returncode != 0:
                print(f"❌ Error: {result.stderr}")
                return None
            
            data = {
                "timestamp": datetime.now().isoformat(),
                "url": url,
                "duration_seconds": duration,
                "raw_content": result.stdout
            }
            
            # Save dump
            filename = f"dump_{int(time.time())}.json"
            with open(CACHE_DIR / filename, "w") as f:
                json.dump(data, f, indent=2)
                
            print(f"✅ Extracted {len(data['raw_content'])} chars. Saved to {filename}")
            return data

        except Exception as e:
            print(f"💥 Exception: {e}")
            return None

    def monitor(self, interval_sec: int = 900):
        """Infinite loop to watch targets."""
        targets = [
            "https://cryptopanic.com/news/gold/",
            "https://www.investing.com/commodities/gold-news"
        ]
        
        print(f"🔭 Starting Watchtower. Scanning every {interval_sec}s.")
        while True:
            for url in targets:
                self.fetch(url)
            
            print(f"💤 Sleeping for {interval_sec}s...")
            time.sleep(interval_sec)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="Single URL to fetch")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring loop")
    args = parser.parse_args()
    
    watcher = Watcher()
    
    if args.url:
        watcher.fetch(args.url)
    elif args.monitor:
        watcher.monitor()
    else:
        parser.print_help()
