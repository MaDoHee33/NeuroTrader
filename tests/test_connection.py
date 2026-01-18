
import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.body.mt5_driver import MT5Driver

async def test_connection():
    print("🔄 Testing MT5 Connection...")
    
    driver = MT5Driver()
    
    # Force off Mock mode for this test to verify real connection
    driver.is_mock = False 
    
    success = await driver.initialize()
    
    if success:
        print("✅ Connection Successful!")
        # Try to get some info
        # We need to access the mt5 object directly or add a method in driver
        # Since we are importing conditional mt5 in driver, let's just trust initialize() for now
        # OR better: add a method to get terminal info in driver or use the hidden import
        
        # Let's try to get latest data
        data = await driver.get_latest_data("EURUSD")
        if data:
             print(f"📊 Latest EURUSD Data: {data}")
        else:
             print("⚠️  Connected but failed to get data (Market closed or Symbol invalid?)")
             
        driver.shutdown()
    else:
        print("❌ Connection Failed.")
        print("Possible fixes:")
        print("1. Ensure MT5 Terminal is OPEN.")
        print("2. On Linux: Ensure you are running with 'mt5linux' and MT5 is running in Wine.")
        print("3. Check 'Algo Trading' is enabled in MT5 options.")

if __name__ == "__main__":
    asyncio.run(test_connection())
