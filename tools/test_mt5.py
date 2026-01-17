import sys
import platform

print("🔍 DIAGNOSTIC: Testing MT5 Connection...")
print(f"   OS: {platform.system()} {platform.release()}")
print(f"   Python: {sys.version}")

try:
    import MetaTrader5 as mt5
    print("✅ Library 'MetaTrader5' imported successfully.")
except ImportError as e:
    print("❌ ERROR: Could not import 'MetaTrader5'.")
    print(f"   Details: {e}")
    print("\n💡 NOTE FOR LINUX USERS:")
    print("   The official 'MetaTrader5' package only works on Windows.")
    print("   On Linux, this script WILL FAIL unless you are running inside Wine with a compatible Python.")
    print("   If you see this error, NeuroTrader will fall back to 'MOCK MODE'.")
    sys.exit(1)

# Initialize
print("⏳ Attempting mt5.initialize()...")
if not mt5.initialize():
    print(f"❌ ERROR: mt5.initialize() failed, error code = {mt5.last_error()}")
    
    # Common Error Codes
    err = mt5.last_error()
    if err == -10000:
        print("   Hint: Is MetaTrader 5 terminal installed and acceptable version?")
    elif err == -10003:
        print("   Hint: Unsupported OS (Are you on Linux/Mac/Wine?)")
else:
    print("✅ MT5 Initialized Successfully!")
    
    # Account Info
    account_info = mt5.account_info()
    if account_info:
        print(f"   Login: {account_info.login}")
        print(f"   Server: {account_info.server}")
        print(f"   Balance: {account_info.balance} {account_info.currency}")
    else:
        print("❌ ERROR: Connected, but failed to get account info.")

    mt5.shutdown()
    print("✅ Connection closed.")
