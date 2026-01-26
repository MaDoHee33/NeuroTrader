
import MetaTrader5 as mt5
import time
import sys

def main():
    print("🧪 Starting MT5 Connectivity Test...")
    
    # 1. Initialize
    if not mt5.initialize():
        print(f"❌ Initialize failed: {mt5.last_error()}")
        return

    symbol = "XAUUSDm"
    volume = 0.01
    
    # Check Symbol
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Failed to select {symbol}")
        mt5.shutdown()
        return

    # 2. Place BUY Order    
    print(f"🛒 Sending BUY {volume} {symbol}...")
    price = mt5.symbol_info_tick(symbol).ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 20,
        "magic": 999999,
        "comment": "TEST ORDER",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Buy Failed: {result.retcode} ({result.comment})")
        mt5.shutdown()
        return
        
    order_ticket = result.order
    print(f"✅ BUY Executed! Ticket: {order_ticket} | Price: {result.price}")
    
    # 3. Wait
    print("⏳ Waiting 3 seconds...")
    time.sleep(3)
    
    # 4. Close Order (Sell)
    print("🔄 Closing Position...")
    
    # Find Position
    positions = mt5.positions_get(ticket=order_ticket)
    if not positions:
        print("❌ Position not found (Closed by SL/TP/Stopout?)")
    else:
        # Close
        pos = positions[0]
        price = mt5.symbol_info_tick(symbol).bid
        
        request_close = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": 999999,
            "comment": "TEST CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        res_close = mt5.order_send(request_close)
        if res_close.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Close Failed: {res_close.retcode}")
        else:
            print(f"✅ CLOSE Executed! Price: {res_close.price}")
            
            # Profit?
            profit = (res_close.price - result.price) * volume * 100 # Approx
            print(f"💰 Result Ticket {order_ticket}: Closed.")

    mt5.shutdown()
    print("\n✅ TEST COMPLETE.")

if __name__ == "__main__":
    main()
