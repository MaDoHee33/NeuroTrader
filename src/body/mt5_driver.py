import logging
import asyncio
import platform
import random
from datetime import datetime, timedelta
import pandas as pd

# Conditional Import
try:
    if platform.system() == "Windows":
        import MetaTrader5 as mt5
        MT5_AVAILABLE = True
    else:
        # Linux / MacOS
        # Requires: pip install mt5linux
        from mt5linux import MetaTrader5 as mt5
        MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    # If mt5linux fails, we might still want to log it specifically
    if platform.system() == "Linux":
        print("hint: pip install mt5linux")

class MT5Driver:
    def __init__(self, config=None, storage=None):
        self.config = config or {}
        self.storage = storage # Database Connection
        self.logger = logging.getLogger("MT5Driver")
        self.os_type = platform.system()
        
        # Determine Mode
        self.shadow_mode = self.config.get('system', {}).get('shadow_mode', False)
        
        # Try to initialize connection
        # We do NOT strictly block Linux anymore, as user might be using Wine/mt5linux
        if not MT5_AVAILABLE:
             self.logger.warning("MetaTrader5 package not found. Enforcing MOCK MODE.")
             self.is_mock = True
        else:
             self.is_mock = False # Will be confirmed in initialize()
        
        self.connected = False

    async def initialize(self):
        """Connects to MT5 or initializes Mock state."""
        if self.is_mock:
            self.logger.info("Initializing MOCK MT5 connection...")
            await asyncio.sleep(0.5) 
            self.connected = True
            return True
        
        # Real Initialization
        if not MT5_AVAILABLE:
             self.logger.error("MT5 Library not available. Cannot connect.")
             return False

        self.logger.info("Initializing Real MT5 connection...")
        if not mt5.initialize():
             self.logger.error(f"MT5 Initialize failed, error code: {mt5.last_error()}")
             # If real connection fails, fallback to Mock if configured, else fail
             return False
        
        self.connected = True
        self.logger.info("MT5 Connected Successfully.")
        return True

    async def fetch_history(self, symbol="EURUSD", timeframe="D1", count=1000):
        """
        Fetches historical candle data.
        timeframe: "M1", "H1", "D1", etc.
        """
        if self.is_mock:
            self.logger.info(f"Generating {count} mock candles for {symbol}...")
            # Mock Data Generation
            dates = [datetime.now() - timedelta(minutes=i) for i in range(count)]
            dates.reverse()
            
            data = []
            price = 1.1000
            for date in dates:
                change = random.uniform(-0.001, 0.001)
                price += change
                data.append({
                    "time": date,
                    "open": price,
                    "high": price + 0.0005,
                    "low": price - 0.0005,
                    "close": price + 0.0002,
                    "tick_volume": int(random.uniform(100, 1000))
                })
            return pd.DataFrame(data)

        # Real MT5 Data Fetch
        if not self.connected:
            self.logger.error("MT5 not connected.")
            return None

        # Map string timeframe to MT5 constant (Simplified mapping)
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "H1": mt5.TIMEFRAME_H1,
            "D1": mt5.TIMEFRAME_D1
        }
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_D1)
        
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        if rates is None:
            self.logger.error(f"Failed to fetch history: {mt5.last_error()}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    async def fetch_history_range(self, symbol="EURUSD", timeframe="D1", date_from=None, date_to=None):
        """
        Fetches historical candle data within a date range.
        date_from/date_to: datetime objects
        """
        if date_to is None:
            date_to = datetime.now()
        if date_from is None:
            date_from = datetime(2010, 1, 1) # Default far back

        if self.is_mock:
            # Mock Data Logic reuse
            # Estimate count based on timeframe (approx) to reuse logic or just generate simplistic mock
            return await self.fetch_history(symbol, timeframe, count=1000)

        if not self.connected:
            self.logger.error("MT5 not connected.")
            return None
        
        # Map string timeframe
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_D1)
        
        self.logger.info(f"Fetching {symbol} {timeframe} from {date_from} to {date_to}...")
        
        rates = mt5.copy_rates_range(symbol, mt5_tf, date_from, date_to)
        
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            self.logger.warning(f"No data for {symbol} {timeframe}. Error: {err}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    async def get_latest_data(self, symbol="EURUSD"):
        """Fetches latest tick/candle data."""
        if not self.connected:
            self.logger.error("Not connected to MT5/Mock.")
            return None

        if self.is_mock:
            # Generate random walk data for testing
            mock_price = 1.1000 + random.uniform(-0.0005, 0.0005)
            data = {
                "symbol": symbol,
                "bid": mock_price,
                "ask": mock_price + 0.0001,
                "time": datetime.now().timestamp()
            }
            return data

        # Real MT5 Data Fetch
        # tick = mt5.symbol_info_tick(symbol)
        # if tick:
        #     return tick._asdict()
        return None 

    async def execute_trade(self, decision):
        """Executes an order based on decision dict."""
        action = decision.get('action')
        price = decision.get('price', 0.0) # Assume price comes from decision or current market
        volume = decision.get('volume', 0.01)
        symbol = decision.get('symbol', 'EURUSD')

        if action == "HOLD":
            return
            
        self.logger.info(f"Executing Trade: {decision}")
        
        # 1. Shadow Mode Check
        is_shadow = self.shadow_mode or self.is_mock
        
        if is_shadow:
             self.logger.info(f"👻 [SHADOW] Trade Executed: {action} {symbol} @ {price}")
             # Record to DB
             if self.storage:
                 self.storage.log_trade(symbol, action, price, volume, is_shadow=True, comment="Shadow Trade")
             return True

        # 2. Real Execution Logic (Stub)
        # request = { ... }
        # result = mt5.order_send(request)
        # if result.retcode == mt5.TRADE_RETCODE_DONE:
        #      if self.storage:
        #          self.storage.log_trade(...)
        
        return False

    def shutdown(self):
        if self.is_mock:
            self.logger.info("Mock MT5 Shutdown.")
        elif MT5_AVAILABLE:
            mt5.shutdown()
