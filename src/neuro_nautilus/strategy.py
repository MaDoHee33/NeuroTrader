from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity

from src.brain.rl_agent import RLAgent
from src.neuro_nautilus.config import NeuroNautilusConfig
import asyncio

class NeuroBridgeStrategy(Strategy):
    def __init__(self, config: NeuroNautilusConfig):
        super().__init__(config)
        self.instrument_id = config.instrument_id
        
        # Initialize the Brain
        self.agent = RLAgent(config.agent_config)
        self.log.info("🧠 NeuroBridgeStrategy Initialized")

    def on_start(self):
        # Cache instrument
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument {self.instrument_id} in cache")
            self.stop()
            return

        # Create BarType from config string + instrument ID
        # Assuming config.bar_type is "1-MINUTE-LAST"
        bar_spec = BarSpecification.from_str(self.config.bar_type)
        bar_type = BarType(self.instrument_id, bar_spec)
        self.subscribe_bars(bar_type)
        self.log.info(f"Subscribed to {bar_type}")

    def on_bar(self, bar: Bar):
        # self.log.info(f"DEBUG: on_bar {bar}") 
        # 1. Adapt Data
        # Simplified observation: just close price for now + dummy sentiment
        # In real scenario, we would build a full DataFrame buffer here
        market_data = {
            "close": bar.close.as_double(),
            "volume": bar.volume.as_double(),
            "high": bar.high.as_double(),
            "low": bar.low.as_double(),
            "open": bar.open.as_double(),
            "timestamp": bar.ts_event 
        }
        
        # 2. Ask Agent (Sync Wrapper)
        # We need to run the async decide method. 
        # Since Nautilus is sync (mostly) in on_bar, we can use asyncio.run or a stored loop.
        # For simplicity/robustness in Phase 2, we assume decision is fast.
        
        try:
            # Create a new loop if needed or get existing?
            # Warning: Nested loops can be tricky. 
            # Ideally RLAgent should have a synchronous 'decide_sync' method.
            # But let's try standard asyncio.run for this proof of concept.
            
            # Get Portfolio State
            try:
                account = self.portfolio.account(self.instrument_id.venue)
                # Try as propertry or method
                if callable(account.balance_free):
                    balance = account.balance_free().as_double()
                else:
                    balance = account.balance_free.as_double()
                    
                # Fix: Use net_position which is available in dir()
                if hasattr(self.portfolio, 'net_position'):
                    # net_position(instrument_id) -> float/Quantity
                    # It likely returns a signed quantity/float directly or a generic Position object
                    # Based on standard Nautilus, it usually returns the signed quantity directly or we check type
                    pos_val = self.portfolio.net_position(self.instrument_id)
                    try:
                         if hasattr(pos_val, 'as_double'):
                             position = pos_val.as_double()
                         else:
                             position = float(pos_val)
                    except:
                         position = 0.0
                else:
                    self.log.error(f"Portfolio has no 'net_position'. Dir: {dir(self.portfolio)}")
                    position = 0.0
            except Exception as e:
                self.log.error(f"Error getting portfolio state: {e}")
                balance = 10000.0
                position = 0.0
            
            portfolio_state = {'balance': balance, 'position': position}
            
            decision = asyncio.run(self.agent.decide(market_data, None, None, portfolio_state=portfolio_state))
        except Exception as e:
            self.log.error(f"Agent failed to decide: {e}")
            return

        # 3. Execute
        action = decision.get("action")
        volume = decision.get("volume", 0.01)

        if action == "BUY":
            self.log.info(f"🤖 Agent says BUY at {bar.close}")
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.instrument.make_qty(volume),
                time_in_force=TimeInForce.GTC
            )
            self.submit_order(order)
            
        elif action == "SELL":
            self.log.info(f"🤖 Agent says SELL at {bar.close}")
             # Check if we have position first? (Nautilus handles net positions, but let's be safe)
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=self.instrument.make_qty(volume),
                time_in_force=TimeInForce.GTC
            )
            self.submit_order(order)
