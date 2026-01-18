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
        # Convert Bar to dict for agent
        bar_dict = {
            'timestamp': bar.ts_init,
            'open': bar.open.as_double(),
            'high': bar.high.as_double(),
            'low': bar.low.as_double(),
            'close': bar.close.as_double(),
            'volume': bar.volume.as_double()
        }
        
        # Get portfolio state - simplified and robust
        try:
            # Get balance
            account = self.portfolio.account(self.instrument_id.venue)
            balance = 10000.0  # Default
            if account and hasattr(account, 'balance_total'):
                try:
                    balance_obj = account.balance_total(USD)
                    if balance_obj:
                        balance = float(balance_obj.as_double())
                except:
                    pass
            
            # Get position
            position = 0.0  # Default
            if hasattr(self.portfolio, 'net_position'):
                try:
                    pos = self.portfolio.net_position(self.instrument_id)
                    if pos and hasattr(pos, 'as_double'):
                        position = float(pos.as_double())
                    elif pos:
                        position = float(pos)
                except:
                    pass
            
            portfolio_state = {
                'balance': balance,
                'position': position
            }
        except Exception as e:
            # Fail gracefully with defaults
            portfolio_state = {
                'balance': 10000.0,
                'position': 0.0
            }
        
        # Get agent decision using process_bar
        try:
            action = self.agent.process_bar(bar_dict, portfolio_state)
            # process_bar returns int directly (0=HOLD, 1=BUY, 2=SELL)
        except Exception as e:
            self.log.error(f"Agent failed to decide: {e}")
            return
        
        # 3. Execute based on action
        volume = 0.01 # Default volume for trades

        if action == 1:  # BUY
            self.log.info(f"🤖 Agent says BUY at {bar.close}")
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.instrument.make_qty(volume),
                time_in_force=TimeInForce.GTC
            )
            self.submit_order(order)
        elif action == 2:  # SELL
            self.log.info(f"🤖 Agent says SELL at {bar.close}")
             # Check if we have position first? (Nautilus handles net positions, but let's be safe)
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=self.instrument.make_qty(volume),
                time_in_force=TimeInForce.GTC
            )
            self.submit_order(order)
        # else: HOLD (action == 0)
