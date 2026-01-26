from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity

from src.brain.rl_agent import RLAgent
from src.brain.risk_manager import RiskManager
from src.neuro_nautilus.config import NeuroNautilusConfig
import asyncio

class NeuroBridgeStrategy(Strategy):
    def __init__(self, config: NeuroNautilusConfig):
        super().__init__(config)
        self.instrument_id = config.instrument_id
        
        # Initialize the Brain
        self.agent = RLAgent(model_path=config.model_path)
        
        # Initialize Risk Manager
        self.risk_manager = RiskManager(config.agent_config)
        self.risk_manager.reset_daily_metrics(10000.0) # Init with default or fetch later
        
        # Progress tracking (to avoid log spam)
        self.bar_count = 0
        self.trade_count = 0
        
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
        # Progress tracking - log every 1000 bars to avoid spam
        self.bar_count += 1
        if self.bar_count % 1000 == 0:
            self.log.info(f"📊 Processed {self.bar_count:,} bars | Trades: {self.trade_count}")
        
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
            
            # Update Risk Manager with real balance
            self.risk_manager.update_metrics(balance, current_time=bar.ts_init)

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
        # Expecting (action, turbulence_val) tuple now
        try:
            action, turbulence_val = self.agent.process_bar(bar_dict, portfolio_state)
            
            # 2.5 Check Turbulence
            self.risk_manager.check_turbulence(turbulence_val)
            
        except Exception as e:
            self.log.error(f"Agent failed to decide: {e}")
            return
        
        # 3. Execute based on action
        # Pre-check Risk Rules
        # Default volume (could be dynamic later)
        volume = 1 
        
        if not self.risk_manager.check_order(self.instrument_id, volume, action, current_time=bar.ts_init):
            return # Blocked by Risk Manager

        if action == 1:  # BUY
            self.trade_count += 1
            # Only log actual trades
            self.log.info(f"💰 BUY #{self.trade_count} @ {bar.close.as_double():.2f}")
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.instrument.make_qty(volume),
                time_in_force=TimeInForce.GTC
            )
            self.submit_order(order)
        elif action == 2:  # SELL
            self.trade_count += 1
            self.log.info(f"💸 SELL #{self.trade_count} @ {bar.close.as_double():.2f}")
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=self.instrument.make_qty(volume),
                time_in_force=TimeInForce.GTC
            )
            self.submit_order(order)
        # else: HOLD (action == 0) - no log spam
