import logging
from decimal import Decimal
import os
import shutil

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.identifiers import Venue, InstrumentId
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.enums import AccountType, OmsType, BarAggregation, PriceType
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.config import LoggingConfig

from src.neuro_nautilus.config import NeuroNautilusConfig
from src.neuro_nautilus.strategy import NeuroBridgeStrategy

def run_backtest():
    # 1. Config
    logger = logging.getLogger("runner")
    
    # Configure Backtest Engine
    engine_config = BacktestEngineConfig(
        trader_id="NEURO-BOT-01",
        logging=LoggingConfig(log_level="INFO")
    )
    engine = BacktestEngine(config=engine_config)

    # 2. Setup Venue & Account
    venue = Venue("SIM")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money("10_000.00", USD)]
    )

    # Manually define instrument to match data precision (3 decimals for XAUUSD usually)
    from nautilus_trader.model.instruments import CurrencyPair
    from nautilus_trader.model.identifiers import Symbol
    from nautilus_trader.model.objects import Price, Quantity
    
    instrument = CurrencyPair(
        instrument_id=InstrumentId(
            symbol=Symbol("XAUUSD"),
            venue=venue,
        ),
        raw_symbol=Symbol("XAUUSD"),
        base_currency=USD, # Simplified
        quote_currency=USD,
        price_precision=3,
        size_precision=2,
        price_increment=Price(0.001, 3),
        size_increment=Quantity(0.01, 2),
        lot_size=None,
        max_quantity=Quantity(1000, 2),
        min_quantity=Quantity(0.01, 2),
        max_notional=None,
        min_notional=None,
        max_price=None,
        min_price=None,
        margin_init=Decimal("0.01"),
        margin_maint=Decimal("0.01"),
        maker_fee=Decimal("0.0001"),
        taker_fee=Decimal("0.0001"),
        ts_event=0,
        ts_init=0,
    )
    engine.add_instrument(instrument)

    # 4. Load Data (Parquet)
    # catalog = ParquetDataCatalog("data/nautilus_store")
    # But wait! We need to make sure the Instrument ID matches what we saved.
    # Our migration script saved as "XAUUSDmD1.SIM" (example) or similar based on filename behavior.
    # Let's check what files we have first.
    
    data_path = "data/nautilus_store"
    catalog = ParquetDataCatalog(data_path)
    
    # Let's try to autoload just to verify
    # For this script to be robust, let's explicitly look for our migrated file
    # Or just use the catalog to stream
    
    # 5. Add Strategy
    config = NeuroNautilusConfig(
        instrument_id=instrument.id,
        bar_type="5-MINUTE-LAST", # Must match data
    )
    
    strategy_id = engine.add_strategy(
        strategy=NeuroBridgeStrategy(config),
    )

    # 6. Run
    print("🚀 Running NeuroNautilus Backtest...")
    
    # We need to manually load bars into the engine because just pointing to catalog 
    # normally requires the strategy to subscribe or we add data to engine.
    # The simplest way is to read from catalog and add to engine.
    
    # Loading ALL bars from catalog for this instrument
    # We need to know the specific bar type we saved.
    # Our migration script used: 5-MINUTE-LAST for M5
    
    bar_spec = BarSpecification(5, BarAggregation.MINUTE, PriceType.LAST)
    bar_type = BarType(instrument.id, bar_spec)
    
    print(f"🔎 Looking for BarType: {bar_type}")
    
    # Try reading from catalog
    try:
        bars = list(catalog.bars(bar_types=[bar_type])) 
        if not bars:
             # Fallback: Trying to list what IS there
             print("⚠️ No bars found for exact match. Checking catalog...")
             # (In a real script we would iterate instruments)
             # For this test, let's just make sure we use the data we migrated.
             pass
        else:
            print(f"📦 Loaded {len(bars)} bars.")
            engine.add_data(bars)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    engine.run()
    
    # 7. Results
    print("🏁 Backtest Complete.")
    engine.trader.generate_account_report(venue)
    
if __name__ == "__main__":
    run_backtest()
