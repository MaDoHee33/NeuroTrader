#!/usr/bin/env python3
"""
NeuroNautilus Backtest Runner
Supports both Local and Colab environments with automatic logging
"""

import logging
import argparse
import sys
import os
from decimal import Decimal
from datetime import datetime
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.identifiers import Venue, InstrumentId, Symbol
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.enums import AccountType, OmsType, BarAggregation, PriceType
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.config import LoggingConfig

from src.neuro_nautilus.config import NeuroNautilusConfig
from src.neuro_nautilus.strategy import NeuroBridgeStrategy

# Environment Detection
def is_colab():
    try:
        import google.colab
        return True
    except:
        return False

def setup_logging(log_path=None):
    """Setup logging to both console and file"""
    if log_path is None:
        # Auto-generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs/backtest")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"backtest_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return log_path

def get_default_paths():
    """Get default paths based on environment"""
    if is_colab():
        workspace = "/content/drive/MyDrive/NeuroTrader_Workspace"
        return {
            'data': f"{workspace}/data/nautilus_catalog",
            'model': f"{workspace}/models/checkpoints/ppo_neurotrader.zip",
            'logs': f"{workspace}/logs/backtest"
        }
    else:
        base = Path(__file__).resolve().parent.parent.parent
        return {
            'data': str(base / "data" / "nautilus_catalog"),
            'model': str(base / "models" / "checkpoints" / "ppo_neurotrader.zip"),
            'logs': str(base / "logs" / "backtest")
        }

def run_backtest(args):
    """Main backtest execution function"""
    
    # Setup logging
    log_path = setup_logging(args.log_path)
    logger = logging.getLogger(__name__)
    
    logger.info(f"{'='*60}")
    logger.info(f"🧠 NeuroNautilus Backtest")
    logger.info(f"📍 Environment: {'Colab' if is_colab() else 'Local'}")
    logger.info(f"📝 Log file: {log_path}")
    logger.info(f"{'='*60}\n")
    
    # Configure Backtest Engine
    engine_config = BacktestEngineConfig(
        trader_id="NEURO-BOT-01",
        logging=LoggingConfig(log_level="WARNING")  # Suppress verbose logs
    )
    engine = BacktestEngine(config=engine_config)

    # Setup Venue & Account
    venue = Venue("SIM")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(args.initial_balance, USD)]
    )

    # Define instrument
    instrument = CurrencyPair(
        instrument_id=InstrumentId(Symbol("XAUUSD"), venue),
        raw_symbol=Symbol("XAUUSD"),
        base_currency=USD,
        quote_currency=USD,
        price_precision=3,
        size_precision=0,  # Match data volume precision
        price_increment=Price(0.001, 3),
        size_increment=Quantity(1, 0),  # Match precision
        lot_size=None,
        max_quantity=Quantity(1000, 0),
        min_quantity=Quantity(1, 0),
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

    # Load Data
    logger.info(f"📂 Loading data from: {args.data_dir}")
    logger.info(f"📊 Bar type: {args.bar_type}")
    
    catalog = ParquetDataCatalog(args.data_dir)
    bar_type = BarType.from_str(f"{args.bar_type}")
    
    try:
        bars = list(catalog.bars(bar_types=[bar_type]))
        if not bars:
            logger.error(f"❌ No bars found for {bar_type}")
            logger.info("💡 Available bar types:")
            # Try to list what's available (simplified)
            return
        
        logger.info(f"✅ Loaded {len(bars):,} bars")
        logger.info(f"   First bar: {bars[0].ts_init}")
        logger.info(f"   Last bar: {bars[-1].ts_init}")
        engine.add_data(bars)
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return

    # Add Strategy
    config = NeuroNautilusConfig(
        instrument_id=instrument.id,
        bar_type=args.bar_type.split('-')[1] + "-" + args.bar_type.split('-')[2] + "-" + args.bar_type.split('-')[3],
        model_path=args.model_path
    )
    
    strategy = NeuroBridgeStrategy(config)
    engine.add_strategy(strategy=strategy)

    # Run backtest
    logger.info("\n🏃 Starting backtest...\n")
    engine.run()
    
    # Results
    logger.info("\n🏁 Backtest Complete!")
    logger.info(f"\n{'='*60}")
    logger.info("📊 RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    
    engine.trader.generate_account_report(venue)
    
    logger.info(f"\n💾 Full log saved to: {log_path}")
    
    return engine

def main():
    parser = argparse.ArgumentParser(description='Run NeuroNautilus Backtest')
    
    defaults = get_default_paths()
    
    parser.add_argument('--data-dir', type=str, default=defaults['data'],
                        help='Path to Nautilus data catalog')
    parser.add_argument('--model-path', type=str, default=defaults['model'],
                        help='Path to trained model (.zip file)')
    parser.add_argument('--bar-type', type=str, 
                        default='XAUUSD.SIM-15-MINUTE-LAST-EXTERNAL',
                        help='Bar type to backtest (e.g., XAUUSD.SIM-5-MINUTE-LAST-EXTERNAL)')
    parser.add_argument('--initial-balance', type=str, default="10000.00",
                        help='Starting account balance')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--log-path', type=str, default=None,
                        help='Custom log file path (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    run_backtest(args)

if __name__ == "__main__":
    main()
