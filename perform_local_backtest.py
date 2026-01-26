
import sys
import os
from pathlib import Path
import pandas as pd
import logging
from decimal import Decimal

# Add src to path
sys.path.append(os.getcwd())

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.identifiers import Venue, InstrumentId, Symbol
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.config import LoggingConfig

from src.neuro_nautilus.config import NeuroNautilusConfig
from src.neuro_nautilus.strategy import NeuroBridgeStrategy

# Setup Paths
BASE_DIR = Path(os.getcwd())
DATA_PATH = BASE_DIR / 'data' / 'nautilus_catalog'
MODEL_PATH = BASE_DIR / 'models' / 'checkpoints' / 'ppo_neurotrader_v3.zip'

# if not MODEL_PATH.exists():
#     fallback = BASE_DIR / 'models' / 'L3_Hybrid' / 'final_model.zip'
#     MODEL_PATH = fallback

BAR_TYPE_STR = "XAUUSD.SIM-15-MINUTE-LAST-EXTERNAL"

# Configuration (Use small range for speed/verification)
START_DATE = pd.Timestamp("2023-01-01", tz="UTC")
END_DATE = pd.Timestamp("2023-06-01", tz="UTC")
INITIAL_BALANCE = 10000.0

print(f"🚀 Starting Optimized Local Backtest...")
print(f"📅 Period: {START_DATE.date()} to {END_DATE.date()}")
print(f"📂 Data: {DATA_PATH}")

# 1. Configure Engine
engine_config = BacktestEngineConfig(
    trader_id="NEURO-BOT-LOCAL",
    logging=LoggingConfig(log_level="WARNING")
)
engine = BacktestEngine(config=engine_config)

# 2. Add Venue & Instrument
venue = Venue("SIM")
engine.add_venue(
    venue=venue,
    oms_type=OmsType.NETTING,
    account_type=AccountType.MARGIN,
    base_currency=USD,
    starting_balances=[Money(INITIAL_BALANCE, USD)]
)

instrument = CurrencyPair(
    instrument_id=InstrumentId(Symbol("XAUUSD"), venue),
    raw_symbol=Symbol("XAUUSD"),
    base_currency=USD,
    quote_currency=USD,
    price_precision=3,
    size_precision=2,
    price_increment=Price(0.001, 3),
    size_increment=Quantity(0.01, 2),
    max_quantity=Quantity(1000, 2),
    min_quantity=Quantity(0.01, 2),
    margin_init=Decimal("0.01"),
    margin_maint=Decimal("0.01"),
    maker_fee=Decimal("0.0001"),
    taker_fee=Decimal("0.0001"),
    ts_event=0,
    ts_init=0,
)
engine.add_instrument(instrument)

# 3. Load & Slice Data
catalog = ParquetDataCatalog(str(DATA_PATH))
bar_type = BarType.from_str(BAR_TYPE_STR)

print("⏳ Loading Data...")
all_bars = list(catalog.bars(bar_types=[BAR_TYPE_STR]))

start_ns = START_DATE.value
end_ns = END_DATE.value
filtered_bars = [b for b in all_bars if start_ns <= b.ts_init <= end_ns]

if not filtered_bars:
    print("❌ No data found in date range!")
    sys.exit(1)

print(f"✅ Loaded {len(filtered_bars):,} bars")
engine.add_data(filtered_bars)

# 4. Strategy
config = NeuroNautilusConfig(
    instrument_id=instrument.id,
    bar_type=BAR_TYPE_STR.split('-')[1] + "-" + BAR_TYPE_STR.split('-')[2] + "-" + BAR_TYPE_STR.split('-')[3],
    model_path=str(MODEL_PATH)
)
strategy = NeuroBridgeStrategy(config)
engine.add_strategy(strategy=strategy)

# 5. Run
print("\n🏃 Running Backtest Logic...")
engine.run()

# 6. Analyze & Export Trade Log
print("\n📊 Analyzing Results...")

# Extract Executed Trades
# Use built-in report generator if available, or access via cache correctly
try:
    fills_df = engine.trader.generate_order_fills_report(venue)
    if fills_df is not None and not fills_df.empty:
        # Standardize columns for our CSV
        # Nautilus report usually has:
        # instrument_id, venue_order_id, trade_id, order_side, last_px, last_qty, commission, ts_init
        
        # Rename for clarity
        df_trades = fills_df.copy()
        df_trades['timestamp'] = pd.to_datetime(df_trades.index) # Often index is timestamp
        if 'ts_init' in df_trades.columns:
             df_trades['timestamp'] = pd.to_datetime(df_trades['ts_init'], unit='ns')
             
        df_trades = df_trades.sort_values('timestamp')
        
        csv_path = "backtest_trades.csv"
        df_trades.to_csv(csv_path)
        print(f"💾 Trade Log saved: {csv_path}")
        print(df_trades.head(5))
    else:
        print("⚠️ No trades executed (Empty Report)")
        
except Exception as e:
    print(f"⚠️ Could not extract trades via report: {e}")
    # Fallback to cache iterator if report fails
    try:
        trade_data = []
        # Try accessing orders and getting fills from them
        for order in engine.cache.orders():
             if order.filled_qty > 0:
                 trade_data.append({
                     "timestamp": order.ts_last, # Close enough
                     "symbol": order.instrument_id.symbol.value,
                     "side": order.side.name,
                     "price": float(order.avg_px) if order.avg_px else 0.0,
                     "quantity": float(order.filled_qty),
                     "type": "ORDER_FILLED"
                 })
        
        if trade_data:
            df_trades = pd.DataFrame(trade_data)
            df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'], unit='ns')
            df_trades.to_csv("backtest_trades.csv", index=False)
            print("💾 Trade Log saved (from Orders Cache): backtest_trades.csv")
    except Exception as e2:
         print(f"❌ Failed to extract trades completely: {e2}")


# ... (Existing Account Report Logic) ...
account_report = engine.trader.generate_account_report(venue)

# Safely get balance
if hasattr(account_report, 'empty') and not account_report.empty:
    # Try to find total balance column
    # Often 'total' or 'balance' or 'SIM' (if venue name is column?)
    # Nautilus reports are often index=Timestamp, columns=[Venue(Account)...] or flat
    
    # Assuming standard dataframe structure
    try:
        final_row = account_report.iloc[-1]
        
        # Look for something resembling balance (float)
        # Usually it has 'total' in name
        balance_col = [c for c in account_report.columns if 'total' in str(c).lower()]
        if not balance_col:
             balance_col = [c for c in account_report.columns if 'balance' in str(c).lower()]
        
        if balance_col:
            final_balance = float(final_row[balance_col[0]])
        else:
            # Fallback: take the first column
            final_balance = float(final_row.iloc[0])
            
    except Exception as e:
        print(f"⚠️ Error parsing dataframe: {e}")
        final_balance = INITIAL_BALANCE
        
else:
    print("⚠️ Account report is empty/invalid")
    final_balance = INITIAL_BALANCE

# Metrics
total_return = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE

# Trades
filled_orders = [o for o in engine.cache.orders() if o.is_closed and o.filled_qty > 0]
total_trades = len(filled_orders)
win_rate = 0.5  # Placeholder as we can't easily calculating per-trade PnL without parsing fills

# Output
print(f"   - Return: {total_return:.2%}")
print(f"   - Final Balance: ${final_balance:,.2f}")
print(f"   - Trades: {total_trades}")

report = f"""# NeuroNautilus Local Backtest Report
**Period:** {START_DATE.date()} to {END_DATE.date()}
**Model:** {MODEL_PATH.name}
**Data:** {BAR_TYPE_STR}

## Performance
- **Total Return:** {total_return:.2%}
- **Final Balance:** ${final_balance:,.2f}
- **Total Trades:** {total_trades}
- **Initial Balance:** ${INITIAL_BALANCE:,.2f}

## Conclusion
{'✅ PROFITABLE' if total_return > 0 else '❌ UNPROFITABLE' if total_return < 0 else '⚠️ NO TRADES / FLAT'}
"""

with open("local_backtest_report.md", "w") as f:
    f.write(report)
print("\n📝 Report saved to local_backtest_report.md")
