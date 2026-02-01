
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.identifiers import Venue, InstrumentId, Symbol
from nautilus_trader.model.objects import Money, Quantity
from nautilus_trader.model.position import Position
from nautilus_trader.portfolio.account import MarginAccount

def test():
    # 1. Inspect Money
    m = Money(100, USD)
    print(f"Money as double: {m.as_double()}")
    
    # 2. Inspect Account
    acct = MarginAccount(Venue("SIM"), base_currency=USD)
    # We can't easily populate it without engine events but we can check attributes
    print(f"Account.balance_free type: {type(acct.balance_free)}")
    # It seems balance_free is a property in Python, but let's check.
    
    # 3. Inspect Position
    # Position init is complex, need InstrumentId
    iid = InstrumentId(Symbol("XAUUSD"), Venue("SIM"))
    # Position constructor is internal usually, but lets try
    try:
        p = Position(
            instrument_id=iid,
            account_id=acct.id,
            init_qty=Quantity(1.0, 2),
            init_entry=None,
            ts_init=0
        )
        print(f"Position.quantity type: {type(p.quantity)}")
    except Exception as e:
        print(f"Skipping Position check: {e}")

if __name__ == "__main__":
    test()
