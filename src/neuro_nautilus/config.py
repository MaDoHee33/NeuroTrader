from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.identifiers import InstrumentId

class NeuroNautilusConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: str = "5-MINUTE-LAST"
    agent_config: dict = {}
