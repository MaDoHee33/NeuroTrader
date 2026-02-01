from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.identifiers import InstrumentId

class NeuroNautilusConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: str = "5-MINUTE-LAST"
    model_path: str = "models/checkpoints/ppo_neurotrader.zip"  # Path to trained model
    agent_config: dict = {}
