from nautilus_trader.model.data import CustomData
from nautilus_trader.model.identifiers import InstrumentId
import pandas as pd

class SentimentData(CustomData):
    """
    Custom Data type for Sentiment Analysis.
    Carries a 'score' (-1.0 to 1.0) and a 'source' (e.g., 'News', 'Twitter').
    """
    def __init__(
        self,
        instrument_id: InstrumentId,
        ts_event: int, # UNIX nanoseconds
        score: float,
        source: str = "Unknown",
        ts_init: int = None,
    ):
        super().__init__(
            ts_event=ts_event,
            ts_init=ts_init if ts_init else ts_event
        )
        self.instrument_id = instrument_id
        self.score = score
        self.source = source

    @classmethod
    def from_dict(cls, data):
        return cls(
            instrument_id=InstrumentId.from_str(data['instrument_id']),
            ts_event=data['ts_event'],
            score=data['score'],
            source=data.get('source', 'Unknown')
        )

    def to_dict(self):
        return {
            'type': 'SentimentData',
            'instrument_id': self.instrument_id.value,
            'ts_event': self.ts_event,
            'score': self.score,
            'source': self.source
        }
