from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, desc
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class TradeHistory(Base):
    __tablename__ = 'trade_history'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    symbol = Column(String)
    action = Column(String) # BUY, SELL
    price = Column(Float)
    volume = Column(Float)
    is_shadow = Column(Boolean, default=False)
    profit = Column(Float, nullable=True) # Filled when closing trade
    comment = Column(String, nullable=True)

class StorageEngine:
    def __init__(self, db_path="data/memory/neurotrader.db"):
        # Ensure dir exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def log_trade(self, symbol, action, price, volume, is_shadow=False, comment=None):
        session = self.Session()
        try:
            trade = TradeHistory(
                symbol=symbol,
                action=action,
                price=price,
                volume=volume,
                is_shadow=is_shadow,
                comment=comment
            )
            session.add(trade)
            session.commit()
            return trade.id
        except Exception as e:
            print(f"DB Error: {e}")
            session.rollback()
        finally:
            session.close()

    def get_recent_trades(self, limit=50):
        session = self.Session()
        try:
            return session.query(TradeHistory).order_by(desc(TradeHistory.timestamp)).limit(limit).all()
        finally:
            session.close()
