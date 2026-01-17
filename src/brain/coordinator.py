import logging
import asyncio
from src.brain.llm_processor import LLMProcessor
from src.brain.ml_predictor import MLPredictor
from src.brain.rl_agent import RLAgent

class BrainCoordinator:
    def __init__(self, config=None):
        self.logger = logging.getLogger("Brain")
        self.llm = LLMProcessor(config)
        self.ml = MLPredictor(config)
        self.rl = RLAgent(config)

    async def process(self, market_data):
        """
        Main cognitive loop:
        1. Parse Data
        2. Get signals from LLM (News/Sentiment) and ML (Price Forecast)
        3. Pass signals to RL Agent for final decision
        """
        self.logger.info("Brain Processing Data...")
        
        # In a real async architecture, some of these might run in parallel
        # or have their own loops updating state.
        
        # 1. LLM Analysis (Mocked/Stubbed for speed)
        sentiment = await self.llm.analyze(market_data)
        
        # 2. ML Prediction
        forecast = await self.ml.predict(market_data)
        
        # 3. RL Decision
        decision = await self.rl.decide(market_data, sentiment, forecast)
        
        self.logger.info(f"Decision: {decision}")
        return decision
