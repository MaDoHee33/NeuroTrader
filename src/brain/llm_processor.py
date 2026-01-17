import logging
import asyncio
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain # Deprecated/Removed

class LLMProcessor:
    def __init__(self, config=None):
        self.logger = logging.getLogger("LLM")
        self.config = config or {}
        
        llm_conf = self.config.get('llm', {})
        self.model_name = llm_conf.get('model_name', 'mistral')
        self.base_url = llm_conf.get('base_url', 'http://localhost:11434')
        self.low_vram = llm_conf.get('low_vram', False)
        
        self.llm = None
        self.chain = None
        self._initialize_llm()

    def _initialize_llm(self):
        try:
            self.logger.info(f"Initializing Local LLM (Ollama): {self.model_name}")
            if self.low_vram:
                self.logger.info("⚡ Low VRAM Mode Enabled: Limiting context size to 2048")

            self.llm = Ollama(
                base_url=self.base_url,
                model=self.model_name,
                temperature=0.3, # Low temp for analytical tasks
                num_ctx=2048 if self.low_vram else 4096, # Limit context window in low vram mode
            )
            
            # Simple Sentiment Analysis Prompt
            template = """
            You are a senior financial analyst. Analyze the following market data and news provided below.
            
            Market Data: {market_data}
            
            Task:
            1. Determine the sentiment (BULLISH, BEARISH, NEUTRAL).
            2. Provide a confidence score (0.0 to 1.0).
            3. Give a concise reasoning.
            
            Output format:
            Sentiment: [BULLISH/BEARISH/NEUTRAL]
            Confidence: [0.0-1.0]
            Reason: [One sentence summary]
            """
            
            prompt = PromptTemplate(template=template, input_variables=["market_data"])
            self.chain = prompt | self.llm # LCEL Syntax
            
        except Exception as e:
            self.logger.error(f"Failed to init LLM: {e}")

    async def analyze(self, market_data):
        """
        Analyzes market data using local Ollama instance.
        """
        if not self.llm:
            return {"sentiment": "NEUTRAL", "confidence": 0.0, "reason": "LLM Offline"}

        self.logger.info("🧠 Thinking (Local Brain)...")
        
        try:
            # Run in executor to avoid blocking asyncio loop
            # LangChain invoke can be blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.chain.invoke, {"market_data": str(market_data)})
            
            self.logger.debug(f"LLM Output: {response}")
            
            # Simple parsing (In production, use PydanticOutputParser)
            # LCEL with Ollama might return just string or an object depending on setup.
            # Usually Ollama LLM wrapper returns string.
            text_response = str(response)
            
            sentiment = "NEUTRAL"
            if "BULLISH" in text_response.upper(): sentiment = "BULLISH"
            elif "BEARISH" in text_response.upper(): sentiment = "BEARISH"
            
            return {
                "sentiment_score": 1.0 if sentiment == "BULLISH" else -1.0 if sentiment == "BEARISH" else 0.0,
                "raw_output": response.strip()
            }
            
        except Exception as e:
            self.logger.error(f"LLM Analysis Failed: {e}")
            return {"sentiment_score": 0.0, "error": str(e)}
