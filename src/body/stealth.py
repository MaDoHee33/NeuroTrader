import asyncio
import random
import logging

class StealthLayer:
    def __init__(self, config=None):
        """
        Initialize StealthLayer.
        config: dict or None. If dict, expects 'stealth' key with 'jitter_range'.
        """
        self.config = config or {}
        stealth_conf = self.config.get('stealth', {})
        self.jitter_range = stealth_conf.get('jitter_range', [-0.5, 0.5])
        self.logger = logging.getLogger("Stealth")

    async def sleep_random(self, base_seconds=0.0):
        """
        Sleeps for a random duration to mimic human behavior or network variance.
        base_seconds: The ensures minimum sleep time before adding jitter.
        """
        jitter = random.uniform(self.jitter_range[0], self.jitter_range[1])
        # Ensure we never sleep for a negative amount or practically zero if not intended
        total_sleep = max(0.1, base_seconds + jitter) 
        
        # self.logger.debug(f"Jitter sleep: {total_sleep:.2f}s")
        await asyncio.sleep(total_sleep)
