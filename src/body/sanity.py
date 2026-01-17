import logging
import pandas as pd
import numpy as np

class DataSanity:
    def __init__(self):
        self.logger = logging.getLogger("Sanity")

    def validate_tick(self, tick_data):
        """
        Validates incoming tick data.
        Returns True if valid, False otherwise.
        """
        if not tick_data:
            return False
            
        required_keys = ['symbol', 'bid', 'ask', 'time']
        if not all(k in tick_data for k in required_keys):
            self.logger.warning(f"Malformed tick data: {tick_data}")
            return False
            
        if tick_data['bid'] <= 0 or tick_data['ask'] <= 0:
            self.logger.warning(f"Zero/Negative price detected: {tick_data}")
            return False
            
        if tick_data['bid'] > tick_data['ask']:
             # Inverted spread?
             self.logger.warning(f"Inverted spread detected: {tick_data}")
             return False

        return True

    def validate_history(self, df: pd.DataFrame):
        """
        Validates historical candle dataframe.
        """
        if df.empty:
            return False
        
        # Check for NaNs
        if df.isnull().values.any():
            self.logger.warning("NaNs found in historical data.")
            return False
            
        return True
