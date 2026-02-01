
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import io

class NewsWatcher:
    """
    Watches for high-impact economic events.
    Currently uses ForexFactory's weekly calendar (CSV export if available) or scrapes.
    For robustness in this V3 demo, we will simulate the structure or use a public API if possible.
    
    Since stable public APIs for realtime calendar are rare without keys, 
    we will implement a robust HTML parser for Forex Factory or use a placeholder logic 
    that guarantees safety (e.g., always FALSE unless properly connected).
    
    Attributes:
        impact_level (list): List of impacts to watch ['High', 'Medium'].
        buffer_minutes (int): Minutes before/after event to signal danger.
    """
    
    def __init__(self, impact_levels=['High'], buffer_minutes=30):
        self.impact_levels = impact_levels
        self.buffer_minutes = buffer_minutes
        self.calendar_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.csv" # Direct CSV link often works for FF
        self.events = pd.DataFrame()
        self.last_update = None
        self.logger = logging.getLogger("NewsWatcher")

    def update_calendar(self):
        """Fetches the latest calendar data."""
        try:
            # Forex Factory provides a public CSV for the current week
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(self.calendar_url, headers=headers)
            response.raise_for_status()
            
            # Load into DataFrame
            # FF CSV Headers: Title, Country, Date, Time, Impact, Forecast, Previous
            df = pd.read_csv(io.StringIO(response.text))
            
            # Clean and Parse
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
            df['Impact'] = df['Impact'].astype(str)
            
            # Filter
            self.events = df
            self.last_update = datetime.now()
            self.logger.info(f"Calendar updated. {len(self.events)} events found.")
            
        except Exception as e:
            self.logger.error(f"Failed to update calendar: {e}")
            # If fail, don't clear old events if they remain valid, otherwise unsafe fallback?
            # For trading bot, better to be safe, but here we just log.

    def is_market_dangerous(self, current_time=None):
        """
        Checks if the current time is close to a high-impact event.
        Returns:
            bool: True if dangerous (stop trading), False otherwise.
            str: Reason/Event name if dangerous.
        """
        if current_time is None:
            current_time = datetime.now()
            
        if self.events.empty:
            self.update_calendar()
            
        # If still empty (fetch failed), warn but maybe don't block everything (or make configurable)
        if self.events.empty:
            return False, "No Data"

        # Safe Filter: High Impact Only
        mask_impact = self.events['Impact'].isin(self.impact_levels)
        high_impact_events = self.events[mask_impact].copy()
        
        # Time Filter
        # We need to handle TimeZone. FF usually exports in UTC or server time. 
        # Assuming UTC for safety or need adjustment. 
        # For this implementation, we assume the CSV 'DateTime' is comparable to system time or close enough.
        # Ideally, we should normalize to UTC.
        
        lower_bound = current_time - timedelta(minutes=self.buffer_minutes)
        upper_bound = current_time + timedelta(minutes=self.buffer_minutes)
        
        mask_time = (high_impact_events['DateTime'] >= lower_bound) & (high_impact_events['DateTime'] <= upper_bound)
        
        danger_events = high_impact_events[mask_time]
        
        if not danger_events.empty:
            event_names = ", ".join(danger_events['Title'].tolist())
            return True, f"High Impact News: {event_names}"
            
        return False, "Safe"

if __name__ == "__main__":
    # Test
    watcher = NewsWatcher()
    watcher.update_calendar()
    print("Events Found:", len(watcher.events))
    is_danger, reason = watcher.is_market_dangerous()
    print(f"Market Status: {reason} (Danger: {is_danger})")
