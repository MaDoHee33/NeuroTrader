
import logging
from datetime import datetime, date
from src.skills.news_watcher import NewsWatcher

class RiskManager:
    """
    Enforces hard-coded risk rules to protect capital from AI errors.
    """
    def __init__(self, config=None):
        self.logger = logging.getLogger("RiskManager")
        self.config = config or {}
        
        # Skills
        self.news_watcher = NewsWatcher(impact_levels=['High'], buffer_minutes=30)
        # Try update on init (for live mode), but don't crash if offline
        # try:
        #     self.news_watcher.update_calendar()
        # except:
        #     pass
        
        # 1. Hard Limits
        # Dynamic Lot Sizing Configuration
        self.dynamic_lot_sizing = self.config.get('dynamic_lot_sizing', True) # Enable by default for V2.1
        self.lots_per_10k_equity = self.config.get('lots_per_10k', 5.0)       # Allow 5.0 lots per $10k
        
        # Default: Max 1.0 Lot per trade (Fallback/Fixed)
        self.max_lots_per_trade = self.config.get('max_lots', 1.0)
        
        # Default: 5% Daily Loss Limit
        self.daily_loss_limit_pct = self.config.get('daily_loss_pct', 0.05)
        
        # Default: 20% Max Drawdown (Circuit Breaker)
        self.max_drawdown_limit_pct = self.config.get('max_drawdown_pct', 0.20)

        # Default: Turbulence Threshold (Typical Chi-Square cutoff for 3 degrees of freedom is ~7.8 for 95%, 11.3 for 99%)
        # We start conservative with 15.0 or dynamic
        self.turbulence_limit = self.config.get('turbulence_limit', 15.0)
        
        # 3. News / Sentiment
        # self.calendar = EconomicCalendar(self.config.get('calendar_csv_path'))
        self.calendar = None # Disabled because file is missing
        self.news_window_minutes = self.config.get('news_window_minutes', 30)

        # 2. State Tracking
        self.current_date = date.today()
        self.daily_start_balance = None
        self.current_balance = None
        self.peak_balance = None
        
        self.is_circuit_breaker_active = False
        self.daily_stop_triggered = False
        self.turbulence_triggered = False

    def reset_daily_metrics(self, balance):
        """Resets daily trackers at start of new day."""
        self.current_date = date.today()
        self.daily_start_balance = balance
        self.daily_stop_triggered = False
        self.logger.info(f"RiskManager: New Day Reset. Start Balance: {balance}")

    def update_metrics(self, balance, current_time=None):
        """Updates balance tracking and checks for stop-loss triggers."""
        # Initialize if first run
        if self.daily_start_balance is None:
            self.daily_start_balance = balance
            self.peak_balance = balance
        
        # Handle day change
        now_date = current_time.date() if current_time else date.today()
        if now_date != self.current_date:
            self.reset_daily_metrics(balance)
            
        self.current_balance = balance
        
        # Update Peak (High Water Mark)
        if self.peak_balance is None or balance > self.peak_balance:
            self.peak_balance = balance

        # 1. Check Circuit Breaker (Total Drawdown)
        if self.peak_balance > 0:
            drawdown_pct = (self.peak_balance - balance) / self.peak_balance
        else:
            drawdown_pct = 0.0
            
        if drawdown_pct >= self.max_drawdown_limit_pct:
            if not self.is_circuit_breaker_active:
                self.logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED! Drawdown: {drawdown_pct:.2%}")
                self.is_circuit_breaker_active = True
        
        # 2. Check Daily Loss
        daily_pnl = balance - self.daily_start_balance
        daily_loss_pct = -daily_pnl / self.daily_start_balance
        
        if daily_loss_pct >= self.daily_loss_limit_pct:
            if not self.daily_stop_triggered:
                self.logger.warning(f"🛑 DAILY LOSS LIMIT REACHED! Loss: {daily_loss_pct:.2%}")
                self.daily_stop_triggered = True

    def check_turbulence(self, turbulence_index):
        """
        Checks if market turbulence exceeds safety threshold.
        """
        if turbulence_index > self.turbulence_limit:
            if not self.turbulence_triggered:
                self.logger.warning(f"🌪️ TURBULENCE ALERT! Index: {turbulence_index:.2f} > {self.turbulence_limit}")
                self.turbulence_triggered = True
        else:
             if self.turbulence_triggered:
                 self.logger.info(f"🌤️ Turbulence subsided. Index: {turbulence_index:.2f}")
                 self.turbulence_triggered = False


    def check_order(self, symbol, volume, order_type, current_time=None):
        """
        Validates an order against risk rules.
        Returns: True (Allowed), False (Blocked)
        """
        # Rule 0: Circuit Breaker
        if self.is_circuit_breaker_active:
            self.logger.warning("Order Blocked: Circuit Breaker Active.")
            return False
            
        # Rule 1: Daily Stop
        if self.daily_stop_triggered:
            self.logger.warning("Order Blocked: Daily Loss Limit Reached.")
            return False

        # Rule 1.5: Turbulence
        if self.turbulence_triggered:
             self.logger.warning("Order Blocked: High Market Turbulence.")
             return False

        # Rule 2: Max Lots (Dynamic)
        # Use current balance, fallback to daily start, fallback to 0
        balance_for_calc = self.current_balance if self.current_balance is not None else self.daily_start_balance
        
        if self.dynamic_lot_sizing and balance_for_calc:
             # Scale allowed lots based on equity
             # e.g. Balance $5000 -> (5000/10000) * 5.0 = 2.5 Lots allowed
             allowed_lots = (balance_for_calc / 10000.0) * self.lots_per_10k_equity
             # Ensure at least minimal trade possibility (e.g. 0.01 lot) or fallback
             allowed_lots = max(allowed_lots, 0.01) 
        else:
             allowed_lots = self.max_lots_per_trade

        if volume > allowed_lots:
            self.logger.warning(f"Order Blocked: Volume {volume:.2f} exceeds limit {allowed_lots:.2f} (Balance: {self.current_balance})")
            return False
            
        # Rule 3: News Filter (High Impact)
        if current_time and self.news_watcher:
             is_danger, reason = self.news_watcher.is_market_dangerous(current_time)
             if is_danger:
                 self.logger.warning(f"Order Blocked: {reason}")
                 return False

        return True

    def get_status(self):
        return {
            "circuit_breaker": self.is_circuit_breaker_active,
            "daily_stop": self.daily_stop_triggered,
            "balance": self.current_balance
        }
