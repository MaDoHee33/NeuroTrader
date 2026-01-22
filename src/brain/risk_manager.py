
import logging
from datetime import datetime, date
from src.data.economic_calendar import EconomicCalendar

class RiskManager:
    """
    Enforces hard-coded risk rules to protect capital from AI errors.
    """
    def __init__(self, config=None):
        self.logger = logging.getLogger("RiskManager")
        self.config = config or {}
        
        # 1. Hard Limits
        # Default: Max 1.0 Lot per trade
        self.max_lots_per_trade = self.config.get('max_lots', 1.0)
        
        # Default: 5% Daily Loss Limit
        self.daily_loss_limit_pct = self.config.get('daily_loss_pct', 0.05)
        
        # Default: 20% Max Drawdown (Circuit Breaker)
        self.max_drawdown_limit_pct = self.config.get('max_drawdown_pct', 0.20)
        
        # 3. News / Sentiment
        self.calendar = EconomicCalendar(self.config.get('calendar_csv_path'))
        self.news_window_minutes = self.config.get('news_window_minutes', 30)

        # 2. State Tracking
        self.current_date = date.today()
        self.daily_start_balance = None
        self.current_balance = None
        self.peak_balance = None
        
        self.is_circuit_breaker_active = False
        self.daily_stop_triggered = False

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

        # Rule 2: Max Lots
        if volume > self.max_lots_per_trade:
            self.logger.warning(f"Order Blocked: Volume {volume} exceeds limit {self.max_lots_per_trade}")
            return False
            
        # Rule 3: News Filter (High Impact)
        if current_time:
             is_news, event_name = self.calendar.is_high_impact_window(current_time, self.news_window_minutes)
             if is_news:
                 self.logger.warning(f"Order Blocked: High Impact News Event Nearby ({event_name})")
                 return False

        return True

    def get_status(self):
        return {
            "circuit_breaker": self.is_circuit_breaker_active,
            "daily_stop": self.daily_stop_triggered,
            "balance": self.current_balance
        }
