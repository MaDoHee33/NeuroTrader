
import unittest
from datetime import date, timedelta, datetime
from src.brain.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.config = {
            'max_lots': 1.0,
            'daily_loss_pct': 0.05,
            'max_drawdown_pct': 0.20
        }
        self.rm = RiskManager(self.config)

    def test_max_lots_check(self):
        # Allowed
        self.assertTrue(self.rm.check_order("XAUUSD", 0.5, "BUY"))
        self.assertTrue(self.rm.check_order("XAUUSD", 1.0, "BUY"))
        # Blocked
        self.assertFalse(self.rm.check_order("XAUUSD", 1.01, "BUY"))

    def test_daily_loss_limit(self):
        # Start Day
        start_bal = 10000
        self.rm.update_metrics(start_bal, datetime.now())
        
        # Loss 4% -> OK
        current_bal = 9600
        self.rm.update_metrics(current_bal, datetime.now())
        self.assertFalse(self.rm.daily_stop_triggered)
        self.assertTrue(self.rm.check_order("XAUUSD", 0.1, "BUY"))
        
        # Loss 6% -> Triggered
        current_bal = 9400
        self.rm.update_metrics(current_bal, datetime.now())
        self.assertTrue(self.rm.daily_stop_triggered)
        self.assertFalse(self.rm.check_order("XAUUSD", 0.1, "BUY"))

    def test_new_day_reset(self):
        # Day 1: Trigger Limit
        today = datetime.now()
        self.rm.update_metrics(10000, today)
        self.rm.update_metrics(9000, today) # -10% -> Triggered
        self.assertTrue(self.rm.daily_stop_triggered)
        
        # Day 2: Should Reset
        tomorrow = today + timedelta(days=1)
        self.rm.update_metrics(9000, tomorrow) # New start balance = 9000
        self.assertFalse(self.rm.daily_stop_triggered)
        self.assertTrue(self.rm.check_order("XAUUSD", 0.1, "BUY"))

    def test_circuit_breaker(self):
        # Start
        self.rm.update_metrics(10000, datetime.now()) # Peak = 10000
        
        # Drawdown 15% -> OK
        self.rm.update_metrics(8500, datetime.now())
        self.assertFalse(self.rm.is_circuit_breaker_active)
        
        # Drawdown 21% -> Triggered
        self.rm.update_metrics(7900, datetime.now())
        self.assertTrue(self.rm.is_circuit_breaker_active)
        self.assertFalse(self.rm.check_order("XAUUSD", 0.1, "BUY"))
        
        # Should stay triggered even if balance recovers slightly
        self.rm.update_metrics(8000, datetime.now())
        self.assertTrue(self.rm.is_circuit_breaker_active)

if __name__ == '__main__':
    unittest.main()
