
import pytest
import pandas as pd
import numpy as np
from src.brain.env.trading_env import TradingEnv

class TestScalperMechanics:
    """
    Test suite to verify Scalper mechanics fixes:
    1. Timer Persistence: specific check that pyramiding does NOT reset steps_in_position
    2. Bonus Cap: specific check that entry bonus is only given on first entry
    """
    
    def setup_method(self):
        # Create a dummy dataframe
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        self.df = pd.DataFrame({
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 100.0,
            'volume': 1000,
            'time': dates
        }, index=dates)
        
        # Initialize Env for Scalper
        self.env = TradingEnv(
            {'TEST_ASSET': self.df}, 
            agent_type='scalper',
            initial_balance=10000,
            risk_config={
                'max_lots': 100.0,
                'dynamic_lot_sizing': False
            }
        )
        self.env.reset()

    def test_timer_persistence_on_pyramiding(self):
        """
        Verify that adding to a position (Pyramiding) does NOT reset the holding timer.
        """
        # Step 1: Initial Entry
        # Action 1 = BUY
        obs, reward, done, _, info = self.env.step(1)
        assert self.env.position > 0
        assert self.env.steps_in_position == 1  # 1 after first step (update happens at end of step)
        
        # Step 2: Hold for 5 steps
        for _ in range(5):
            self.env.step(0) # HOLD
            
        current_timer = self.env.steps_in_position
        assert current_timer == 6
        
        # Step 3: Pyramid (Buy Again)
        # Action 1 = BUY
        # PRE-FIX BEHAVIOR: This would reset steps_in_position to 0
        # POST-FIX BEHAVIOR: This should keep it at 5 (or increment to 6 depending on update order)
        self.env.step(1) 
        
        new_timer = self.env.steps_in_position
        
        print(f"Timer before pyramid: {current_timer}, Timer after pyramid: {new_timer}")
        
        # CRITICAL ASSERTION
        assert new_timer >= 5, f"Timer reset detected! Went from {current_timer} to {new_timer}"

    def test_entry_bonus_cap(self):
        """
        Verify that Entry Bonus (+0.08) is ONLY awarded on the first entry.
        """
        # Force a configuration where we can isolate the bonus
        # We need to know what the bonus is. In code it's currently hardcoded 0.08 for scalper.
        
        # Step 1: First Entry
        # We expect a positive reward that includes the bonus
        obs, reward_1, done, _, info_1 = self.env.step(1)
        
        # Calculate expected components approximately
        # Since price didn't change (close=100 every step), PnL reward is 0 (minus fees/spread)
        # But we get Entry Bonus +0.08
        print(f"Reward 1 (First Entry): {reward_1}")
        
        # Step 2: Hold
        self.env.step(0)
        
        # Step 3: Second Entry (Pyramid)
        # Should NOT have the large 0.08 bonus this time
        obs, reward_2, done, _, info_2 = self.env.step(1)
        print(f"Reward 2 (Pyramid): {reward_2}")
        
        # The rewards might differ due to fees on larger size, but the bonus is significant (0.08).
        # Without bonus, reward should be negative (transaction cost) or 0.
        # With bonus, it would be positive.
        
        # We assert that Reward 2 is significantly less than Reward 1 (assuming price is flat)
        # Or specifically check if it's devoid of the bonus.
        
        # If PnL is 0, Reward 1 ~= 0.08 - transaction_cost
        # Reward 2 should just be -transaction_cost
        
        assert reward_1 > 0.05, "First entry did not get bonus"
        assert reward_2 < 0.05, "Second entry INVALIDLY got bonus!"

