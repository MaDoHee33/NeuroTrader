import pandas as pd
import numpy as np

class PerformanceMetrics:
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_np = np.array(returns)
        mean_return = np.mean(returns_np)
        std_return = np.std(returns_np)
        
        if std_return == 0:
            return 0.0
            
        # Annualized Sharpe (assuming daily returns input, simplified)
        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe * np.sqrt(252)

    @staticmethod
    def calculate_max_drawdown(equity_curve):
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
            
        equity_np = np.array(equity_curve)
        peak = equity_np[0]
        max_dd = 0.0
        
        for value in equity_np:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd * 100 # Percentage
