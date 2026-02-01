
import pandas as pd
import numpy as np

def calculate_behavioral_metrics(df_res: pd.DataFrame, position_col='position', time_col='step', equity_col='equity'):
    """
    Calculates detailed behavioral metrics from trade history.
    
    Args:
        df_res (pd.DataFrame): DataFrame containing at least position, equity history.
        position_col (str): Column name for position size/direction.
        
    Returns:
        dict: Dictionary of metrics.
    """
    metrics = {}
    
    # Ensure boolean holding status
    # Assuming position != 0 means holding
    df_res['is_holding'] = df_res[position_col] != 0
    df_res['long_pos'] = df_res[position_col] > 0
    df_res['short_pos'] = df_res[position_col] < 0
    
    # 1. Holding Period Analysis
    # Identify groups of consecutive holding
    # Use shift to find where status changes
    # cumsum identifies unique groups
    df_res['hold_group'] = (df_res['is_holding'] != df_res['is_holding'].shift()).cumsum()
    
    # Filter groups where is_holding is True
    holding_groups = df_res[df_res['is_holding']].groupby('hold_group')
    
    if len(holding_groups) == 0:
        metrics['avg_holding_time_steps'] = 0
        metrics['max_holding_time_steps'] = 0
        metrics['total_trades_approx'] = 0
    else:
        holding_times = holding_groups.size()
        metrics['avg_holding_time_steps'] = float(holding_times.mean())
        metrics['min_holding_time_steps'] = float(holding_times.min())
        metrics['max_holding_time_steps'] = float(holding_times.max())
        metrics['total_trades_approx'] = len(holding_groups)
        
    # 2. Exposure Analysis
    metrics['market_exposure_pct'] = df_res['is_holding'].mean() * 100
    metrics['long_exposure_pct'] = df_res['long_pos'].mean() * 100
    metrics['short_exposure_pct'] = df_res['short_pos'].mean() * 100
    
    # 3. PnL per "Trade" (Approximate)
    # We calculate PnL change per group
    # Note: Precise trade list needs trade log, this is equity-based approximation
    trade_pnls = []
    if len(holding_groups) > 0:
        for _, group in holding_groups:
            start_equity = group[equity_col].iloc[0]
            end_equity = group[equity_col].iloc[-1]
            trade_pnls.append(end_equity - start_equity)
            
        metrics['avg_pnl_per_trade'] = np.mean(trade_pnls)
        metrics['win_rate_pct'] = (np.sum(np.array(trade_pnls) > 0) / len(trade_pnls)) * 100
        metrics['best_trade'] = np.max(trade_pnls)
        metrics['worst_trade'] = np.min(trade_pnls)
        
        gains = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        
        metrics['avg_win'] = np.mean(gains) if gains else 0
        metrics['avg_loss'] = np.mean(losses) if losses else 0
        metrics['profit_factor'] = abs(np.sum(gains) / np.sum(losses)) if losses and sum(losses) != 0 else float('inf')
        
    else:
        metrics['win_rate_pct'] = 0
        metrics['profit_factor'] = 0
        
    return metrics

def generate_text_report(metrics):
    report =  "---------------------------------------------------\n"
    report += "🔍 MODEL BEHAVIORAL DIAGNOSTICS\n"
    report += "---------------------------------------------------\n"
    report += f"Total 'Trades' Detected : {metrics.get('total_trades_approx', 0)}\n"
    report += f"Win Rate                : {metrics.get('win_rate_pct', 0):.2f}%\n"
    report += f"Profit Factor           : {metrics.get('profit_factor', 0):.2f}\n"
    report += "\n"
    report += f"Avg Holding Time (Steps): {metrics.get('avg_holding_time_steps', 0):.2f}\n"
    report += f"Max Holding Time (Steps): {metrics.get('max_holding_time_steps', 0):.0f}\n"
    report += f"Market Exposure         : {metrics.get('market_exposure_pct', 0):.2f}%\n"
    report += "\n"
    report += f"Avg PnL per Trade       : ${metrics.get('avg_pnl_per_trade', 0):.2f}\n"
    report += f"Best Trade              : ${metrics.get('best_trade', 0):.2f}\n"
    report += f"Worst Trade             : ${metrics.get('worst_trade', 0):.2f}\n"
    report += "---------------------------------------------------\n"
    return report
