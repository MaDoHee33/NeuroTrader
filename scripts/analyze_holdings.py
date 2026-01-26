
import pandas as pd

def analyze_holdings():
    try:
        df = pd.read_csv('analysis/backtest_short_term_data.csv')
        
        # Calculate holding (Position > 0)
        df['is_holding'] = df['position'] > 0
        
        # Group consecutive True values
        # We identify groups where the 'is_holding' status changes
        # cumsum() increments every time is_holding changes value
        df['group'] = (df['is_holding'] != df['is_holding'].shift()).cumsum()
        
        # Filter only for holding periods
        holdings = df[df['is_holding']]
        
        if len(holdings) == 0:
            print("No holding periods found.")
            return

        # Count size of each group (number of steps)
        holding_periods_steps = holdings.groupby('group').size()
        
        # Convert to minutes (1 step = 5 mins)
        min_hold = holding_periods_steps.min() * 5
        max_hold = holding_periods_steps.max() * 5
        avg_hold = holding_periods_steps.mean() * 5
        
        print(f"Shortest Holding: {min_hold} mins")
        print(f"Longest Holding: {max_hold} mins ({max_hold/60:.2f} hours)")
        print(f"Average Holding: {avg_hold:.2f} mins")
        print(f"Total Discrete Holding Periods: {len(holding_periods_steps)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_holdings()
