
import pandas as pd
import numpy as np
from src.brain.feature_eng import add_features

# Create dummy data
dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
df = pd.DataFrame({
    'timestamp': dates,
    'open': np.random.randn(100) + 100,
    'high': np.random.randn(100) + 105,
    'low': np.random.randn(100) + 95,
    'close': np.random.randn(100) + 100,
    'volume': np.random.randint(100, 1000, 100)
    # Note: 'symbol' column usually not needed for indicators
})
df.set_index('timestamp', inplace=True)

print("🧪 Testing Feature Engineering...")
try:
    df_processed = add_features(df)
    
    expected_cols = ['rsi', 'stoch_k', 'stoch_d', 'vwap', 'atr', 'bb_width']
    missing = [c for c in expected_cols if c not in df_processed.columns]
    
    if missing:
        print(f"❌ Missing expected columns: {missing}")
        exit(1)
        
    print(f"✅ Success! Generated columns: {list(df_processed.columns)}")
    print(df_processed[['rsi', 'stoch_k', 'vwap']].tail())

except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
