
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.brain.features import FeatureRegistry

def generate_sample_data(n_rows=300):
    """Generate mock OHLCV data."""
    dates = [datetime(2025, 1, 1) + timedelta(minutes=5*i) for i in range(n_rows)]
    
    # Random walk price
    price = 1000.0
    data = []
    for d in dates:
        change = np.random.normal(0, 1.0)
        price += change
        high = price + abs(np.random.normal(0, 0.5))
        low = price - abs(np.random.normal(0, 0.5))
        volume = abs(np.random.normal(1000, 200))
        
        data.append({
            'time': d,
            'open': price,
            'high': high,
            'low': low,
            'close': price + np.random.normal(0, 0.2), # Slight move from open
            'volume': volume
        })
        
    return pd.DataFrame(data)

def test_feature_consistency():
    """
    The Golden Rule Test:
    Verify that result from compute_batch(df) matches 
    iterative update_stream(tick) EXACTLY.
    """
    registry = FeatureRegistry()
    
    # 1. Generate Data
    df = generate_sample_data(n_rows=250)
    
    # 2. Batch Calculation
    batch_features = registry.compute_batch(df)
    
    # 3. Stream Calculation
    stream_results = []
    
    # Feed ticks one by one
    print("\nStreaming ticks...")
    for idx, row in df.iterrows():
        tick = row.to_dict()
        feat_vector = registry.update_stream(tick)
        stream_results.append(feat_vector)
    
    stream_matrix = np.array(stream_results)
    
    # 4. Compare Tail
    # Indicators like EMA/RSI need warmup. Compare last 50 points.
    lookback = 50
    
    batch_tail = batch_features.iloc[-lookback:].values
    stream_tail = stream_matrix[-lookback:]
    
    print(f"\nComparing last {lookback} rows...")
    print(f"Batch Shape: {batch_tail.shape}")
    print(f"Stream Shape: {stream_tail.shape}")
    
    # Find mismatch
    diff = np.abs(batch_tail - stream_tail)
    max_diff = np.max(diff)
    
    if max_diff > 1e-4:
        print("\n[MISMATCH] DETAILS:")
        for i, col in enumerate(registry.feature_names):
            col_diff = np.max(diff[:, i])
            if col_diff > 1e-4:
                print(f"   Feature '{col}': Max Diff = {col_diff}")
                print(f"     Batch (last): {batch_tail[-1, i]}")
                print(f"     Stream (last): {stream_tail[-1, i]}")
    
    assert np.allclose(batch_tail, stream_tail, atol=1e-4), \
        f"Mismatch detected! Max diff: {max_diff}"
        
    print("\n✅ GOLDEN RULE PASSED: Batch == Stream")

def test_schema_integrity():
    """Ensure feature list matches expectation."""
    registry = FeatureRegistry()
    cols = registry.feature_names
    
    expected = [
        'rsi', 'macd', 'macd_signal', 'bb_width', 'bb_position',
        'ema_20_dist', 'ema_50_dist', 'atr_norm', 
        'stoch_k', 'stoch_d', 'log_ret',
        'vwap_dist', 'normalized_volume',
        'body_size_norm', 'is_bullish',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]
    
    assert len(cols) == len(expected)
    assert set(cols) == set(expected)
    print(f"\n✅ SCHEMA VIABLE: {len(cols)} features verified.")

if __name__ == "__main__":
    test_feature_consistency()
    test_schema_integrity()
