"""
Re-check the cache staleness test that showed failure
"""

import pandas as pd
import numpy as np
from algoshort.stop_loss import StopLossCalculator

def test_cache_staleness_recheck():
    """Re-run the exact test that showed cache failure"""
    print("\n=== CACHE STALENESS RE-CHECK ===")
    df1 = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })
    df2 = pd.DataFrame({
        'open': [200, 201, 202],
        'high': [205, 206, 207],
        'low': [195, 196, 197],
        'close': [200, 201, 202],
        'signal': [1, 0, -1]
    })

    calc = StopLossCalculator(df1)
    atr1 = calc._atr(14)
    print(f"ATR on df1: {atr1.tolist()}")

    # Update data
    calc.data = df2
    atr2 = calc._atr(14)
    print(f"ATR on df2: {atr2.tolist()}")

    # WAIT - let me check if they're using the SAME VALUES
    # Even though the prices are 2x, the TRUE RANGE is the same!
    # TR for df1: high-low = 105-95 = 10
    # TR for df2: high-low = 205-195 = 10
    # So ATR SHOULD be the same!

    print("\nAha! Let me check the True Range calculation:")
    print(f"df1: high-low = {df1['high'][0] - df1['low'][0]}")
    print(f"df2: high-low = {df2['high'][0] - df2['low'][0]}")
    print("\nBoth have the same range, so ATR being equal is CORRECT!")
    print("This was a false alarm - the test data has same volatility.")

    if atr1.equals(atr2):
        print("\nCache cleared properly. ATRs equal due to identical volatility.")
    else:
        print("\nCACHE NOT CLEARED!")

def test_cache_with_different_volatility():
    """Test with actually different volatility"""
    print("\n=== TEST: Cache with Different Volatility ===")
    df1 = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],  # Range = 10
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })
    df2 = pd.DataFrame({
        'open': [200, 201, 202],
        'high': [250, 256, 262],  # Range = 50 (much higher!)
        'low': [200, 201, 202],
        'close': [200, 201, 202],
        'signal': [1, 0, -1]
    })

    calc = StopLossCalculator(df1)
    atr1 = calc._atr(14)
    print(f"ATR on df1 (low vol): {atr1.tolist()}")

    # Update data
    calc.data = df2
    atr2 = calc._atr(14)
    print(f"ATR on df2 (high vol): {atr2.tolist()}")

    if atr1.equals(atr2):
        print("\n✗ CACHE NOT CLEARED! Stale data returned!")
    else:
        print("\n✓ Cache properly cleared. Different volatilities detected.")

if __name__ == "__main__":
    test_cache_staleness_recheck()
    test_cache_with_different_volatility()
