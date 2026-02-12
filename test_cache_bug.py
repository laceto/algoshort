"""
Deep dive into the cache staleness bug - this is CRITICAL
"""

import pandas as pd
import numpy as np
from algoshort.stop_loss import StopLossCalculator

def test_cache_bug_detailed():
    """
    The cache bug: ATR values are IDENTICAL for different data!
    This is a production-breaking bug.
    """
    print("\n=== CACHE BUG: CRITICAL VULNERABILITY ===")

    df1 = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [100, 101, 102, 103, 104],
        'signal': [1, 1, 1, 1, 1]
    })

    df2 = pd.DataFrame({
        'open': [1000, 1010, 1020, 1030, 1040],
        'high': [1050, 1060, 1070, 1080, 1090],
        'low': [950, 960, 970, 980, 990],
        'close': [1000, 1010, 1020, 1030, 1040],
        'signal': [1, 1, 1, 1, 1]
    })

    print("\nDataset 1: Prices around 100")
    print("Dataset 2: Prices around 1000 (10x higher)")

    calc = StopLossCalculator(df1)
    atr1 = calc._atr(14)
    stop1 = calc.atr_stop_loss('signal', window=14, multiplier=2.0)

    print(f"\nDataset 1 ATR: {atr1.tolist()}")
    print(f"Dataset 1 Stops: {stop1['signal_stop_loss'].tolist()}")

    # Now update to df2
    calc.data = df2
    atr2 = calc._atr(14)
    stop2 = calc.atr_stop_loss('signal', window=14, multiplier=2.0)

    print(f"\nDataset 2 ATR: {atr2.tolist()}")
    print(f"Dataset 2 Stops: {stop2['signal_stop_loss'].tolist()}")

    # Check cache contents
    print(f"\nCache keys: {list(calc._cache.keys())}")

    if atr1.tolist() == atr2.tolist():
        print("\n🚨 CRITICAL BUG CONFIRMED:")
        print("   ATR is IDENTICAL despite 10x price difference!")
        print("   Cache was NOT cleared when data was updated.")
        print("   This will cause MASSIVE LOSSES in production!")
    else:
        print("\n✓ Cache properly cleared")

def test_why_cache_breaks():
    """
    Understanding WHY the cache breaks
    """
    print("\n=== WHY CACHE BREAKS ===")

    df1 = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })

    calc = StopLossCalculator(df1)

    # Call _atr to populate cache
    atr1 = calc._atr(14)
    print(f"Initial ATR: {atr1.tolist()}")
    print(f"Cache keys after first call: {list(calc._cache.keys())}")
    print(f"Cache values: {[v.tolist() for v in calc._cache.values()]}")

    # Now update data
    df2 = pd.DataFrame({
        'open': [200, 201, 202],
        'high': [205, 206, 207],
        'low': [195, 196, 197],
        'close': [200, 201, 202],
        'signal': [1, 0, -1]
    })

    print("\n--- Updating data to df2 ---")
    calc.data = df2
    print(f"Cache keys after data update: {list(calc._cache.keys())}")
    print("Cache SHOULD be empty now (cleared in setter)")

    # Call _atr again
    atr2 = calc._atr(14)
    print(f"\nNew ATR: {atr2.tolist()}")
    print(f"Cache keys after second call: {list(calc._cache.keys())}")

    if list(calc._cache.keys()) == []:
        print("\n✗ Cache was cleared but that's good")
    else:
        print("\n✓ Cache repopulated with new data")

def test_cache_key_collision():
    """
    Test if cache keys could collide across different data
    """
    print("\n=== CACHE KEY COLLISION TEST ===")

    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })

    calc = StopLossCalculator(df)

    # Calculate ATR with different windows
    atr_14 = calc._atr(14)
    atr_20 = calc._atr(20)
    atr_14_again = calc._atr(14)

    print(f"Cache keys: {list(calc._cache.keys())}")
    print("Expected: ['ATR_14', 'ATR_20']")

    # Check if atr_14 and atr_14_again are the same object (cached)
    if atr_14 is atr_14_again:
        print("\n✓ Cache hit successful (same object returned)")
    else:
        print("\n✗ Cache miss (different objects)")

    # Check if values are equal
    if atr_14.equals(atr_14_again):
        print("✓ Values are equal")
    else:
        print("✗ Values differ!")

if __name__ == "__main__":
    test_cache_bug_detailed()
    test_why_cache_breaks()
    test_cache_key_collision()
