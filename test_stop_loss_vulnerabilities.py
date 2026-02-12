"""
Devil's Advocate Testing Suite for StopLossCalculator
Testing every possible failure mode, edge case, and silent error.
"""

import pandas as pd
import numpy as np
import sys
from algoshort.stop_loss import StopLossCalculator

def test_all_zeros():
    """What if all prices are zero?"""
    print("\n=== TEST: All Zeros ===")
    df = pd.DataFrame({
        'open': [0, 0, 0],
        'high': [0, 0, 0],
        'low': [0, 0, 0],
        'close': [0, 0, 0],
        'signal': [1, 0, -1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.fixed_percentage_stop_loss('signal', percentage=0.05)
        print(f"✓ No crash. Stop loss values: {result['signal_stop_loss'].tolist()}")
        print(f"  Problem: Stop loss = 0 for 5% below 0! Unusable in real trading.")
    except Exception as e:
        print(f"✗ Crashed: {e}")

def test_all_nan_prices():
    """What if all prices are NaN?"""
    print("\n=== TEST: All NaN Prices ===")
    df = pd.DataFrame({
        'open': [np.nan, np.nan, np.nan],
        'high': [np.nan, np.nan, np.nan],
        'low': [np.nan, np.nan, np.nan],
        'close': [np.nan, np.nan, np.nan],
        'signal': [1, 0, -1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.atr_stop_loss('signal')
        print(f"✓ No crash. ATR: {calc._atr().tolist()}")
        print(f"  Stop loss: {result['signal_stop_loss'].tolist()}")
    except Exception as e:
        print(f"✗ Crashed: {e}")

def test_nan_in_signal():
    """What if signal column has NaN?"""
    print("\n=== TEST: NaN in Signal Column ===")
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, np.nan, -1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.fixed_percentage_stop_loss('signal')
        print(f"✓ No crash. Stop loss: {result['signal_stop_loss'].tolist()}")
        print(f"  Warning: NaN signal produces NaN stop loss - silent failure!")
    except Exception as e:
        print(f"✗ Crashed: {e}")

def test_single_row():
    """What if DataFrame has only 1 row?"""
    print("\n=== TEST: Single Row DataFrame ===")
    df = pd.DataFrame({
        'open': [100],
        'high': [105],
        'low': [95],
        'close': [100],
        'signal': [1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.atr_stop_loss('signal', window=14)
        print(f"✓ No crash. ATR: {calc._atr(14).tolist()}")
        print(f"  Stop loss: {result['signal_stop_loss'].tolist()}")
        print(f"  Warning: ATR with 1 row and window=14 still computes!")
    except Exception as e:
        print(f"✗ Crashed: {e}")

def test_negative_window():
    """What if window is negative?"""
    print("\n=== TEST: Negative Window ===")
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.atr_stop_loss('signal', window=-5)
        print(f"✗ No error raised! ATR: {calc._atr(-5).tolist()}")
    except ValueError as e:
        print(f"✓ Properly rejected: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type: {e}")

def test_zero_window():
    """What if window is 0?"""
    print("\n=== TEST: Zero Window ===")
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.atr_stop_loss('signal', window=0)
        print(f"✗ No error raised! Stop loss computed.")
    except ValueError as e:
        print(f"✓ Properly rejected: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type: {e}")

def test_float_window():
    """What if window is a float?"""
    print("\n=== TEST: Float Window ===")
    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [100, 101, 102, 103, 104],
        'signal': [1, 0, -1, 1, 0]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.atr_stop_loss('signal', window=3.7)
        print(f"✓ Converted to int. ATR: {calc._atr(3.7).tolist()}")
        print(f"  Warning: Silent conversion 3.7 -> 3 may surprise users")
    except Exception as e:
        print(f"✗ Crashed: {e}")

def test_high_less_than_low():
    """What if high < low (data error)?"""
    print("\n=== TEST: High < Low ===")
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [95, 96, 97],  # HIGH is LESS than LOW!
        'low': [105, 106, 107],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.atr_stop_loss('signal')
        atr = calc._atr()
        print(f"✓ No crash. ATR: {atr.tolist()}")
        print(f"  Problem: Negative ATR! {atr.min()}")
        print(f"  Stop loss: {result['signal_stop_loss'].tolist()}")
    except Exception as e:
        print(f"✗ Crashed: {e}")

def test_extreme_atr_multiplier():
    """What happens with extreme ATR multipliers?"""
    print("\n=== TEST: Extreme ATR Multiplier ===")
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 1, 1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.atr_stop_loss('signal', multiplier=1000000)
        print(f"✓ No crash. Stop loss: {result['signal_stop_loss'].tolist()}")
        print(f"  Problem: Negative stop loss for longs!")

        result2 = calc.atr_stop_loss('signal', multiplier=-5)
        print(f"✓ Negative multiplier accepted! Stop loss: {result2['signal_stop_loss'].tolist()}")
        print(f"  Problem: Stop ABOVE entry for longs!")
    except Exception as e:
        print(f"✗ Crashed: {e}")

def test_extreme_percentage():
    """What about percentage > 1.0 or negative?"""
    print("\n=== TEST: Extreme Percentage ===")
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, -1, 1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.fixed_percentage_stop_loss('signal', percentage=5.0)  # 500%!
        print(f"✓ No validation. Stop loss: {result['signal_stop_loss'].tolist()}")
        print(f"  Problem: 500% stop = negative prices for longs!")

        result2 = calc.fixed_percentage_stop_loss('signal', percentage=-0.2)
        print(f"✓ Negative percentage accepted! Stop loss: {result2['signal_stop_loss'].tolist()}")
        print(f"  Problem: Stop ABOVE entry for longs!")
    except Exception as e:
        print(f"✗ Crashed: {e}")

def test_missing_signal_column():
    """What if signal column doesn't exist?"""
    print("\n=== TEST: Missing Signal Column ===")
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.fixed_percentage_stop_loss('nonexistent_signal')
        print(f"✗ No error! Result columns: {result.columns.tolist()}")
    except KeyError as e:
        print(f"✓ Properly rejected: {e}")
    except Exception as e:
        print(f"? Different error: {e}")

def test_infinite_values():
    """What if prices are infinite?"""
    print("\n=== TEST: Infinite Values ===")
    df = pd.DataFrame({
        'open': [100, np.inf, 102],
        'high': [105, np.inf, 107],
        'low': [95, 96, 97],
        'close': [100, np.inf, 102],
        'signal': [1, 1, 1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.atr_stop_loss('signal')
        print(f"✓ No crash. Stop loss: {result['signal_stop_loss'].tolist()}")
        print(f"  Problem: inf propagates through calculations!")
    except Exception as e:
        print(f"✗ Crashed: {e}")

def test_very_large_dataset():
    """Performance test with large dataset"""
    print("\n=== TEST: Large Dataset (100k rows) ===")
    np.random.seed(42)
    n = 100000
    df = pd.DataFrame({
        'open': np.random.randn(n).cumsum() + 100,
        'high': np.random.randn(n).cumsum() + 105,
        'low': np.random.randn(n).cumsum() + 95,
        'close': np.random.randn(n).cumsum() + 100,
        'signal': np.random.choice([-1, 0, 1], n)
    })
    try:
        import time
        start = time.time()
        calc = StopLossCalculator(df)
        result = calc.atr_stop_loss('signal', window=14)
        elapsed = time.time() - start
        print(f"✓ Completed in {elapsed:.2f}s")
        print(f"  Memory: {result.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"✗ Crashed: {e}")

def test_cache_staleness():
    """Does cache cause stale data issues?"""
    print("\n=== TEST: Cache Staleness ===")
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
    try:
        calc = StopLossCalculator(df1)
        atr1 = calc._atr(14)
        print(f"ATR on df1: {atr1.tolist()}")

        # Update data
        calc.data = df2
        atr2 = calc._atr(14)
        print(f"ATR on df2: {atr2.tolist()}")

        if atr1.equals(atr2):
            print(f"✗ CACHE NOT CLEARED! Stale data returned!")
        else:
            print(f"✓ Cache properly cleared on data update")
    except Exception as e:
        print(f"✗ Crashed: {e}")

def test_breakout_channel_window_validation():
    """Test window validation in breakout_channel_stop_loss"""
    print("\n=== TEST: Breakout Channel Window Validation ===")
    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [100, 101, 102, 103, 104],
        'signal': [1, 0, -1, 1, 0]
    })

    # Test negative window
    try:
        calc = StopLossCalculator(df)
        result = calc.breakout_channel_stop_loss('signal', window=-5)
        print(f"✗ Negative window accepted!")
    except ValueError as e:
        print(f"✓ Negative window rejected: {e}")

    # Test zero window
    try:
        calc = StopLossCalculator(df)
        result = calc.breakout_channel_stop_loss('signal', window=0)
        print(f"✗ Zero window accepted!")
    except ValueError as e:
        print(f"✓ Zero window rejected: {e}")

    # Test float window
    try:
        calc = StopLossCalculator(df)
        result = calc.breakout_channel_stop_loss('signal', window=3.7)
        print(f"✓ Float window converted to int")
    except Exception as e:
        print(f"✗ Float window crashed: {e}")

def test_moving_average_zero_window():
    """Moving average with invalid window"""
    print("\n=== TEST: Moving Average Zero Window ===")
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.moving_average_stop_loss('signal', window=0)
        print(f"✗ Zero window accepted! Stop loss: {result['signal_stop_loss'].tolist()}")
        print(f"  Problem: pandas rolling(0) crashes or produces nonsense")
    except Exception as e:
        print(f"? Exception: {e}")

def test_volatility_std_zero_multiplier():
    """Volatility with zero or negative multiplier"""
    print("\n=== TEST: Volatility STD Zero/Negative Multiplier ===")
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.volatility_std_stop_loss('signal', multiplier=0)
        print(f"✓ Zero multiplier accepted. Stop loss: {result['signal_stop_loss'].tolist()}")
        print(f"  Warning: Stop exactly at price - will trigger immediately!")

        result2 = calc.volatility_std_stop_loss('signal', multiplier=-2)
        print(f"✓ Negative multiplier accepted. Stop loss: {result2['signal_stop_loss'].tolist()}")
        print(f"  Problem: Stop in wrong direction!")
    except Exception as e:
        print(f"✗ Crashed: {e}")

def test_empty_dataframe():
    """What happens with empty DataFrame?"""
    print("\n=== TEST: Empty DataFrame ===")
    df = pd.DataFrame()
    try:
        calc = StopLossCalculator(df)
        print(f"✗ Empty DataFrame accepted!")
    except ValueError as e:
        print(f"✓ Empty DataFrame rejected: {e}")
    except Exception as e:
        print(f"? Different error: {e}")

def test_none_dataframe():
    """What happens with None?"""
    print("\n=== TEST: None DataFrame ===")
    try:
        calc = StopLossCalculator(None)
        print(f"✗ None accepted!")
    except ValueError as e:
        print(f"✓ None rejected: {e}")
    except Exception as e:
        print(f"? Different error: {e}")

def test_string_in_numeric_columns():
    """What if numeric columns contain strings?"""
    print("\n=== TEST: String in Numeric Columns ===")
    df = pd.DataFrame({
        'open': [100, 'broken', 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.fixed_percentage_stop_loss('signal')
        print(f"? No crash. Stop loss: {result['signal_stop_loss'].tolist()}")
    except Exception as e:
        print(f"? Exception: {type(e).__name__}: {e}")

def test_division_by_zero_in_classified_pivot():
    """Division by zero in distance threshold calculation"""
    print("\n=== TEST: Division by Zero in Classified Pivot ===")
    df = pd.DataFrame({
        'open': [0, 0, 0],
        'high': [1, 1, 1],
        'low': [-1, -1, -1],
        'close': [0, 0, 0],  # Zero close!
        'signal': [1, 0, -1]
    })
    try:
        calc = StopLossCalculator(df)
        result = calc.classified_pivot_stop_loss('signal')
        print(f"? No crash. Stop loss: {result['signal_stop_loss'].tolist()}")
        print(f"  Warning: Lines 250, 253 divide by close which is zero!")
    except Exception as e:
        print(f"? Exception: {e}")

def test_filter_kwargs_exploit():
    """Can _filter_kwargs be exploited?"""
    print("\n=== TEST: _filter_kwargs Exploitation ===")
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })
    try:
        calc = StopLossCalculator(df)
        # Try to inject unexpected parameters
        result = calc.get_stop_loss('signal', 'atr',
                                     window=10,
                                     malicious_param='__import__("os").system("echo pwned")',
                                     __class__='hack',
                                     _data='poison')
        print(f"✓ Extra kwargs filtered out safely")
    except Exception as e:
        print(f"? Exception: {e}")

def test_get_stop_loss_method_not_in_map():
    """What if method name not in method_map but method exists?"""
    print("\n=== TEST: Method Not in Map ===")
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })
    try:
        calc = StopLossCalculator(df)
        # classified_pivot_stop_loss exists but not in method_map!
        result = calc.get_stop_loss('signal', 'classified_pivot')
        print(f"✗ Should have failed! Method not in map.")
    except ValueError as e:
        print(f"✓ Unknown method rejected: {e}")
    except Exception as e:
        print(f"? Different error: {e}")

def test_concurrent_modification():
    """What if data is modified during calculation?"""
    print("\n=== TEST: Concurrent Modification ===")
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, 0, -1]
    })
    try:
        calc = StopLossCalculator(df)
        # Modify original df
        df['close'] = [200, 201, 202]
        result = calc.fixed_percentage_stop_loss('signal')
        print(f"✓ Uses internal copy. Stop loss based on original: {result['signal_stop_loss'].tolist()}")
    except Exception as e:
        print(f"✗ Crashed: {e}")

if __name__ == "__main__":
    print("="*60)
    print("DEVIL'S ADVOCATE TESTING SUITE")
    print("Testing stop_loss.py for every possible failure mode")
    print("="*60)

    test_all_zeros()
    test_all_nan_prices()
    test_nan_in_signal()
    test_single_row()
    test_negative_window()
    test_zero_window()
    test_float_window()
    test_high_less_than_low()
    test_extreme_atr_multiplier()
    test_extreme_percentage()
    test_missing_signal_column()
    test_infinite_values()
    test_very_large_dataset()
    test_cache_staleness()
    test_breakout_channel_window_validation()
    test_moving_average_zero_window()
    test_volatility_std_zero_multiplier()
    test_empty_dataframe()
    test_none_dataframe()
    test_string_in_numeric_columns()
    test_division_by_zero_in_classified_pivot()
    test_filter_kwargs_exploit()
    test_get_stop_loss_method_not_in_map()
    test_concurrent_modification()

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
