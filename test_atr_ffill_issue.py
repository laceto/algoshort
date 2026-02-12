"""
Test the atr_stop_loss ffill behavior to understand its trading implications
"""

import pandas as pd
import numpy as np
from algoshort.stop_loss import StopLossCalculator

def test_atr_ffill_behavior():
    """
    Test what happens when signal changes from long to flat to short.
    Does ffill leak stop loss values across position changes?
    """
    print("\n=== TEST: ATR ffill Leakage Across Position Changes ===")

    # Scenario: Long position -> flat -> short position
    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106],
        'high': [105, 106, 107, 108, 109, 110, 111],
        'low': [95, 96, 97, 98, 99, 100, 101],
        'close': [100, 101, 102, 103, 104, 105, 106],
        'signal': [1, 1, 0, 0, -1, -1, -1]  # Long -> Flat -> Short
    })

    calc = StopLossCalculator(df)
    result = calc.atr_stop_loss('signal', window=3, multiplier=2.0)

    print("\nDataFrame with stops:")
    print(result[['close', 'signal', 'signal_stop_loss']])

    print("\n⚠️  CRITICAL ISSUE ANALYSIS:")
    print("Row 2-3: signal=0 (flat), but stop_loss is forward-filled from long position")
    print("Row 4: signal=-1 (short), gets its own stop calculated THEN ffilled")
    print("\nProblem: When flat (signal=0), old stop loss persists via ffill!")
    print("This could trigger false exits if you're checking stops when not in position.")

def test_atr_no_ffill_comparison():
    """Show what it would look like WITHOUT ffill"""
    print("\n=== TEST: ATR without ffill (for comparison) ===")

    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106],
        'high': [105, 106, 107, 108, 109, 110, 111],
        'low': [95, 96, 97, 98, 99, 100, 101],
        'close': [100, 101, 102, 103, 104, 105, 106],
        'signal': [1, 1, 0, 0, -1, -1, -1]
    })

    calc = StopLossCalculator(df)
    # Manually calculate without the ffill
    atr = calc._atr(window=3)
    price = df['close']
    stop_distance = atr * 2.0

    long_stop = price - stop_distance
    short_stop = price + stop_distance

    # Apply only where signal exists (no ffill)
    df['stop_no_ffill'] = np.nan
    df.loc[df['signal'] > 0, 'stop_no_ffill'] = long_stop
    df.loc[df['signal'] < 0, 'stop_no_ffill'] = short_stop

    # With ffill (actual implementation)
    result = calc.atr_stop_loss('signal')
    df['stop_with_ffill'] = result['signal_stop_loss']

    print("\nComparison:")
    print(df[['close', 'signal', 'stop_no_ffill', 'stop_with_ffill']])
    print("\nNote: ffill carries forward values when signal=0")

def test_position_change_immediately():
    """What if position flips immediately from long to short?"""
    print("\n=== TEST: Immediate Position Flip ===")

    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'signal': [1, -1, 1]  # Long -> Short -> Long (immediate flips)
    })

    calc = StopLossCalculator(df)
    result = calc.atr_stop_loss('signal', window=2, multiplier=2.0)

    print("\nDataFrame:")
    print(result[['close', 'signal', 'signal_stop_loss']])
    print("\nNote: Each row gets its own stop, then ffill occurs")
    print("Since signal changes every row, ffill has less impact")

if __name__ == "__main__":
    test_atr_ffill_behavior()
    test_atr_no_ffill_comparison()
    test_position_change_immediately()
