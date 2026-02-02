from algoshort.combiner import HybridSignalCombiner
import pandas as pd
import numpy as np

# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================

def test_long_short_logic():
    """
    Test all long/short scenarios to ensure correct behavior.
    """
    print("="*70)
    print("TESTING LONG/SHORT LOGIC")
    print("="*70)
    
    # Test Case 1: Simple Long Entry and Exit
    print("\n1. SIMPLE LONG ENTRY & EXIT")
    print("-" * 70)
    df1 = pd.DataFrame({
        'floor_ceiling': [0, 1, 1, 1, 1, -1, -1],
        'range_breakout': [0, 1, 0, 0, 0, 0, 0],
        'ma_crossover': [0, 0, 0, 0, -1, 0, 0]
    })
    
    combiner = HybridSignalCombiner(verbose=True)
    df1 = combiner.combine_signals(df1, allow_flips=True, require_regime_alignment=True)
    print(df1[['floor_ceiling', 'range_breakout', 'ma_crossover', 'hybrid_signal']])
    print(f"Expected: [0, 1, 1, 1, 0, 0, 0]")
    print(f"Actual:   {df1['hybrid_signal'].tolist()}")
    
    # Test Case 2: Simple Short Entry and Exit
    print("\n2. SIMPLE SHORT ENTRY & EXIT")
    print("-" * 70)
    df2 = pd.DataFrame({
        'floor_ceiling': [0, -1, -1, -1, -1, 1, 1],
        'range_breakout': [0, -1, 0, 0, 0, 0, 0],
        'ma_crossover': [0, 0, 0, 0, 1, 0, 0]
    })
    
    combiner = HybridSignalCombiner(verbose=True)
    df2 = combiner.combine_signals(df2, allow_flips=True, require_regime_alignment=True)
    print(df2[['floor_ceiling', 'range_breakout', 'ma_crossover', 'hybrid_signal']])
    print(f"Expected: [0, -1, -1, -1, 0, 0, 0]")
    print(f"Actual:   {df2['hybrid_signal'].tolist()}")
    
    # Test Case 3: Flip from Long to Short
    print("\n3. FLIP LONG → SHORT")
    print("-" * 70)
    df3 = pd.DataFrame({
        'floor_ceiling': [0, 1, 1, 1, -1, -1, -1],
        'range_breakout': [0, 1, 0, 0, -1, 0, 0],
        'ma_crossover': [0, 0, 0, 0, 0, 0, 0]
    })
    
    combiner = HybridSignalCombiner(verbose=True)
    df3 = combiner.combine_signals(df3, allow_flips=True, require_regime_alignment=True)
    print(df3[['floor_ceiling', 'range_breakout', 'ma_crossover', 'hybrid_signal']])
    print(f"Expected: [0, 1, 1, 1, -1, -1, -1]")
    print(f"Actual:   {df3['hybrid_signal'].tolist()}")
    
    # Test Case 4: Flip from Short to Long
    print("\n4. FLIP SHORT → LONG")
    print("-" * 70)
    df4 = pd.DataFrame({
        'floor_ceiling': [0, -1, -1, -1, 1, 1, 1],
        'range_breakout': [0, -1, 0, 0, 1, 0, 0],
        'ma_crossover': [0, 0, 0, 0, 0, 0, 0]
    })
    
    combiner = HybridSignalCombiner(verbose=True)
    df4 = combiner.combine_signals(df4, allow_flips=True, require_regime_alignment=True)
    print(df4[['floor_ceiling', 'range_breakout', 'ma_crossover', 'hybrid_signal']])
    print(f"Expected: [0, -1, -1, -1, 1, 1, 1]")
    print(f"Actual:   {df4['hybrid_signal'].tolist()}")
    
    # Test Case 5: No Flips Allowed
    print("\n5. NO FLIPS (MUST EXIT FIRST)")
    print("-" * 70)
    df5 = pd.DataFrame({
        'floor_ceiling': [0, 1, 1, 1, -1, -1, -1],
        'range_breakout': [0, 1, 0, 0, -1, 0, 0],
        'ma_crossover': [0, 0, 0, 0, 0, 0, 0]
    })
    
    combiner = HybridSignalCombiner(verbose=True)
    df5 = combiner.combine_signals(df5, allow_flips=False, require_regime_alignment=True)
    print(df5[['floor_ceiling', 'range_breakout', 'ma_crossover', 'hybrid_signal']])
    print(f"Expected: [0, 1, 1, 1, 0, 0, 0] (exits but doesn't flip)")
    print(f"Actual:   {df5['hybrid_signal'].tolist()}")
    
    # Test Case 6: Exit on MA Crossover (Long)
    print("\n6. EXIT LONG ON MA CROSSOVER")
    print("-" * 70)
    df6 = pd.DataFrame({
        'floor_ceiling': [0, 1, 1, 1, 1, 1, 0],
        'range_breakout': [0, 1, 0, 0, 0, 0, 0],
        'ma_crossover': [0, 0, 0, -1, 0, 0, 0]
    })
    
    combiner = HybridSignalCombiner(verbose=True)
    df6 = combiner.combine_signals(df6, allow_flips=True, require_regime_alignment=True)
    print(df6[['floor_ceiling', 'range_breakout', 'ma_crossover', 'hybrid_signal']])
    print(f"Expected: [0, 1, 1, 0, 0, 0, 0] (exits on MA cross)")
    print(f"Actual:   {df6['hybrid_signal'].tolist()}")
    
    # Test Case 7: Exit on MA Crossover (Short)
    print("\n7. EXIT SHORT ON MA CROSSOVER")
    print("-" * 70)
    df7 = pd.DataFrame({
        'floor_ceiling': [0, -1, -1, -1, -1, -1, 0],
        'range_breakout': [0, -1, 0, 0, 0, 0, 0],
        'ma_crossover': [0, 0, 0, 1, 0, 0, 0]
    })
    
    combiner = HybridSignalCombiner(verbose=True)
    df7 = combiner.combine_signals(df7, allow_flips=True, require_regime_alignment=True)
    print(df7[['floor_ceiling', 'range_breakout', 'ma_crossover', 'hybrid_signal']])
    print(f"Expected: [0, -1, -1, 0, 0, 0, 0] (exits on MA cross)")
    print(f"Actual:   {df7['hybrid_signal'].tolist()}")
    
    print("\n" + "="*70)
    print("TESTS COMPLETE")
    print("="*70)


# ============================================================================
# REAL-WORLD USAGE EXAMPLE
# ============================================================================

def create_realistic_example():
    """
    Create a realistic example with random but structured signals.
    """
    np.random.seed(42)
    n = 200
    
    # Simulate realistic signal patterns
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n),
        'close': (np.random.randn(n) * 2).cumsum() + 100,
        'chg1D_fx': np.random.randn(n) * 0.02,
        'stop_loss': None,  # Will calculate
    })
    
    # Floor/ceiling: Slow-changing regime (tends to persist)
    floor_ceiling = np.zeros(n)
    current_regime = 0
    for i in range(n):
        if np.random.random() < 0.05:  # 5% chance of regime change
            current_regime = np.random.choice([-1, 0, 1])
        floor_ceiling[i] = current_regime
    df['floor_ceiling'] = floor_ceiling.astype(int)
    
    # Range breakout: Rare signals (mostly 0)
    df['range_breakout'] = np.random.choice([-1, 0, 1], n, p=[0.05, 0.90, 0.05])
    
    # MA crossover: Occasional signals
    df['ma_crossover'] = np.random.choice([-1, 0, 1], n, p=[0.10, 0.80, 0.10])
    
    # Calculate ATR-based stop loss
    df['high'] = df['close'] * 1.01
    df['low'] = df['close'] * 0.99
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['stop_loss'] = df['close'] - (df['atr'] * 2)
    df['stop_loss'] = df['stop_loss'].fillna(df['close'] * 0.95)
    
    return df


# Run the test suite
if __name__ == "__main__":
    test_long_short_logic()
    
    print("\n" + "="*70)
    print("REALISTIC EXAMPLE")
    print("="*70)
    
    # Create realistic data
    df = create_realistic_example()
    
    # Combine signals
    combiner = HybridSignalCombiner(verbose=False)
    df = combiner.combine_signals(df, allow_flips=True, require_regime_alignment=True)
    
    # Add metadata
    df = combiner.add_signal_metadata(df)
    
    # Get summary
    summary = combiner.get_trade_summary(df)
    
    print("\nTrade Summary:")
    print("-" * 70)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:10.2f}")
        else:
            print(f"{key:30s}: {value:10}")
    
    # Show sample of trades
    print("\nSample Trades:")
    print("-" * 70)
    trade_rows = df[df['trade_type'] != 'hold'].head(20)
    print(trade_rows[['date', 'floor_ceiling', 'range_breakout', 'ma_crossover', 
                      'hybrid_signal', 'trade_type', 'position_direction']])
    
    # Validate long/short balance
    print("\nPosition Balance:")
    print("-" * 70)
    print(f"Long trades:  {summary['entry_long_count']}")
    print(f"Short trades: {summary['entry_short_count']}")
    print(f"Long→Short flips: {summary['flip_long_to_short_count']}")
    print(f"Short→Long flips: {summary['flip_short_to_long_count']}")
    
    # Show that both long and short work
    has_long = (df['hybrid_signal'] == 1).any()
    has_short = (df['hybrid_signal'] == -1).any()
    print(f"\nHas long positions: {has_long}")
    print(f"Has short positions: {has_short}")
    print(f"✓ BOTH LONG AND SHORT VERIFIED!" if (has_long and has_short) else "✗ Missing positions")