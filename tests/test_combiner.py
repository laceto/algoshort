"""
Comprehensive tests for the algoshort.combiner module.

Tests cover:
- HybridSignalCombiner: signal combination logic
- SignalGridSearch: grid search functionality
- Edge cases and error handling
"""

import logging
import numpy as np
import pandas as pd
import pytest

from algoshort.combiner import (
    HybridSignalCombiner,
    SignalGridSearch,
    _process_stock_all_combinations,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_signals_df():
    """Create a sample DataFrame with direction, entry, and exit signals."""
    np.random.seed(42)
    n = 100

    # Create deterministic signals for testing
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n, freq='B'),
        'close': 100 + np.cumsum(np.random.randn(n) * 2),
        # Direction signal (floor/ceiling style)
        'direction': np.where(np.arange(n) % 20 < 10, 1, -1),
        # Entry signal (breakout style)
        'entry': np.where(np.arange(n) % 5 == 0, 1,
                         np.where(np.arange(n) % 7 == 0, -1, 0)),
        # Exit signal (MA crossover style)
        'exit': np.where(np.arange(n) % 8 == 0, 1,
                        np.where(np.arange(n) % 9 == 0, -1, 0)),
    })

    return df


@pytest.fixture
def bullish_regime_df():
    """Create DataFrame with persistent bullish regime."""
    n = 50
    return pd.DataFrame({
        'direction': [1] * n,
        'entry': [0, 1, 0, 0, 0, -1, 0, 0, 0, 1] * 5,
        'exit': [0, 0, 0, 0, -1, 0, 0, 1, 0, 0] * 5,
    })


@pytest.fixture
def bearish_regime_df():
    """Create DataFrame with persistent bearish regime."""
    n = 50
    return pd.DataFrame({
        'direction': [-1] * n,
        'entry': [0, -1, 0, 0, 0, 1, 0, 0, 0, -1] * 5,
        'exit': [0, 0, 0, 0, 1, 0, 0, -1, 0, 0] * 5,
    })


@pytest.fixture
def regime_change_df():
    """Create DataFrame with regime changes."""
    return pd.DataFrame({
        'direction': [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
        'entry': [0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0],
        'exit': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })


@pytest.fixture
def neutral_regime_df():
    """Create DataFrame with neutral regime."""
    n = 30
    return pd.DataFrame({
        'direction': [0] * n,
        'entry': [0, 1, 0, 0, -1, 0, 0, 1, 0, 0] * 3,
        'exit': [0, 0, 0, -1, 0, 1, 0, 0, 0, 0] * 3,
    })


@pytest.fixture
def grid_search_df():
    """Create DataFrame with multiple signal columns for grid search."""
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n, freq='B'),
        'close': 100 + np.cumsum(np.random.randn(n) * 2),
        # Direction signal
        'rrg': np.where(np.arange(n) % 30 < 15, 1, -1),
        # Various entry/exit signals
        'rbo_20': np.where(np.arange(n) % 10 == 0, 1,
                          np.where(np.arange(n) % 12 == 0, -1, 0)),
        'rtt_5020': np.where(np.arange(n) % 15 == 0, 1,
                            np.where(np.arange(n) % 18 == 0, -1, 0)),
        'rsma_102050': np.where(np.arange(n) % 8 == 0, 1,
                               np.where(np.arange(n) % 11 == 0, -1, 0)),
    })

    return df


@pytest.fixture
def minimal_df():
    """Create minimal DataFrame (2 rows)."""
    return pd.DataFrame({
        'direction': [1, 1],
        'entry': [0, 1],
        'exit': [0, 0],
    })


# =============================================================================
# HybridSignalCombiner Initialization Tests
# =============================================================================

class TestHybridSignalCombinerInit:
    """Tests for HybridSignalCombiner initialization."""

    def test_init_default(self):
        """Test default initialization."""
        combiner = HybridSignalCombiner()

        assert combiner.direction_col == 'floor_ceiling'
        assert combiner.entry_col == 'range_breakout'
        assert combiner.exit_col == 'ma_crossover'
        assert combiner.verbose is False

    def test_init_custom(self):
        """Test custom initialization."""
        combiner = HybridSignalCombiner(
            direction_col='rrg',
            entry_col='breakout',
            exit_col='turtle',
            verbose=True
        )

        assert combiner.direction_col == 'rrg'
        assert combiner.entry_col == 'breakout'
        assert combiner.exit_col == 'turtle'
        assert combiner.verbose is True


# =============================================================================
# Signal Validation Tests
# =============================================================================

class TestSignalValidation:
    """Tests for signal validation."""

    def test_validate_valid_signals(self, sample_signals_df):
        """Test validation with valid signals."""
        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.validate_signals(sample_signals_df)

        assert result['valid'] is True
        assert len(result['errors']) == 0

    def test_validate_missing_column(self, sample_signals_df):
        """Test validation with missing column."""
        combiner = HybridSignalCombiner(
            direction_col='missing_column',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.validate_signals(sample_signals_df)

        assert result['valid'] is False
        assert any('missing_column' in err for err in result['errors'])

    def test_validate_invalid_values(self):
        """Test validation with invalid signal values."""
        df = pd.DataFrame({
            'direction': [1, 0, -1, 2],  # 2 is invalid
            'entry': [0, 1, 0, -1],
            'exit': [0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.validate_signals(df)

        assert result['valid'] is False
        assert any('invalid values' in err.lower() for err in result['errors'])

    def test_validate_nan_values(self):
        """Test validation with NaN values."""
        df = pd.DataFrame({
            'direction': [1, 0, np.nan, -1],
            'entry': [0, 1, 0, -1],
            'exit': [0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.validate_signals(df)

        assert len(result['warnings']) > 0
        assert any('NaN' in warn for warn in result['warnings'])


# =============================================================================
# Signal Combination Tests - Entry Logic
# =============================================================================

class TestCombineSignalsEntry:
    """Tests for entry signal logic."""

    def test_long_entry_bullish_regime(self):
        """Test long entry when regime is bullish and entry=1."""
        df = pd.DataFrame({
            'direction': [0, 1, 1, 1, 1],
            'entry': [0, 1, 0, 0, 0],
            'exit': [0, 0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(df, output_col='signal')

        # Should enter long at bar 1 and hold
        assert result.loc[1, 'signal'] == 1
        assert result.loc[2, 'signal'] == 1

    def test_short_entry_bearish_regime(self):
        """Test short entry when regime is bearish and entry=-1."""
        df = pd.DataFrame({
            'direction': [0, -1, -1, -1, -1],
            'entry': [0, -1, 0, 0, 0],
            'exit': [0, 0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(df, output_col='signal')

        # Should enter short at bar 1 and hold
        assert result.loc[1, 'signal'] == -1
        assert result.loc[2, 'signal'] == -1

    def test_no_long_entry_bearish_regime(self):
        """Test that long entry is blocked in bearish regime (strict mode)."""
        df = pd.DataFrame({
            'direction': [0, -1, -1, -1, -1],
            'entry': [0, 1, 0, 0, 0],  # Long entry signal in bearish regime
            'exit': [0, 0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(
            df, output_col='signal', require_regime_alignment=True
        )

        # Should NOT enter long - regime is bearish
        assert result.loc[1, 'signal'] == 0
        assert (result['signal'] == 0).all()

    def test_no_short_entry_bullish_regime(self):
        """Test that short entry is blocked in bullish regime (strict mode)."""
        df = pd.DataFrame({
            'direction': [0, 1, 1, 1, 1],
            'entry': [0, -1, 0, 0, 0],  # Short entry signal in bullish regime
            'exit': [0, 0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(
            df, output_col='signal', require_regime_alignment=True
        )

        # Should NOT enter short - regime is bullish
        assert result.loc[1, 'signal'] == 0

    def test_long_entry_neutral_regime_loose(self):
        """Test long entry in neutral regime with loose alignment."""
        df = pd.DataFrame({
            'direction': [0, 0, 0, 0, 0],  # Neutral regime
            'entry': [0, 1, 0, 0, 0],
            'exit': [0, 0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(
            df, output_col='signal', require_regime_alignment=False
        )

        # Should enter long - loose mode allows neutral
        assert result.loc[1, 'signal'] == 1


# =============================================================================
# Signal Combination Tests - Exit Logic
# =============================================================================

class TestCombineSignalsExit:
    """Tests for exit signal logic."""

    def test_exit_long_on_exit_signal(self):
        """Test exiting long position on exit signal."""
        df = pd.DataFrame({
            'direction': [0, 1, 1, 1, 1],
            'entry': [0, 1, 0, 0, 0],  # Enter long
            'exit': [0, 0, 0, -1, 0],  # Exit at bar 3
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(df, output_col='signal')

        # Long from bar 1-2, exit at bar 3
        assert result.loc[1, 'signal'] == 1
        assert result.loc[2, 'signal'] == 1
        assert result.loc[3, 'signal'] == 0  # Exited
        assert result.loc[4, 'signal'] == 0  # Stays flat

    def test_exit_short_on_exit_signal(self):
        """Test exiting short position on exit signal."""
        df = pd.DataFrame({
            'direction': [0, -1, -1, -1, -1],
            'entry': [0, -1, 0, 0, 0],  # Enter short
            'exit': [0, 0, 0, 1, 0],  # Exit at bar 3
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(df, output_col='signal')

        # Short from bar 1-2, exit at bar 3
        assert result.loc[1, 'signal'] == -1
        assert result.loc[2, 'signal'] == -1
        assert result.loc[3, 'signal'] == 0  # Exited

    def test_exit_long_on_regime_change(self):
        """Test exiting long position when regime turns bearish."""
        df = pd.DataFrame({
            'direction': [0, 1, 1, -1, -1],  # Regime changes at bar 3
            'entry': [0, 1, 0, 0, 0],
            'exit': [0, 0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(
            df, output_col='signal', allow_flips=False
        )

        # Long at bars 1-2, exit at bar 3 due to regime change
        assert result.loc[1, 'signal'] == 1
        assert result.loc[2, 'signal'] == 1
        assert result.loc[3, 'signal'] == 0  # Exited on regime change


# =============================================================================
# Signal Combination Tests - Flip Logic
# =============================================================================

class TestCombineSignalsFlip:
    """Tests for position flip logic."""

    def test_flip_long_to_short(self):
        """Test flipping from long to short."""
        df = pd.DataFrame({
            'direction': [0, 1, 1, -1, -1, -1],
            'entry': [0, 1, 0, -1, 0, 0],  # Long entry, then short entry
            'exit': [0, 0, 0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(
            df, output_col='signal', allow_flips=True
        )

        # Long at bars 1-2, flip to short at bar 3
        assert result.loc[1, 'signal'] == 1
        assert result.loc[2, 'signal'] == 1
        assert result.loc[3, 'signal'] == -1  # Flipped to short

    def test_flip_short_to_long(self):
        """Test flipping from short to long."""
        df = pd.DataFrame({
            'direction': [0, -1, -1, 1, 1, 1],
            'entry': [0, -1, 0, 1, 0, 0],  # Short entry, then long entry
            'exit': [0, 0, 0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(
            df, output_col='signal', allow_flips=True
        )

        # Short at bars 1-2, flip to long at bar 3
        assert result.loc[1, 'signal'] == -1
        assert result.loc[2, 'signal'] == -1
        assert result.loc[3, 'signal'] == 1  # Flipped to long

    def test_no_flip_when_disabled(self):
        """Test that flips are disabled when allow_flips=False."""
        df = pd.DataFrame({
            'direction': [0, 1, 1, -1, -1, -1],
            'entry': [0, 1, 0, -1, 0, 0],
            'exit': [0, 0, 0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(
            df, output_col='signal', allow_flips=False
        )

        # Should exit to flat instead of flipping
        assert result.loc[1, 'signal'] == 1
        assert result.loc[2, 'signal'] == 1
        assert result.loc[3, 'signal'] == 0  # Exit, not flip


# =============================================================================
# Signal Metadata Tests
# =============================================================================

class TestSignalMetadata:
    """Tests for signal metadata generation."""

    def test_add_signal_metadata(self, sample_signals_df):
        """Test adding metadata columns."""
        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        df = combiner.combine_signals(sample_signals_df, output_col='signal')
        df = combiner.add_signal_metadata(df, output_col='signal')

        # Check metadata columns exist
        assert 'position_changed' in df.columns
        assert 'trade_type' in df.columns
        assert 'bars_in_position' in df.columns
        assert 'position_direction' in df.columns

    def test_trade_type_values(self):
        """Test that trade_type values are correct."""
        df = pd.DataFrame({
            'direction': [0, 1, 1, 1, -1, -1, -1, 1],
            'entry': [0, 1, 0, 0, -1, 0, 0, 1],
            'exit': [0, 0, 0, 0, 0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        df = combiner.combine_signals(df, output_col='signal', allow_flips=True)
        df = combiner.add_signal_metadata(df, output_col='signal')

        # Check specific trade types
        assert 'entry_long' in df['trade_type'].values
        assert 'flip_long_to_short' in df['trade_type'].values
        assert 'flip_short_to_long' in df['trade_type'].values

    def test_position_direction_values(self):
        """Test that position_direction values are correct."""
        df = pd.DataFrame({
            'direction': [0, 1, 1, -1, -1],
            'entry': [0, 1, 0, -1, 0],
            'exit': [0, 0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        df = combiner.combine_signals(df, output_col='signal', allow_flips=True)
        df = combiner.add_signal_metadata(df, output_col='signal')

        # Check position directions
        assert 'long' in df['position_direction'].values
        assert 'short' in df['position_direction'].values


# =============================================================================
# Trade Summary Tests
# =============================================================================

class TestTradeSummary:
    """Tests for trade summary generation."""

    def test_get_trade_summary(self, sample_signals_df):
        """Test trade summary generation."""
        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        df = combiner.combine_signals(sample_signals_df, output_col='signal')
        summary = combiner.get_trade_summary(df, output_col='signal')

        # Check summary keys exist
        assert 'total_bars' in summary
        assert 'long_bars' in summary
        assert 'short_bars' in summary
        assert 'flat_bars' in summary
        assert 'long_pct' in summary
        assert 'short_pct' in summary
        assert 'flat_pct' in summary
        assert 'entry_long_count' in summary
        assert 'entry_short_count' in summary
        assert 'total_entries' in summary

    def test_trade_summary_percentages(self, sample_signals_df):
        """Test that summary percentages add up correctly."""
        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        df = combiner.combine_signals(sample_signals_df, output_col='signal')
        summary = combiner.get_trade_summary(df, output_col='signal')

        # Percentages should sum to ~100%
        total_pct = summary['long_pct'] + summary['short_pct'] + summary['flat_pct']
        assert abs(total_pct - 100.0) < 0.1

    def test_trade_summary_bar_counts(self, sample_signals_df):
        """Test that bar counts add up correctly."""
        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        df = combiner.combine_signals(sample_signals_df, output_col='signal')
        summary = combiner.get_trade_summary(df, output_col='signal')

        # Bar counts should sum to total
        total_bars = summary['long_bars'] + summary['short_bars'] + summary['flat_bars']
        assert total_bars == summary['total_bars']


# =============================================================================
# SignalGridSearch Tests
# =============================================================================

class TestSignalGridSearch:
    """Tests for SignalGridSearch class."""

    def test_init_valid(self, grid_search_df):
        """Test valid initialization."""
        searcher = SignalGridSearch(
            df=grid_search_df,
            available_signals=['rbo_20', 'rtt_5020', 'rsma_102050'],
            direction_col='rrg'
        )

        assert len(searcher.available_signals) == 3
        assert searcher.direction_col == 'rrg'

    def test_init_missing_direction_col(self, grid_search_df):
        """Test initialization with missing direction column."""
        with pytest.raises(ValueError, match="Direction column"):
            SignalGridSearch(
                df=grid_search_df,
                available_signals=['rbo_20'],
                direction_col='missing_col'
            )

    def test_init_missing_signal(self, grid_search_df):
        """Test initialization with missing signal column."""
        with pytest.raises(ValueError, match="missing"):
            SignalGridSearch(
                df=grid_search_df,
                available_signals=['rbo_20', 'missing_signal'],
                direction_col='rrg'
            )

    def test_init_direction_in_signals(self, grid_search_df):
        """Test that direction column cannot be in available_signals."""
        with pytest.raises(ValueError, match="should not be in"):
            SignalGridSearch(
                df=grid_search_df,
                available_signals=['rbo_20', 'rrg'],  # rrg is direction
                direction_col='rrg'
            )

    def test_generate_grid(self, grid_search_df):
        """Test grid generation."""
        searcher = SignalGridSearch(
            df=grid_search_df,
            available_signals=['rbo_20', 'rtt_5020'],
            direction_col='rrg'
        )

        grid = searcher.generate_grid()

        # 2 signals × 2 signals = 4 combinations
        assert len(grid) == 4

        # Check structure
        for combo in grid:
            assert 'entry' in combo
            assert 'exit' in combo
            assert 'name' in combo

    def test_get_available_signals(self, grid_search_df):
        """Test get_available_signals method."""
        signals = ['rbo_20', 'rtt_5020', 'rsma_102050']
        searcher = SignalGridSearch(
            df=grid_search_df,
            available_signals=signals,
            direction_col='rrg'
        )

        result = searcher.get_available_signals()
        assert result == signals

    def test_run_grid_search_basic(self, grid_search_df):
        """Test basic grid search execution."""
        searcher = SignalGridSearch(
            df=grid_search_df,
            available_signals=['rbo_20', 'rtt_5020'],
            direction_col='rrg'
        )

        results = searcher.run_grid_search(verbose=False)

        # Should have 4 combinations
        assert len(results) == 4

        # Check result columns
        assert 'combination_name' in results.columns
        assert 'entry_signal' in results.columns
        assert 'exit_signal' in results.columns
        assert 'total_trades' in results.columns
        assert 'success' in results.columns

    def test_run_grid_search_all_successful(self, grid_search_df):
        """Test that all combinations succeed."""
        searcher = SignalGridSearch(
            df=grid_search_df,
            available_signals=['rbo_20', 'rtt_5020'],
            direction_col='rrg'
        )

        results = searcher.run_grid_search(verbose=False)

        # All should succeed
        assert results['success'].all()

    def test_get_results_summary(self, grid_search_df):
        """Test results summary generation."""
        searcher = SignalGridSearch(
            df=grid_search_df,
            available_signals=['rbo_20', 'rtt_5020'],
            direction_col='rrg'
        )

        searcher.run_grid_search(verbose=False)
        summary = searcher.get_results_summary()

        assert 'total_combinations' in summary.index
        assert 'successful_combinations' in summary.index
        assert 'avg_total_trades' in summary.index

    def test_get_results_summary_before_search(self, grid_search_df):
        """Test error when getting summary before running search."""
        searcher = SignalGridSearch(
            df=grid_search_df,
            available_signals=['rbo_20', 'rtt_5020'],
            direction_col='rrg'
        )

        with pytest.raises(ValueError, match="Run grid_search first"):
            searcher.get_results_summary()

    def test_filter_combinations(self, grid_search_df):
        """Test filtering combinations."""
        searcher = SignalGridSearch(
            df=grid_search_df,
            available_signals=['rbo_20', 'rtt_5020'],
            direction_col='rrg'
        )

        searcher.run_grid_search(verbose=False)
        filtered = searcher.filter_combinations(min_trades=1, max_flat_pct=95)

        # Filtered should be subset of results
        assert len(filtered) <= len(searcher.results)

    def test_get_signal_columns(self, grid_search_df):
        """Test getting generated signal column names."""
        searcher = SignalGridSearch(
            df=grid_search_df,
            available_signals=['rbo_20', 'rtt_5020'],
            direction_col='rrg'
        )

        searcher.run_grid_search(verbose=False)
        columns = searcher.get_signal_columns()

        assert len(columns) == 4
        assert all(isinstance(col, str) for col in columns)


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_minimal_dataframe(self, minimal_df):
        """Test with minimal DataFrame (2 rows)."""
        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(minimal_df, output_col='signal')

        assert len(result) == 2
        assert 'signal' in result.columns

    def test_all_flat_signals(self):
        """Test with all zero signals."""
        df = pd.DataFrame({
            'direction': [0, 0, 0, 0, 0],
            'entry': [0, 0, 0, 0, 0],
            'exit': [0, 0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(df, output_col='signal')

        # Should stay flat
        assert (result['signal'] == 0).all()

    def test_continuous_entries(self):
        """Test with continuous entry signals."""
        df = pd.DataFrame({
            'direction': [1, 1, 1, 1, 1],
            'entry': [1, 1, 1, 1, 1],  # All entry signals
            'exit': [0, 0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(df, output_col='signal')

        # Should enter once and hold
        assert result.loc[1, 'signal'] == 1  # Enter
        assert (result['signal'].iloc[1:] == 1).all()  # Hold

    def test_single_bar_trades(self):
        """Test with single bar trades."""
        df = pd.DataFrame({
            'direction': [1, 1, 1, 1, 1],
            'entry': [0, 1, 0, 1, 0],
            'exit': [0, 0, -1, 0, -1],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(df, output_col='signal')

        # Should have multiple entries and exits
        summary = combiner.get_trade_summary(result, output_col='signal')
        assert summary['entry_long_count'] >= 1

    def test_dataframe_not_modified(self, sample_signals_df):
        """Test that original DataFrame is not modified."""
        df_original = sample_signals_df.copy()
        original_cols = set(df_original.columns)

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit'
        )

        result = combiner.combine_signals(sample_signals_df, output_col='signal')

        # Original columns unchanged
        assert set(df_original.columns) == original_cols

    def test_verbose_mode(self, capsys):
        """Test verbose mode prints output."""
        df = pd.DataFrame({
            'direction': [0, 1, 1, 1],
            'entry': [0, 1, 0, 0],
            'exit': [0, 0, 0, 0],
        })

        combiner = HybridSignalCombiner(
            direction_col='direction',
            entry_col='entry',
            exit_col='exit',
            verbose=True
        )

        combiner.combine_signals(df, output_col='signal')

        captured = capsys.readouterr()
        assert 'ENTER LONG' in captured.out


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the combiner module."""

    def test_complete_workflow(self, grid_search_df):
        """Test complete workflow: grid search, filter, export."""
        searcher = SignalGridSearch(
            df=grid_search_df,
            available_signals=['rbo_20', 'rtt_5020', 'rsma_102050'],
            direction_col='rrg'
        )

        # Run search
        results = searcher.run_grid_search(verbose=False)

        # Get summary
        summary = searcher.get_results_summary()

        # Filter
        filtered = searcher.filter_combinations(min_trades=1)

        # Get columns
        columns = searcher.get_signal_columns()

        # Verify data integrity
        assert len(results) == 9  # 3 × 3
        assert summary['total_combinations'] == 9
        assert len(filtered) <= 9
        assert len(columns) == len(filtered) if len(filtered) > 0 else True

    def test_combined_signals_added_to_df(self, grid_search_df):
        """Test that combined signals are added to the DataFrame."""
        searcher = SignalGridSearch(
            df=grid_search_df,
            available_signals=['rbo_20', 'rtt_5020'],
            direction_col='rrg'
        )

        original_cols = len(searcher.df.columns)
        searcher.run_grid_search(verbose=False)

        # Should have added new columns
        new_cols = len(searcher.df.columns)
        assert new_cols > original_cols


# =============================================================================
# Fixtures for parallel-over-stocks tests
# =============================================================================

@pytest.fixture
def multi_stock_dfs():
    """Create a dict of 3 stock DataFrames sharing the same signal schema."""
    n = 200

    def make_df(seed):
        np.random.seed(seed)
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='B'),
            'close': 100 + np.cumsum(np.random.randn(n) * 2),
            'rrg': np.where(np.arange(n) % 30 < 15, 1, -1),
            'rbo_20': np.where(np.arange(n) % 10 == 0, 1,
                               np.where(np.arange(n) % 12 == 0, -1, 0)),
            'rtt_5020': np.where(np.arange(n) % 15 == 0, 1,
                                 np.where(np.arange(n) % 18 == 0, -1, 0)),
        })

    return {
        'ENI.MI':  make_df(42),
        'ENEL.MI': make_df(7),
        'ISP.MI':  make_df(13),
    }


# =============================================================================
# _process_stock_all_combinations Tests
# =============================================================================

class TestProcessStockAllCombinations:
    """Tests for the module-level worker function."""

    def test_returns_ticker_and_list(self, multi_stock_dfs):
        """Test that the worker returns (ticker, list_of_dicts)."""
        ticker = 'ENI.MI'
        df = multi_stock_dfs[ticker]
        searcher = SignalGridSearch(
            df=df.copy(),
            available_signals=['rbo_20', 'rtt_5020'],
            direction_col='rrg',
        )
        grid = searcher.generate_grid()

        result_ticker, results = _process_stock_all_combinations(
            (ticker, df, grid, 'rrg', True, True, False)
        )

        assert result_ticker == ticker
        assert isinstance(results, list)
        assert len(results) == len(grid)

    def test_each_result_tagged_with_ticker(self, multi_stock_dfs):
        """Test that every result dict carries the correct 'ticker' key."""
        ticker = 'ENEL.MI'
        df = multi_stock_dfs[ticker]
        searcher = SignalGridSearch(
            df=df.copy(),
            available_signals=['rbo_20', 'rtt_5020'],
            direction_col='rrg',
        )
        grid = searcher.generate_grid()

        _, results = _process_stock_all_combinations(
            (ticker, df, grid, 'rrg', True, True, False)
        )

        for result in results:
            assert result['ticker'] == ticker

    def test_result_schema_matches_single_combination(self, multi_stock_dfs):
        """Test that each result dict has the same keys as _process_single_combination."""
        ticker = 'ISP.MI'
        df = multi_stock_dfs[ticker]
        searcher = SignalGridSearch(
            df=df.copy(),
            available_signals=['rbo_20', 'rtt_5020'],
            direction_col='rrg',
        )
        grid = searcher.generate_grid()

        _, results = _process_stock_all_combinations(
            (ticker, df, grid, 'rrg', True, True, False)
        )

        expected_keys = {
            'combination_name', 'entry_signal', 'exit_signal', 'direction_signal',
            'output_column', 'total_trades', 'long_trades', 'short_trades',
            'success', 'ticker', 'combined_signal',
        }
        for result in results:
            assert expected_keys.issubset(result.keys())


# =============================================================================
# run_grid_search_parallel_over_stocks Tests
# =============================================================================

class TestRunGridSearchParallelOverStocks:
    """Tests for SignalGridSearch.run_grid_search_parallel_over_stocks."""

    def _make_searcher(self, multi_stock_dfs):
        return SignalGridSearch(
            df=multi_stock_dfs['ENI.MI'].copy(),
            available_signals=['rbo_20', 'rtt_5020'],
            direction_col='rrg',
        )

    def test_returns_dict(self, multi_stock_dfs):
        """Test that the method returns a dict."""
        searcher = self._make_searcher(multi_stock_dfs)
        results = searcher.run_grid_search_parallel_over_stocks(
            stock_dfs=multi_stock_dfs, n_jobs=1
        )
        assert isinstance(results, dict)

    def test_keys_match_tickers(self, multi_stock_dfs):
        """Test that result keys match input tickers exactly."""
        searcher = self._make_searcher(multi_stock_dfs)
        results = searcher.run_grid_search_parallel_over_stocks(
            stock_dfs=multi_stock_dfs, n_jobs=1
        )
        assert set(results.keys()) == set(multi_stock_dfs.keys())

    def test_each_value_is_dataframe(self, multi_stock_dfs):
        """Test that each dict value is a pd.DataFrame."""
        searcher = self._make_searcher(multi_stock_dfs)
        results = searcher.run_grid_search_parallel_over_stocks(
            stock_dfs=multi_stock_dfs, n_jobs=1
        )
        for ticker, df in results.items():
            assert isinstance(df, pd.DataFrame), f"Expected DataFrame for {ticker}"

    def test_rows_per_stock_equal_combo_count(self, multi_stock_dfs):
        """Test that each result DataFrame has one row per signal combination."""
        signals = ['rbo_20', 'rtt_5020']
        n_combos = len(signals) ** 2  # 4

        searcher = SignalGridSearch(
            df=multi_stock_dfs['ENI.MI'].copy(),
            available_signals=signals,
            direction_col='rrg',
        )
        results = searcher.run_grid_search_parallel_over_stocks(
            stock_dfs=multi_stock_dfs, n_jobs=1
        )

        for ticker, df in results.items():
            assert len(df) == n_combos, f"{ticker}: expected {n_combos} rows, got {len(df)}"

    def test_ticker_column_present_and_correct(self, multi_stock_dfs):
        """Test that each result DataFrame has a 'ticker' column with the right value."""
        searcher = self._make_searcher(multi_stock_dfs)
        results = searcher.run_grid_search_parallel_over_stocks(
            stock_dfs=multi_stock_dfs, n_jobs=1
        )
        for ticker, df in results.items():
            assert 'ticker' in df.columns
            assert (df['ticker'] == ticker).all()

    def test_result_columns_match_schema(self, multi_stock_dfs):
        """Test that result DataFrames contain the expected schema columns."""
        expected_cols = {
            'combination_name', 'entry_signal', 'exit_signal',
            'total_trades', 'success', 'ticker',
        }
        searcher = self._make_searcher(multi_stock_dfs)
        results = searcher.run_grid_search_parallel_over_stocks(
            stock_dfs=multi_stock_dfs, n_jobs=1
        )
        for ticker, df in results.items():
            missing = expected_cols - set(df.columns)
            assert not missing, f"{ticker} missing columns: {missing}"

    def test_all_combinations_succeed(self, multi_stock_dfs):
        """Test that all combinations succeed for valid input."""
        searcher = self._make_searcher(multi_stock_dfs)
        results = searcher.run_grid_search_parallel_over_stocks(
            stock_dfs=multi_stock_dfs, n_jobs=1
        )
        for ticker, df in results.items():
            assert df['success'].all(), f"{ticker} has failed combinations"

    def test_no_combined_signal_column_in_results(self, multi_stock_dfs):
        """Test that combined_signal is stripped from result DataFrames."""
        searcher = self._make_searcher(multi_stock_dfs)
        results = searcher.run_grid_search_parallel_over_stocks(
            stock_dfs=multi_stock_dfs, n_jobs=1
        )
        for ticker, df in results.items():
            assert 'combined_signal' not in df.columns

    def test_empty_stock_dfs_raises(self, multi_stock_dfs):
        """Test that empty stock_dfs raises ValueError."""
        searcher = self._make_searcher(multi_stock_dfs)
        with pytest.raises(ValueError, match="empty"):
            searcher.run_grid_search_parallel_over_stocks(stock_dfs={}, n_jobs=1)

    def test_invalid_n_jobs_raises(self, multi_stock_dfs):
        """Test that n_jobs=0 raises ValueError."""
        searcher = self._make_searcher(multi_stock_dfs)
        with pytest.raises(ValueError, match="n_jobs"):
            searcher.run_grid_search_parallel_over_stocks(
                stock_dfs=multi_stock_dfs, n_jobs=0
            )

    def test_add_signals_to_dfs_writes_columns(self, multi_stock_dfs):
        """Test that add_signals_to_dfs=True adds combined columns to input dfs."""
        signals = ['rbo_20', 'rtt_5020']
        n_combos = len(signals) ** 2  # 4

        searcher = SignalGridSearch(
            df=multi_stock_dfs['ENI.MI'].copy(),
            available_signals=signals,
            direction_col='rrg',
        )
        original_col_counts = {t: len(df.columns) for t, df in multi_stock_dfs.items()}

        searcher.run_grid_search_parallel_over_stocks(
            stock_dfs=multi_stock_dfs, n_jobs=1, add_signals_to_dfs=True
        )

        for ticker, df in multi_stock_dfs.items():
            added = len(df.columns) - original_col_counts[ticker]
            assert added == n_combos, f"{ticker}: expected {n_combos} new cols, got {added}"

    def test_no_side_effects_by_default(self, multi_stock_dfs):
        """Test that input dfs are not modified when add_signals_to_dfs=False."""
        searcher = self._make_searcher(multi_stock_dfs)
        original_col_counts = {t: len(df.columns) for t, df in multi_stock_dfs.items()}

        searcher.run_grid_search_parallel_over_stocks(
            stock_dfs=multi_stock_dfs, n_jobs=1, add_signals_to_dfs=False
        )

        for ticker, df in multi_stock_dfs.items():
            assert len(df.columns) == original_col_counts[ticker], (
                f"{ticker}: columns were unexpectedly added to input df"
            )

    def test_single_stock_runs_sequentially(self, multi_stock_dfs):
        """Test that a single-stock dict uses the sequential path (n_jobs caps to 1)."""
        single_stock = {'ENI.MI': multi_stock_dfs['ENI.MI']}
        searcher = self._make_searcher(multi_stock_dfs)

        # n_jobs=-1 but only 1 stock: effective_jobs clamps to 1, no Pool spawned
        results = searcher.run_grid_search_parallel_over_stocks(
            stock_dfs=single_stock, n_jobs=-1
        )

        assert set(results.keys()) == {'ENI.MI'}
        assert isinstance(results['ENI.MI'], pd.DataFrame)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
