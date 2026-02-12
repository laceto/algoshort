"""
Comprehensive tests for the ReturnsCalculator module.

Tests cover:
- Constructor validation and OHLC column mapping
- Returns calculation (inplace and copy modes)
- Edge cases (empty DataFrame, single row, zero prices, extreme returns)
- Log returns clipping for -inf prevention
- Multiple signals parallel processing
"""
import pytest
import pandas as pd
import numpy as np
import logging
from algoshort.returns import ReturnsCalculator


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def basic_ohlc_df():
    """Basic OHLC DataFrame with both absolute and relative columns."""
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'ropen': [1.0, 1.01, 1.02, 1.03, 1.04],
        'rhigh': [1.05, 1.06, 1.07, 1.08, 1.09],
        'rlow': [0.95, 0.96, 0.97, 0.98, 0.99],
        'rclose': [1.02, 1.03, 1.04, 1.05, 1.06],
        'signal': [0, 1, 1, 1, 0],
    })


@pytest.fixture
def ohlc_with_multiple_signals(basic_ohlc_df):
    """DataFrame with multiple signal columns."""
    df = basic_ohlc_df.copy()
    df['signal_a'] = [0, 1, 1, -1, 0]
    df['signal_b'] = [1, 1, 0, 0, 1]
    df['signal_c'] = [0, 0, 1, 1, 1]
    return df


@pytest.fixture
def calculator(basic_ohlc_df):
    """Standard ReturnsCalculator instance."""
    return ReturnsCalculator(basic_ohlc_df)


# ============================================================================
# CONSTRUCTOR TESTS
# ============================================================================

class TestReturnsCalculatorInit:
    """Tests for ReturnsCalculator constructor."""

    def test_init_with_valid_data(self, basic_ohlc_df):
        """Constructor should succeed with valid OHLC columns."""
        calc = ReturnsCalculator(basic_ohlc_df)
        assert calc.ohlc_stock is basic_ohlc_df
        assert calc._base_cols == ('open', 'high', 'low', 'close')

    def test_init_missing_absolute_columns(self):
        """Constructor should raise KeyError if absolute OHLC columns are missing."""
        df = pd.DataFrame({
            'open': [100], 'high': [105], 'low': [95],  # Missing 'close'
            'ropen': [1.0], 'rhigh': [1.05], 'rlow': [0.95], 'rclose': [1.02]
        })
        with pytest.raises(KeyError, match="Missing OHLC columns"):
            ReturnsCalculator(df)

    def test_init_missing_relative_columns(self):
        """Constructor should raise KeyError if relative OHLC columns are missing."""
        df = pd.DataFrame({
            'open': [100], 'high': [105], 'low': [95], 'close': [102],
            'ropen': [1.0], 'rhigh': [1.05], 'rlow': [0.95],  # Missing 'rclose'
        })
        with pytest.raises(KeyError, match="Missing OHLC columns"):
            ReturnsCalculator(df)

    def test_init_custom_column_names(self):
        """Constructor should work with custom column names."""
        df = pd.DataFrame({
            'Open': [100], 'High': [105], 'Low': [95], 'Close': [102],
            'rOpen': [1.0], 'rHigh': [1.05], 'rLow': [0.95], 'rClose': [1.02]
        })
        calc = ReturnsCalculator(
            df,
            open_col='Open',
            high_col='High',
            low_col='Low',
            close_col='Close',
            relative_prefix='r'
        )
        assert calc._base_cols == ('Open', 'High', 'Low', 'Close')

    def test_init_custom_logger(self, basic_ohlc_df):
        """Constructor should accept custom logger."""
        custom_logger = logging.getLogger('test_logger')
        calc = ReturnsCalculator(basic_ohlc_df, logger=custom_logger)
        assert calc.logger is custom_logger


# ============================================================================
# OHLC COLUMN MAPPING TESTS
# ============================================================================

class TestGetOHLCColumns:
    """Tests for _get_ohlc_columns method."""

    def test_get_absolute_columns(self, calculator):
        """Should return absolute column names when relative=False."""
        cols = calculator._get_ohlc_columns(relative=False)
        assert cols == ('open', 'high', 'low', 'close')

    def test_get_relative_columns(self, calculator):
        """Should return relative column names when relative=True."""
        cols = calculator._get_ohlc_columns(relative=True)
        assert cols == ('ropen', 'rhigh', 'rlow', 'rclose')


# ============================================================================
# GET_RETURNS TESTS - INPUT VALIDATION
# ============================================================================

class TestGetReturnsValidation:
    """Tests for get_returns input validation."""

    def test_empty_dataframe_raises_error(self, calculator):
        """Should raise ValueError for empty DataFrame."""
        empty_df = pd.DataFrame(columns=['close', 'signal'])
        with pytest.raises(ValueError, match="empty"):
            calculator.get_returns(empty_df, 'signal')

    def test_single_row_raises_error(self, calculator):
        """Should raise ValueError for single-row DataFrame."""
        single_row = pd.DataFrame({
            'open': [100.0], 'high': [105.0], 'low': [95.0], 'close': [102.0],
            'ropen': [1.0], 'rhigh': [1.05], 'rlow': [0.95], 'rclose': [1.02],
            'signal': [1]
        })
        with pytest.raises(ValueError, match="at least 2 rows"):
            calculator.get_returns(single_row, 'signal')

    def test_missing_signal_column_raises_error(self, calculator, basic_ohlc_df):
        """Should raise KeyError for missing signal column."""
        with pytest.raises(KeyError, match="not found"):
            calculator.get_returns(basic_ohlc_df, 'nonexistent_signal')

    def test_non_numeric_close_raises_error(self, calculator):
        """Should raise ValueError for non-numeric close column."""
        df = pd.DataFrame({
            'open': [100.0, 101.0], 'high': [105.0, 106.0],
            'low': [95.0, 96.0], 'close': ['a', 'b'],  # Non-numeric
            'ropen': [1.0, 1.01], 'rhigh': [1.05, 1.06],
            'rlow': [0.95, 0.96], 'rclose': [1.02, 1.03],
            'signal': [0, 1]
        })
        with pytest.raises(ValueError, match="numeric"):
            calculator.get_returns(df, 'signal')

    def test_non_numeric_signal_raises_error(self, calculator):
        """Should raise ValueError for non-numeric signal column."""
        df = pd.DataFrame({
            'open': [100.0, 101.0], 'high': [105.0, 106.0],
            'low': [95.0, 96.0], 'close': [102.0, 103.0],
            'ropen': [1.0, 1.01], 'rhigh': [1.05, 1.06],
            'rlow': [0.95, 0.96], 'rclose': [1.02, 1.03],
            'signal': ['buy', 'sell']  # Non-numeric
        })
        with pytest.raises(ValueError, match="numeric"):
            calculator.get_returns(df, 'signal')


# ============================================================================
# GET_RETURNS TESTS - CALCULATION CORRECTNESS
# ============================================================================

class TestGetReturnsCalculation:
    """Tests for get_returns calculation correctness."""

    def test_returns_columns_created(self, calculator, basic_ohlc_df):
        """Should create all expected return columns."""
        result = calculator.get_returns(basic_ohlc_df, 'signal')

        expected_columns = [
            'signal_chg1D', 'signal_chg1D_fx',
            'signal_PL_cum', 'signal_PL_cum_fx',
            'signal_returns', 'signal_log_returns', 'signal_cumul'
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_inplace_false_returns_copy(self, calculator, basic_ohlc_df):
        """With inplace=False, should return a copy and not modify original."""
        original_columns = list(basic_ohlc_df.columns)
        result = calculator.get_returns(basic_ohlc_df, 'signal', inplace=False)

        assert list(basic_ohlc_df.columns) == original_columns
        assert len(result.columns) > len(original_columns)

    def test_inplace_true_modifies_original(self, calculator, basic_ohlc_df):
        """With inplace=True, should modify original DataFrame."""
        original_columns = len(basic_ohlc_df.columns)
        result = calculator.get_returns(basic_ohlc_df, 'signal', inplace=True)

        # The critical bug was that inplace=True didn't add columns
        # Now it should add columns to the original DataFrame
        assert len(basic_ohlc_df.columns) > original_columns
        assert 'signal_chg1D' in basic_ohlc_df.columns
        assert result is basic_ohlc_df

    def test_signal_nan_filled_with_zero(self, calculator):
        """NaN values in signal should be filled with 0."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [102.0, 103.0, 104.0],
            'ropen': [1.0, 1.01, 1.02],
            'rhigh': [1.05, 1.06, 1.07],
            'rlow': [0.95, 0.96, 0.97],
            'rclose': [1.02, 1.03, 1.04],
            'signal': [np.nan, 1, np.nan]
        })
        result = calculator.get_returns(df, 'signal')
        # Original NaN at index 0 should now be 0
        assert result['signal'].iloc[0] == 0
        assert result['signal'].iloc[2] == 0

    def test_lagged_signal_first_row_nan(self, calculator, basic_ohlc_df):
        """First row of calculated columns should have NaN due to shift/diff."""
        result = calculator.get_returns(basic_ohlc_df, 'signal')

        # First row should be NaN for lagged calculations
        assert pd.isna(result['signal_chg1D'].iloc[0])
        assert pd.isna(result['signal_returns'].iloc[0])

    def test_cumulative_calculation(self, calculator, basic_ohlc_df):
        """Cumulative sum should accumulate correctly."""
        result = calculator.get_returns(basic_ohlc_df, 'signal')

        # Manual verification of cumsum
        chg1D = result['signal_chg1D']
        expected_cumsum = chg1D.cumsum()
        pd.testing.assert_series_equal(
            result['signal_PL_cum'],
            expected_cumsum,
            check_names=False  # Names differ: 'signal_PL_cum' vs 'signal_chg1D'
        )

    def test_relative_mode(self, calculator, basic_ohlc_df):
        """Should use relative columns when relative=True."""
        result = calculator.get_returns(basic_ohlc_df, 'signal', relative=True)

        # Calculations should be based on 'rclose', not 'close'
        assert 'signal_chg1D' in result.columns


# ============================================================================
# GET_RETURNS TESTS - EDGE CASES
# ============================================================================

class TestGetReturnsEdgeCases:
    """Tests for get_returns edge cases."""

    def test_all_zero_signal(self, calculator):
        """Should handle all-zero signal correctly."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [102.0, 103.0, 104.0],
            'ropen': [1.0, 1.01, 1.02],
            'rhigh': [1.05, 1.06, 1.07],
            'rlow': [0.95, 0.96, 0.97],
            'rclose': [1.02, 1.03, 1.04],
            'signal': [0, 0, 0]
        })
        result = calculator.get_returns(df, 'signal')

        # All values should be 0 (or NaN for first row)
        assert result['signal_chg1D'].iloc[1:].sum() == 0
        assert result['signal_returns'].iloc[1:].sum() == 0

    def test_constant_prices(self, calculator):
        """Should handle constant prices (no change)."""
        df = pd.DataFrame({
            'open': [100.0, 100.0, 100.0],
            'high': [100.0, 100.0, 100.0],
            'low': [100.0, 100.0, 100.0],
            'close': [100.0, 100.0, 100.0],
            'ropen': [1.0, 1.0, 1.0],
            'rhigh': [1.0, 1.0, 1.0],
            'rlow': [1.0, 1.0, 1.0],
            'rclose': [1.0, 1.0, 1.0],
            'signal': [0, 1, 1]
        })
        result = calculator.get_returns(df, 'signal')

        # Daily change should be 0 for constant prices
        assert result['signal_chg1D'].iloc[1] == 0
        assert result['signal_chg1D'].iloc[2] == 0

    def test_negative_signal_short_position(self, calculator):
        """Should handle negative signals (short positions)."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [102.0, 103.0, 104.0],
            'ropen': [1.0, 1.01, 1.02],
            'rhigh': [1.05, 1.06, 1.07],
            'rlow': [0.95, 0.96, 0.97],
            'rclose': [1.02, 1.03, 1.04],
            'signal': [0, -1, -1]  # Short position
        })
        result = calculator.get_returns(df, 'signal')

        # Price increase with short = negative return
        assert result['signal_chg1D'].iloc[2] < 0

    def test_log_returns_extreme_negative_clipping(self, calculator):
        """Should clip extreme negative returns to prevent -inf."""
        # Create scenario with 99.99% loss (just above clip threshold)
        df = pd.DataFrame({
            'open': [100.0, 0.01, 0.01],  # Near-zero price
            'high': [105.0, 0.02, 0.02],
            'low': [95.0, 0.005, 0.005],
            'close': [100.0, 0.01, 0.01],  # 99.99% drop
            'ropen': [1.0, 0.0001, 0.0001],
            'rhigh': [1.05, 0.0002, 0.0002],
            'rlow': [0.95, 0.00005, 0.00005],
            'rclose': [1.0, 0.0001, 0.0001],
            'signal': [0, 1, 1]
        })
        result = calculator.get_returns(df, 'signal')

        # Log returns should not be -inf (clipped to -0.9999)
        assert not np.isinf(result['signal_log_returns']).any()

    def test_two_row_minimum(self, calculator):
        """Should work with exactly 2 rows (minimum required)."""
        df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [95.0, 96.0],
            'close': [102.0, 103.0],
            'ropen': [1.0, 1.01],
            'rhigh': [1.05, 1.06],
            'rlow': [0.95, 0.96],
            'rclose': [1.02, 1.03],
            'signal': [0, 1]
        })
        result = calculator.get_returns(df, 'signal')

        assert len(result) == 2
        assert 'signal_chg1D' in result.columns


# ============================================================================
# GET_RETURNS_MULTIPLE TESTS
# ============================================================================

class TestGetReturnsMultiple:
    """Tests for get_returns_multiple parallel processing."""

    def test_empty_signals_list(self, calculator, basic_ohlc_df):
        """Should return input DataFrame when signals list is empty."""
        result = calculator.get_returns_multiple(basic_ohlc_df, [])
        pd.testing.assert_frame_equal(result, basic_ohlc_df)

    def test_empty_signals_list_inplace(self, calculator, basic_ohlc_df):
        """Should return same DataFrame when signals list is empty (inplace)."""
        result = calculator.get_returns_multiple(basic_ohlc_df, [], inplace=True)
        assert result is basic_ohlc_df

    def test_missing_signal_columns(self, calculator, basic_ohlc_df):
        """Should raise KeyError for missing signal columns."""
        with pytest.raises(KeyError, match="Missing signal columns"):
            calculator.get_returns_multiple(basic_ohlc_df, ['nonexistent_a', 'nonexistent_b'])

    def test_multiple_signals_processed(self, calculator, ohlc_with_multiple_signals):
        """Should process multiple signals and add all columns."""
        signals = ['signal_a', 'signal_b', 'signal_c']
        result = calculator.get_returns_multiple(ohlc_with_multiple_signals, signals, n_jobs=1)

        for sig in signals:
            assert f'{sig}_chg1D' in result.columns
            assert f'{sig}_PL_cum' in result.columns
            assert f'{sig}_returns' in result.columns
            assert f'{sig}_log_returns' in result.columns
            assert f'{sig}_cumul' in result.columns

    def test_multiple_signals_inplace(self, calculator, ohlc_with_multiple_signals):
        """Should modify original DataFrame when inplace=True."""
        signals = ['signal_a', 'signal_b']
        original_cols = len(ohlc_with_multiple_signals.columns)

        result = calculator.get_returns_multiple(
            ohlc_with_multiple_signals, signals, inplace=True, n_jobs=1
        )

        # Columns should be added to original
        assert len(ohlc_with_multiple_signals.columns) > original_cols
        assert result is ohlc_with_multiple_signals

    def test_parallel_vs_sequential_same_result(self, calculator, ohlc_with_multiple_signals):
        """Parallel processing should produce same result as sequential."""
        signals = ['signal_a', 'signal_b']

        # Sequential (n_jobs=1)
        result_seq = calculator.get_returns_multiple(
            ohlc_with_multiple_signals.copy(), signals, n_jobs=1, verbose=False
        )

        # Parallel (n_jobs=2)
        result_par = calculator.get_returns_multiple(
            ohlc_with_multiple_signals.copy(), signals, n_jobs=2, verbose=False
        )

        # Results should be identical
        for sig in signals:
            pd.testing.assert_series_equal(
                result_seq[f'{sig}_chg1D'],
                result_par[f'{sig}_chg1D']
            )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self, basic_ohlc_df):
        """Test complete workflow from initialization to returns calculation."""
        # Initialize
        calc = ReturnsCalculator(basic_ohlc_df)

        # Calculate returns
        result = calc.get_returns(basic_ohlc_df, 'signal')

        # Verify structure
        assert len(result) == len(basic_ohlc_df)
        assert all(col in result.columns for col in basic_ohlc_df.columns)
        assert 'signal_cumul' in result.columns

        # Verify calculations are reasonable
        assert not result['signal_chg1D'].iloc[1:].isna().all()

    def test_chained_calculations(self, basic_ohlc_df):
        """Test chained calculations with inplace modifications."""
        calc = ReturnsCalculator(basic_ohlc_df)
        df = basic_ohlc_df.copy()

        # Add multiple signals
        df['signal2'] = [1, 0, 1, 0, 1]

        # Calculate for both signals sequentially
        calc.get_returns(df, 'signal', inplace=True)
        calc.get_returns(df, 'signal2', inplace=True)

        # Both should exist
        assert 'signal_chg1D' in df.columns
        assert 'signal2_chg1D' in df.columns


# ============================================================================
# LOGGING TESTS
# ============================================================================

class TestLogging:
    """Tests for logging functionality."""

    def test_debug_logging_on_success(self, calculator, basic_ohlc_df, caplog):
        """Should log debug message on successful calculation."""
        with caplog.at_level(logging.DEBUG):
            calculator.get_returns(basic_ohlc_df, 'signal')

        assert any('Computed returns' in record.message for record in caplog.records)

    def test_error_logging_on_empty_df(self, calculator, caplog):
        """Should log error before raising on empty DataFrame."""
        empty_df = pd.DataFrame(columns=['close', 'signal'])

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                calculator.get_returns(empty_df, 'signal')

        assert any('empty' in record.message.lower() for record in caplog.records)

    def test_error_logging_on_missing_signal(self, calculator, basic_ohlc_df, caplog):
        """Should log error before raising on missing signal."""
        with caplog.at_level(logging.ERROR):
            with pytest.raises(KeyError):
                calculator.get_returns(basic_ohlc_df, 'missing_signal')

        assert any('not found' in record.message.lower() for record in caplog.records)
