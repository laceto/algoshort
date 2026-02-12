"""
Comprehensive tests for the StopLossCalculator module.

Tests cover:
- Constructor and data setter validation
- Signal column validation
- Parameter validation (percentage, multiplier, window)
- All stop-loss method calculations
- Edge cases (negative stops, zero prices, extreme parameters)
- Cache mechanism
- Generic interface (get_stop_loss)
"""
import pytest
import pandas as pd
import numpy as np
import logging
from algoshort.stop_loss import StopLossCalculator, MIN_STOP_PRICE


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def basic_ohlc_df():
    """Basic OHLC DataFrame with signal column."""
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'signal': [0, 1, 1, -1, 0],
    })


@pytest.fixture
def relative_ohlc_df():
    """OHLC DataFrame with relative columns."""
    return pd.DataFrame({
        'r_open': [1.00, 1.01, 1.02, 1.03, 1.04],
        'r_high': [1.05, 1.06, 1.07, 1.08, 1.09],
        'r_low': [0.95, 0.96, 0.97, 0.98, 0.99],
        'r_close': [1.02, 1.03, 1.04, 1.05, 1.06],
        'signal': [0, 1, 1, -1, 0],
    })


@pytest.fixture
def calculator(basic_ohlc_df):
    """Standard StopLossCalculator instance."""
    return StopLossCalculator(basic_ohlc_df)


@pytest.fixture
def large_ohlc_df():
    """Larger DataFrame for performance and calculation tests."""
    np.random.seed(42)
    n = 100
    prices = 100 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        'open': prices + np.random.rand(n),
        'high': prices + np.abs(np.random.randn(n) * 2),
        'low': prices - np.abs(np.random.randn(n) * 2),
        'close': prices,
        'signal': np.random.choice([-1, 0, 1], n),
    })


# ============================================================================
# CONSTRUCTOR TESTS
# ============================================================================

class TestStopLossCalculatorInit:
    """Tests for StopLossCalculator constructor."""

    def test_init_with_valid_data(self, basic_ohlc_df):
        """Constructor should succeed with valid OHLC columns."""
        calc = StopLossCalculator(basic_ohlc_df)
        assert calc.data is not None
        assert len(calc.data) == len(basic_ohlc_df)
        assert calc.price_cols == {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}

    def test_init_with_relative_columns(self, relative_ohlc_df):
        """Constructor should detect relative columns."""
        calc = StopLossCalculator(relative_ohlc_df)
        assert calc.price_cols == {'open': 'r_open', 'high': 'r_high', 'low': 'r_low', 'close': 'r_close'}

    def test_init_empty_dataframe_raises_error(self):
        """Constructor should raise ValueError for empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            StopLossCalculator(empty_df)

    def test_init_none_raises_error(self):
        """Constructor should raise ValueError for None."""
        with pytest.raises(ValueError, match="empty"):
            StopLossCalculator(None)

    def test_init_missing_ohlc_columns_raises_error(self):
        """Constructor should raise KeyError if OHLC columns missing."""
        df = pd.DataFrame({
            'open': [100], 'high': [105], 'low': [95],  # Missing 'close'
            'signal': [1]
        })
        with pytest.raises(KeyError, match="OHLC columns not found"):
            StopLossCalculator(df)


# ============================================================================
# DATA SETTER TESTS
# ============================================================================

class TestDataSetter:
    """Tests for data property setter."""

    def test_data_setter_clears_cache(self, calculator, basic_ohlc_df):
        """Setting new data should clear the cache."""
        # Populate cache
        _ = calculator._atr(window=14)
        assert len(calculator._cache) > 0

        # Set new data
        calculator.data = basic_ohlc_df.copy()

        # Cache should be cleared
        assert len(calculator._cache) == 0

    def test_data_setter_redetects_columns(self, basic_ohlc_df, relative_ohlc_df):
        """Setting new data should re-detect columns."""
        calc = StopLossCalculator(basic_ohlc_df)
        assert calc.price_cols['close'] == 'close'

        calc.data = relative_ohlc_df
        assert calc.price_cols['close'] == 'r_close'


# ============================================================================
# VALIDATION TESTS
# ============================================================================

class TestSignalValidation:
    """Tests for signal column validation."""

    def test_missing_signal_column_raises_error(self, calculator):
        """Should raise KeyError for missing signal column."""
        with pytest.raises(KeyError, match="not found"):
            calculator.fixed_percentage_stop_loss('nonexistent_signal')

    def test_non_numeric_signal_raises_error(self, basic_ohlc_df):
        """Should raise ValueError for non-numeric signal column."""
        basic_ohlc_df['string_signal'] = ['buy', 'sell', 'hold', 'buy', 'sell']
        calc = StopLossCalculator(basic_ohlc_df)
        with pytest.raises(ValueError, match="numeric"):
            calc.fixed_percentage_stop_loss('string_signal')


class TestPercentageValidation:
    """Tests for percentage parameter validation."""

    def test_percentage_zero_raises_error(self, calculator):
        """Percentage of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculator.fixed_percentage_stop_loss('signal', percentage=0)

    def test_percentage_negative_raises_error(self, calculator):
        """Negative percentage should raise ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculator.fixed_percentage_stop_loss('signal', percentage=-0.05)

    def test_percentage_one_raises_error(self, calculator):
        """Percentage of 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculator.fixed_percentage_stop_loss('signal', percentage=1.0)

    def test_percentage_greater_than_one_raises_error(self, calculator):
        """Percentage > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculator.fixed_percentage_stop_loss('signal', percentage=5.0)


class TestMultiplierValidation:
    """Tests for multiplier parameter validation."""

    def test_multiplier_zero_raises_error(self, calculator):
        """Multiplier of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calculator.atr_stop_loss('signal', multiplier=0)

    def test_multiplier_negative_raises_error(self, calculator):
        """Negative multiplier should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calculator.atr_stop_loss('signal', multiplier=-2.0)


class TestWindowValidation:
    """Tests for window parameter validation."""

    def test_window_zero_raises_error(self, calculator):
        """Window of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            calculator.breakout_channel_stop_loss('signal', window=0)

    def test_window_negative_raises_error(self, calculator):
        """Negative window should raise ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            calculator.breakout_channel_stop_loss('signal', window=-5)

    def test_window_non_integer_converted(self, calculator):
        """Float window should be converted to int."""
        result = calculator.breakout_channel_stop_loss('signal', window=5.7)
        assert 'signal_stop_loss' in result.columns


# ============================================================================
# FIXED PERCENTAGE STOP LOSS TESTS
# ============================================================================

class TestFixedPercentageStopLoss:
    """Tests for fixed_percentage_stop_loss method."""

    def test_creates_stop_loss_column(self, calculator):
        """Should create '{signal}_stop_loss' column."""
        result = calculator.fixed_percentage_stop_loss('signal')
        assert 'signal_stop_loss' in result.columns

    def test_long_stop_below_price(self, calculator):
        """Long position stops should be below entry price."""
        result = calculator.fixed_percentage_stop_loss('signal', percentage=0.05)
        long_mask = result['signal'] > 0
        if long_mask.any():
            assert (result.loc[long_mask, 'signal_stop_loss'] < result.loc[long_mask, 'close']).all()

    def test_short_stop_above_price(self, calculator):
        """Short position stops should be above entry price."""
        result = calculator.fixed_percentage_stop_loss('signal', percentage=0.05)
        short_mask = result['signal'] < 0
        if short_mask.any():
            assert (result.loc[short_mask, 'signal_stop_loss'] > result.loc[short_mask, 'close']).all()

    def test_neutral_signal_has_nan(self, calculator):
        """Neutral positions (signal=0) should have NaN stop loss."""
        result = calculator.fixed_percentage_stop_loss('signal')
        neutral_mask = result['signal'] == 0
        assert result.loc[neutral_mask, 'signal_stop_loss'].isna().all()

    def test_forward_fill_option(self, calculator):
        """forward_fill=True should fill NaN values."""
        result = calculator.fixed_percentage_stop_loss('signal', forward_fill=True)
        # After forward fill, only initial NaN rows should remain
        assert result['signal_stop_loss'].isna().sum() <= 1  # First row may be NaN


# ============================================================================
# ATR STOP LOSS TESTS
# ============================================================================

class TestATRStopLoss:
    """Tests for atr_stop_loss method."""

    def test_creates_stop_loss_column(self, calculator):
        """Should create '{signal}_stop_loss' column."""
        result = calculator.atr_stop_loss('signal')
        assert 'signal_stop_loss' in result.columns

    def test_default_forward_fill_true(self, calculator):
        """By default, forward_fill should be True for ATR."""
        result = calculator.atr_stop_loss('signal')
        # With forward_fill=True, fewer NaN values
        neutral_mask = result['signal'] == 0
        if neutral_mask.any():
            # Some neutral positions should have filled values
            assert not result['signal_stop_loss'].isna().all()

    def test_multiplier_affects_distance(self, calculator):
        """Higher multiplier should produce wider stops."""
        result_small = calculator.atr_stop_loss('signal', multiplier=1.0)
        result_large = calculator.atr_stop_loss('signal', multiplier=3.0)

        long_mask = result_small['signal'] > 0
        if long_mask.any():
            # Larger multiplier = lower long stops (wider)
            assert (result_large.loc[long_mask, 'signal_stop_loss'] <=
                    result_small.loc[long_mask, 'signal_stop_loss']).any()


# ============================================================================
# BREAKOUT CHANNEL STOP LOSS TESTS
# ============================================================================

class TestBreakoutChannelStopLoss:
    """Tests for breakout_channel_stop_loss method."""

    def test_creates_stop_loss_column(self, calculator):
        """Should create '{signal}_stop_loss' column."""
        result = calculator.breakout_channel_stop_loss('signal')
        assert 'signal_stop_loss' in result.columns

    def test_long_stop_is_swing_low(self, large_ohlc_df):
        """Long stops should be based on rolling minimum of lows."""
        calc = StopLossCalculator(large_ohlc_df)
        result = calc.breakout_channel_stop_loss('signal', window=10)

        swing_lows = large_ohlc_df['low'].rolling(window=10).min()
        swing_lows = np.maximum(swing_lows, MIN_STOP_PRICE)

        long_mask = result['signal'] > 0
        if long_mask.any():
            # Account for MIN_STOP_PRICE floor
            np.testing.assert_array_almost_equal(
                result.loc[long_mask, 'signal_stop_loss'].values,
                swing_lows[long_mask].values
            )


# ============================================================================
# MOVING AVERAGE STOP LOSS TESTS
# ============================================================================

class TestMovingAverageStopLoss:
    """Tests for moving_average_stop_loss method."""

    def test_creates_stop_loss_column(self, calculator):
        """Should create '{signal}_stop_loss' column."""
        result = calculator.moving_average_stop_loss('signal')
        assert 'signal_stop_loss' in result.columns

    def test_offset_affects_stop(self, large_ohlc_df):
        """Offset should shift stops away from MA."""
        calc = StopLossCalculator(large_ohlc_df)
        result_no_offset = calc.moving_average_stop_loss('signal', offset=0)
        result_with_offset = calc.moving_average_stop_loss('signal', offset=5.0)

        long_mask = result_no_offset['signal'] > 0
        if long_mask.any():
            # With offset, long stops should be lower
            assert (result_with_offset.loc[long_mask, 'signal_stop_loss'] <=
                    result_no_offset.loc[long_mask, 'signal_stop_loss']).any()


# ============================================================================
# VOLATILITY STD STOP LOSS TESTS
# ============================================================================

class TestVolatilityStdStopLoss:
    """Tests for volatility_std_stop_loss method."""

    def test_creates_stop_loss_column(self, calculator):
        """Should create '{signal}_stop_loss' column."""
        result = calculator.volatility_std_stop_loss('signal')
        assert 'signal_stop_loss' in result.columns

    def test_multiplier_validation(self, calculator):
        """Should validate multiplier is positive."""
        with pytest.raises(ValueError, match="positive"):
            calculator.volatility_std_stop_loss('signal', multiplier=-1.5)


# ============================================================================
# SUPPORT RESISTANCE STOP LOSS TESTS
# ============================================================================

class TestSupportResistanceStopLoss:
    """Tests for support_resistance_stop_loss method."""

    def test_is_alias_for_breakout_channel(self, calculator):
        """Should produce same results as breakout_channel_stop_loss."""
        result_sr = calculator.support_resistance_stop_loss('signal', window=10)
        result_bc = calculator.breakout_channel_stop_loss('signal', window=10)

        pd.testing.assert_frame_equal(result_sr, result_bc)


# ============================================================================
# CLASSIFIED PIVOT STOP LOSS TESTS
# ============================================================================

class TestClassifiedPivotStopLoss:
    """Tests for classified_pivot_stop_loss method."""

    def test_creates_stop_loss_column(self, large_ohlc_df):
        """Should create '{signal}_stop_loss' column."""
        calc = StopLossCalculator(large_ohlc_df)
        result = calc.classified_pivot_stop_loss('signal')
        assert 'signal_stop_loss' in result.columns

    def test_configurable_swing_window(self, large_ohlc_df):
        """swing_window parameter should affect calculation."""
        calc = StopLossCalculator(large_ohlc_df)
        result_10 = calc.classified_pivot_stop_loss('signal', swing_window=10)
        result_30 = calc.classified_pivot_stop_loss('signal', swing_window=30)

        # Different swing windows should produce different results
        assert not result_10['signal_stop_loss'].equals(result_30['signal_stop_loss'])

    def test_division_by_zero_protection(self):
        """Should handle zero prices without division error."""
        df = pd.DataFrame({
            'open': [100.0, 0.0, 102.0],  # Zero price
            'high': [105.0, 0.0, 107.0],
            'low': [95.0, 0.0, 97.0],
            'close': [102.0, 0.0, 104.0],
            'signal': [1, 1, 1],
        })
        calc = StopLossCalculator(df)
        # Should not raise division by zero error
        result = calc.classified_pivot_stop_loss('signal')
        assert 'signal_stop_loss' in result.columns


# ============================================================================
# NEGATIVE STOP PROTECTION TESTS
# ============================================================================

class TestNegativeStopProtection:
    """Tests for MIN_STOP_PRICE floor protection."""

    def test_long_stops_floored_at_minimum(self, basic_ohlc_df):
        """Long stops should never go below MIN_STOP_PRICE."""
        # Create scenario with very high ATR relative to price
        basic_ohlc_df['close'] = 1.0  # Very low price
        basic_ohlc_df['low'] = 0.5
        basic_ohlc_df['high'] = 1.5
        calc = StopLossCalculator(basic_ohlc_df)

        # Use large multiplier to force negative calculation
        result = calc.atr_stop_loss('signal', multiplier=10.0)

        long_mask = result['signal'] > 0
        if long_mask.any():
            assert (result.loc[long_mask, 'signal_stop_loss'] >= MIN_STOP_PRICE).all()


# ============================================================================
# CACHE TESTS
# ============================================================================

class TestATRCache:
    """Tests for ATR caching mechanism."""

    def test_cache_stores_atr(self, calculator):
        """ATR calculation should be cached."""
        _ = calculator._atr(window=14)
        assert 'ATR_14' in calculator._cache

    def test_cache_hit_returns_same_result(self, calculator):
        """Subsequent calls should return cached result."""
        atr1 = calculator._atr(window=14)
        atr2 = calculator._atr(window=14)
        pd.testing.assert_series_equal(atr1, atr2)

    def test_different_windows_different_cache(self, calculator):
        """Different windows should create different cache entries."""
        _ = calculator._atr(window=10)
        _ = calculator._atr(window=20)
        assert 'ATR_10' in calculator._cache
        assert 'ATR_20' in calculator._cache


# ============================================================================
# GET_STOP_LOSS GENERIC INTERFACE TESTS
# ============================================================================

class TestGetStopLoss:
    """Tests for get_stop_loss generic interface."""

    def test_fixed_percentage_method(self, calculator):
        """Should call fixed_percentage_stop_loss."""
        result = calculator.get_stop_loss('signal', 'fixed_percentage', percentage=0.05)
        assert 'signal_stop_loss' in result.columns

    def test_atr_method(self, calculator):
        """Should call atr_stop_loss."""
        result = calculator.get_stop_loss('signal', 'atr', multiplier=2.0)
        assert 'signal_stop_loss' in result.columns

    def test_breakout_channel_method(self, calculator):
        """Should call breakout_channel_stop_loss."""
        result = calculator.get_stop_loss('signal', 'breakout_channel', window=10)
        assert 'signal_stop_loss' in result.columns

    def test_support_resistance_method(self, calculator):
        """Should call support_resistance_stop_loss."""
        result = calculator.get_stop_loss('signal', 'support_resistance', window=10)
        assert 'signal_stop_loss' in result.columns

    def test_moving_average_method(self, calculator):
        """Should call moving_average_stop_loss."""
        result = calculator.get_stop_loss('signal', 'moving_average', window=20)
        assert 'signal_stop_loss' in result.columns

    def test_volatility_std_method(self, calculator):
        """Should call volatility_std_stop_loss."""
        result = calculator.get_stop_loss('signal', 'volatility_std', multiplier=1.5)
        assert 'signal_stop_loss' in result.columns

    def test_classified_pivot_method(self, large_ohlc_df):
        """Should call classified_pivot_stop_loss."""
        calc = StopLossCalculator(large_ohlc_df)
        result = calc.get_stop_loss('signal', 'classified_pivot')
        assert 'signal_stop_loss' in result.columns

    def test_unknown_method_raises_error(self, calculator):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown stop-loss method"):
            calculator.get_stop_loss('signal', 'nonexistent_method')

    def test_filters_invalid_kwargs(self, calculator):
        """Should filter out kwargs not accepted by method."""
        # Pass extra kwargs that shouldn't cause error
        result = calculator.get_stop_loss(
            'signal', 'fixed_percentage',
            percentage=0.05,
            invalid_param='should_be_ignored'
        )
        assert 'signal_stop_loss' in result.columns


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_zero_signals(self, basic_ohlc_df):
        """Should handle all-zero signals (all NaN stops)."""
        basic_ohlc_df['signal'] = 0
        calc = StopLossCalculator(basic_ohlc_df)
        result = calc.fixed_percentage_stop_loss('signal')
        assert result['signal_stop_loss'].isna().all()

    def test_all_long_signals(self, basic_ohlc_df):
        """Should handle all-long signals."""
        basic_ohlc_df['signal'] = 1
        calc = StopLossCalculator(basic_ohlc_df)
        result = calc.fixed_percentage_stop_loss('signal', percentage=0.05)
        assert not result['signal_stop_loss'].isna().any()

    def test_all_short_signals(self, basic_ohlc_df):
        """Should handle all-short signals."""
        basic_ohlc_df['signal'] = -1
        calc = StopLossCalculator(basic_ohlc_df)
        result = calc.fixed_percentage_stop_loss('signal', percentage=0.05)
        assert not result['signal_stop_loss'].isna().any()

    def test_single_row_dataframe(self):
        """Should handle single-row DataFrame."""
        df = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'signal': [1],
        })
        calc = StopLossCalculator(df)
        result = calc.fixed_percentage_stop_loss('signal')
        assert len(result) == 1

    def test_nan_in_signal(self, basic_ohlc_df):
        """Should handle NaN values in signal column."""
        basic_ohlc_df['signal'] = [np.nan, 1, np.nan, -1, np.nan]
        calc = StopLossCalculator(basic_ohlc_df)
        result = calc.fixed_percentage_stop_loss('signal')
        # NaN signals should result in NaN stops
        assert result['signal_stop_loss'].isna().sum() == 3


# ============================================================================
# LOGGING TESTS
# ============================================================================

class TestLogging:
    """Tests for logging functionality."""

    def test_debug_logging_on_init(self, basic_ohlc_df, caplog):
        """Should log debug message on initialization."""
        with caplog.at_level(logging.DEBUG):
            StopLossCalculator(basic_ohlc_df)
        assert any('initialized' in record.message.lower() for record in caplog.records)

    def test_debug_logging_on_calculation(self, calculator, caplog):
        """Should log debug message on stop-loss calculation."""
        with caplog.at_level(logging.DEBUG):
            calculator.fixed_percentage_stop_loss('signal')
        assert any('fixed percentage' in record.message.lower() for record in caplog.records)

    def test_error_logging_on_missing_signal(self, calculator, caplog):
        """Should log error before raising on missing signal."""
        with caplog.at_level(logging.ERROR):
            with pytest.raises(KeyError):
                calculator.fixed_percentage_stop_loss('missing_signal')
        assert any('not found' in record.message.lower() for record in caplog.records)
