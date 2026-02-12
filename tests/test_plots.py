"""
Comprehensive tests for algoshort.plots module.

Tests cover:
- Validation helper functions
- All 14 plot functions with valid input
- Edge cases (empty DataFrame, missing columns, None ticker)
- Parameter validation (negative/zero window values)
- Already-indexed DataFrame handling
- Memory leak prevention verification
"""

import pytest
import pandas as pd
import numpy as np

# Handle matplotlib import - may fail due to numpy version mismatch
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    import matplotlib.pyplot as plt
    from algoshort.plots import (
        plot_abs_rel,
        plot_signal_bo,
        plot_signal_tt,
        plot_signal_ma,
        plot_signal_abs,
        plot_signal_rel,
        plot_regime_abs,
        plot_regime_rel,
        plot_profit_loss,
        plot_PL,
        plot_price_signal_cumreturns,
        plot_equity_risk,
        plot_shares_signal,
        plot_equity_amount,
        _validate_dataframe,
        _validate_ticker,
        _validate_positive_int,
        _close_figure,
        DATE_COLUMN,
        CLOSE_COLUMN,
        RCLOSE_COLUMN,
    )
    MATPLOTLIB_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    # Dummy definitions for skip
    plot_abs_rel = plot_signal_bo = plot_signal_tt = plot_signal_ma = None
    plot_signal_abs = plot_signal_rel = plot_regime_abs = plot_regime_rel = None
    plot_profit_loss = plot_PL = plot_price_signal_cumreturns = None
    plot_equity_risk = plot_shares_signal = plot_equity_amount = None
    _validate_dataframe = _validate_ticker = _validate_positive_int = None
    _close_figure = None
    DATE_COLUMN = CLOSE_COLUMN = RCLOSE_COLUMN = None

# Skip all tests if matplotlib is not available
pytestmark = pytest.mark.skipif(
    not MATPLOTLIB_AVAILABLE,
    reason="matplotlib not available (numpy version mismatch)"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_df():
    """Create sample DataFrame with date column."""
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(50))
    return pd.DataFrame({
        'date': dates,
        'close': close,
        'rclose': close * 0.95,
        'open': close - 1,
        'high': close + 2,
        'low': close - 2,
    })


@pytest.fixture
def indexed_df(sample_df):
    """Create sample DataFrame already indexed by date."""
    return sample_df.set_index('date')


@pytest.fixture
def breakout_df(sample_df):
    """DataFrame with breakout signal columns."""
    df = sample_df.copy()
    window = 20
    df[f'hi_{window}'] = df['close'].rolling(window).max()
    df[f'lo_{window}'] = df['close'].rolling(window).min()
    df[f'bo_{window}'] = np.where(df['close'] > df[f'hi_{window}'].shift(1), 1,
                                   np.where(df['close'] < df[f'lo_{window}'].shift(1), -1, 0))
    return df


@pytest.fixture
def turtle_df(sample_df):
    """DataFrame with turtle trading columns."""
    df = sample_df.copy()
    df['turtle_5520'] = np.random.choice([-1, 0, 1], size=len(df))
    return df


@pytest.fixture
def ma_df(sample_df):
    """DataFrame with moving average signal columns."""
    df = sample_df.copy()
    df['sma_102050'] = np.random.choice([-1, 0, 1], size=len(df))
    df['ema_102050'] = np.random.choice([-1, 0, 1], size=len(df))
    return df


@pytest.fixture
def signal_abs_df(sample_df):
    """DataFrame with absolute signal columns."""
    df = sample_df.copy()
    df['hi3'] = df['close'] * 1.05
    df['lo3'] = df['close'] * 0.95
    df['clg'] = np.where(np.random.rand(len(df)) > 0.9, df['close'], np.nan)
    df['flr'] = np.where(np.random.rand(len(df)) > 0.9, df['close'], np.nan)
    df['rg_ch'] = np.random.choice([0, 1, -1], size=len(df))
    df['rg'] = np.cumsum(df['rg_ch'])
    return df


@pytest.fixture
def signal_rel_df(sample_df):
    """DataFrame with relative signal columns."""
    df = sample_df.copy()
    df['rh3'] = df['rclose'] * 1.05
    df['rl3'] = df['rclose'] * 0.95
    df['rclg'] = np.where(np.random.rand(len(df)) > 0.9, df['rclose'], np.nan)
    df['rflr'] = np.where(np.random.rand(len(df)) > 0.9, df['rclose'], np.nan)
    df['rrg_ch'] = np.random.choice([0, 1, -1], size=len(df))
    df['rrg'] = np.cumsum(df['rrg_ch'])
    return df


@pytest.fixture
def profit_loss_df(sample_df):
    """DataFrame with P&L columns."""
    df = sample_df.copy()
    df['tt_PL_cum'] = np.cumsum(np.random.randn(len(df)) * 100)
    df['tt_chg1D'] = np.random.randn(len(df)) * 100
    return df


@pytest.fixture
def price_signal_df(sample_df):
    """DataFrame with price/signal/returns columns."""
    df = sample_df.copy()
    df['stop_loss'] = df['close'] * 0.95
    df['signal_col'] = np.random.choice([-1, 0, 1], size=len(df))
    df['tt_cumul'] = np.cumsum(np.random.randn(len(df)) * 0.01)
    return df


@pytest.fixture
def equity_risk_df(sample_df):
    """DataFrame with equity risk columns."""
    df = sample_df.copy()
    df['peak_eqty'] = df['close'].expanding().max()
    df['tolerance'] = df['peak_eqty'] * 0.9
    df['drawdown'] = (df['close'] - df['peak_eqty']) / df['peak_eqty']
    df['constant_risk'] = 0.02
    df['convex_risk'] = np.abs(df['drawdown']) * 0.5
    df['concave_risk'] = 0.02 - np.abs(df['drawdown']) * 0.3
    return df


@pytest.fixture
def shares_signal_df(sample_df):
    """DataFrame with shares/signal columns."""
    df = sample_df.copy()
    df['shs_eql'] = 100
    df['shs_fxd'] = 50
    df['shs_ccv'] = np.random.randint(50, 150, size=len(df))
    df['shs_cvx'] = np.random.randint(50, 150, size=len(df))
    df['my_signal'] = np.random.choice([-1, 0, 1], size=len(df))
    return df


@pytest.fixture
def equity_amount_df(sample_df):
    """DataFrame with equity amount columns."""
    df = sample_df.copy()
    df['constant'] = np.cumsum(np.random.randn(len(df)) * 100) + 10000
    df['concave'] = np.cumsum(np.random.randn(len(df)) * 80) + 10000
    df['convex'] = np.cumsum(np.random.randn(len(df)) * 120) + 10000
    df['equal_weight'] = np.cumsum(np.random.randn(len(df)) * 90) + 10000
    df['tt_PL_cum_fx'] = np.cumsum(np.random.randn(len(df)) * 100) + 10000
    return df


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestValidateDataframe:
    """Tests for _validate_dataframe helper."""

    def test_valid_dataframe_with_date_column(self, sample_df):
        """Test validation with date column present."""
        result = _validate_dataframe(sample_df, ['close'], 'test_func')
        assert result.index.name == 'date'
        assert 'close' in result.columns

    def test_valid_dataframe_already_indexed(self, indexed_df):
        """Test validation with DataFrame already indexed."""
        result = _validate_dataframe(indexed_df, ['close'], 'test_func')
        assert result.index.name == 'date'
        assert 'close' in result.columns

    def test_not_a_dataframe(self):
        """Test validation fails for non-DataFrame input."""
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            _validate_dataframe([1, 2, 3], ['col'], 'test_func')

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_dataframe(pd.DataFrame(), ['col'], 'test_func')

    def test_missing_required_columns(self, sample_df):
        """Test validation fails for missing columns."""
        with pytest.raises(ValueError, match="Missing required columns"):
            _validate_dataframe(sample_df, ['nonexistent'], 'test_func')

    def test_original_df_not_modified(self, sample_df):
        """Test that original DataFrame is not modified."""
        original_columns = sample_df.columns.tolist()
        original_index = sample_df.index.tolist()
        _validate_dataframe(sample_df, ['close'], 'test_func')
        assert sample_df.columns.tolist() == original_columns
        assert sample_df.index.tolist() == original_index


class TestValidateTicker:
    """Tests for _validate_ticker helper."""

    def test_valid_ticker(self):
        """Test validation with valid ticker."""
        assert _validate_ticker('AAPL', 'test_func') == 'AAPL'

    def test_none_ticker(self):
        """Test validation fails for None ticker."""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            _validate_ticker(None, 'test_func')

    def test_empty_ticker(self):
        """Test validation fails for empty string ticker."""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            _validate_ticker('', 'test_func')

    def test_whitespace_ticker(self):
        """Test validation fails for whitespace-only ticker."""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            _validate_ticker('   ', 'test_func')


class TestValidatePositiveInt:
    """Tests for _validate_positive_int helper."""

    def test_valid_positive_int(self):
        """Test validation with positive integer."""
        assert _validate_positive_int(10, 'window', 'test_func') == 10

    def test_zero_value(self):
        """Test validation fails for zero."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            _validate_positive_int(0, 'window', 'test_func')

    def test_negative_value(self):
        """Test validation fails for negative values."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            _validate_positive_int(-5, 'window', 'test_func')

    def test_float_value(self):
        """Test validation fails for float values."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            _validate_positive_int(10.5, 'window', 'test_func')


class TestCloseFigure:
    """Tests for _close_figure helper."""

    def test_close_figure(self):
        """Test that _close_figure closes the current figure."""
        fig, ax = plt.subplots()
        assert plt.get_fignums()  # Figure exists
        _close_figure()
        assert not plt.get_fignums()  # Figure closed


# =============================================================================
# Plot Function Tests - Valid Input
# =============================================================================

class TestPlotAbsRel:
    """Tests for plot_abs_rel function."""

    def test_valid_input(self, sample_df):
        """Test with valid input."""
        fig, ax = plot_abs_rel(sample_df, 'AAPL', 'SPY')
        assert isinstance(fig, plt.Figure)
        assert ax is not None

    def test_already_indexed_df(self, indexed_df):
        """Test with already indexed DataFrame."""
        fig, ax = plot_abs_rel(indexed_df, 'AAPL', 'SPY')
        assert isinstance(fig, plt.Figure)

    def test_none_ticker(self, sample_df):
        """Test with None ticker."""
        with pytest.raises(ValueError, match="cannot be None"):
            plot_abs_rel(sample_df, None, 'SPY')

    def test_missing_columns(self, sample_df):
        """Test with missing required columns."""
        df = sample_df.drop(columns=['rclose'])
        with pytest.raises(ValueError, match="Missing required columns"):
            plot_abs_rel(df, 'AAPL', 'SPY')


class TestPlotSignalBo:
    """Tests for plot_signal_bo function."""

    def test_valid_input(self, breakout_df):
        """Test with valid input."""
        fig, ax = plot_signal_bo(breakout_df, 20, 'AAPL')
        assert isinstance(fig, plt.Figure)

    def test_relative_mode(self, breakout_df):
        """Test with relative mode (requires relative columns)."""
        df = breakout_df.copy()
        df['rhi_20'] = df['hi_20']
        df['rlo_20'] = df['lo_20']
        df['rbo_20'] = df['bo_20']
        fig, ax = plot_signal_bo(df, 20, 'AAPL', relative=True)
        assert isinstance(fig, plt.Figure)

    def test_invalid_window(self, breakout_df):
        """Test with invalid window parameter."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            plot_signal_bo(breakout_df, 0, 'AAPL')

    def test_negative_window(self, breakout_df):
        """Test with negative window parameter."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            plot_signal_bo(breakout_df, -5, 'AAPL')


class TestPlotSignalTt:
    """Tests for plot_signal_tt function."""

    def test_valid_input(self, turtle_df):
        """Test with valid input."""
        fig, ax = plot_signal_tt(turtle_df, 20, 55, 'AAPL')
        assert isinstance(fig, plt.Figure)

    def test_empty_ticker(self, turtle_df):
        """Test with empty ticker (allowed)."""
        fig, ax = plot_signal_tt(turtle_df, 20, 55, '')
        assert isinstance(fig, plt.Figure)

    def test_invalid_fast_period(self, turtle_df):
        """Test with invalid fast period."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            plot_signal_tt(turtle_df, 0, 55, 'AAPL')


class TestPlotSignalMa:
    """Tests for plot_signal_ma function."""

    def test_valid_input(self, ma_df):
        """Test with valid input."""
        fig, axes = plot_signal_ma(ma_df, 10, 20, 50, 'AAPL')
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 2

    def test_invalid_period(self, ma_df):
        """Test with invalid period."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            plot_signal_ma(ma_df, -10, 20, 50, 'AAPL')


class TestPlotSignalAbs:
    """Tests for plot_signal_abs function."""

    def test_valid_input(self, signal_abs_df):
        """Test with valid input."""
        fig, ax = plot_signal_abs(signal_abs_df, 'AAPL')
        assert isinstance(fig, plt.Figure)

    def test_missing_columns(self, sample_df):
        """Test with missing signal columns."""
        with pytest.raises(ValueError, match="Missing required columns"):
            plot_signal_abs(sample_df, 'AAPL')


class TestPlotSignalRel:
    """Tests for plot_signal_rel function."""

    def test_valid_input(self, signal_rel_df):
        """Test with valid input."""
        fig, ax = plot_signal_rel(signal_rel_df, 'AAPL')
        assert isinstance(fig, plt.Figure)


class TestPlotRegimeAbs:
    """Tests for plot_regime_abs function."""

    def test_valid_input(self, signal_abs_df):
        """Test with valid input."""
        fig, ax = plot_regime_abs(signal_abs_df, 'AAPL')
        assert isinstance(fig, plt.Figure)

    def test_with_floor_ceiling_markers(self, signal_abs_df):
        """Test that floor/ceiling markers are rendered."""
        # Ensure we have some non-NaN values in clg and flr
        df = signal_abs_df.copy()
        df.loc[df.index[5], 'clg'] = df.loc[df.index[5], 'close']
        df.loc[df.index[10], 'flr'] = df.loc[df.index[10], 'close']
        fig, ax = plot_regime_abs(df, 'AAPL')
        assert isinstance(fig, plt.Figure)


class TestPlotRegimeRel:
    """Tests for plot_regime_rel function."""

    def test_valid_input(self, signal_rel_df):
        """Test with valid input."""
        fig, ax = plot_regime_rel(signal_rel_df, 'AAPL')
        assert isinstance(fig, plt.Figure)


class TestPlotProfitLoss:
    """Tests for plot_profit_loss function."""

    def test_valid_input(self, profit_loss_df):
        """Test with valid input."""
        fig, ax = plot_profit_loss(profit_loss_df, 'AAPL', 'turtle')
        assert isinstance(fig, plt.Figure)

    def test_backward_compat_alias(self, profit_loss_df):
        """Test backward compatibility alias plot_PL."""
        fig, ax = plot_PL(profit_loss_df, 'AAPL')
        assert isinstance(fig, plt.Figure)


class TestPlotPriceSignalCumreturns:
    """Tests for plot_price_signal_cumreturns function."""

    def test_valid_input(self, price_signal_df):
        """Test with valid input."""
        fig, ax = plot_price_signal_cumreturns(
            price_signal_df, 'AAPL', 'signal_col', 'turtle'
        )
        assert isinstance(fig, plt.Figure)


class TestPlotEquityRisk:
    """Tests for plot_equity_risk function."""

    def test_valid_input(self, equity_risk_df):
        """Test with valid input."""
        fig, axes = plot_equity_risk(equity_risk_df, 'AAPL', 'turtle')
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 2


class TestPlotSharesSignal:
    """Tests for plot_shares_signal function."""

    def test_valid_input(self, shares_signal_df):
        """Test with valid input."""
        fig, ax = plot_shares_signal(
            shares_signal_df, 'AAPL', 'my_signal', 'turtle'
        )
        assert isinstance(fig, plt.Figure)


class TestPlotEquityAmount:
    """Tests for plot_equity_amount function."""

    def test_valid_input(self, equity_amount_df):
        """Test with valid input."""
        fig, ax = plot_equity_amount(equity_amount_df, 'AAPL', 'turtle')
        assert isinstance(fig, plt.Figure)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases across all plot functions."""

    def test_empty_dataframe(self):
        """Test all functions fail gracefully with empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="cannot be empty"):
            plot_abs_rel(df, 'AAPL', 'SPY')

    def test_single_row_dataframe(self, sample_df):
        """Test with single row DataFrame."""
        df = sample_df.head(1)
        # Should work (no NaN in required columns)
        fig, ax = plot_abs_rel(df, 'AAPL', 'SPY')
        assert isinstance(fig, plt.Figure)

    def test_all_nan_values(self, sample_df):
        """Test with DataFrame containing NaN values."""
        df = sample_df.copy()
        df['close'] = np.nan
        df['rclose'] = np.nan
        # Should still plot (matplotlib handles NaN)
        fig, ax = plot_abs_rel(df, 'AAPL', 'SPY')
        assert isinstance(fig, plt.Figure)

    def test_very_long_ticker(self, sample_df):
        """Test with very long ticker name."""
        fig, ax = plot_abs_rel(sample_df, 'A' * 100, 'SPY')
        assert isinstance(fig, plt.Figure)

    def test_special_chars_in_ticker(self, sample_df):
        """Test with special characters in ticker."""
        fig, ax = plot_abs_rel(sample_df, 'BRK.A', 'SPY')
        assert isinstance(fig, plt.Figure)

    def test_unicode_ticker(self, sample_df):
        """Test with unicode characters in ticker."""
        fig, ax = plot_abs_rel(sample_df, 'AAPL\u2122', 'SPY')
        assert isinstance(fig, plt.Figure)


# =============================================================================
# Memory Leak Prevention Tests
# =============================================================================

class TestMemoryLeakPrevention:
    """Tests to verify figures are properly closed."""

    def test_figures_closed_after_plot(self, sample_df):
        """Test that figures are closed after plotting."""
        initial_figs = len(plt.get_fignums())
        plot_abs_rel(sample_df, 'AAPL', 'SPY')
        # After _close_figure(), figure count should be same or less
        assert len(plt.get_fignums()) <= initial_figs

    def test_multiple_plots_no_accumulation(self, sample_df):
        """Test multiple plot calls don't accumulate figures."""
        initial_figs = len(plt.get_fignums())
        for _ in range(5):
            plot_abs_rel(sample_df, 'AAPL', 'SPY')
        # Figures should not accumulate
        assert len(plt.get_fignums()) <= initial_figs + 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for plot functions."""

    def test_plot_sequence_abs_to_regime(self, signal_abs_df):
        """Test plotting absolute signal then regime."""
        fig1, ax1 = plot_signal_abs(signal_abs_df, 'AAPL')
        fig2, ax2 = plot_regime_abs(signal_abs_df, 'AAPL')
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)

    def test_plot_sequence_rel_to_regime(self, signal_rel_df):
        """Test plotting relative signal then regime."""
        fig1, ax1 = plot_signal_rel(signal_rel_df, 'AAPL')
        fig2, ax2 = plot_regime_rel(signal_rel_df, 'AAPL')
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)

    def test_all_plots_with_same_df(self, sample_df):
        """Test that same base DataFrame can be used for multiple plots."""
        # Build up the DataFrame with all needed columns
        df = sample_df.copy()
        df['hi_20'] = df['close'].rolling(20).max()
        df['lo_20'] = df['close'].rolling(20).min()
        df['bo_20'] = 0
        df['turtle_5520'] = 0
        df['sma_102050'] = 0
        df['ema_102050'] = 0

        # Plot different signals
        fig1, ax1 = plot_abs_rel(df, 'AAPL', 'SPY')
        fig2, ax2 = plot_signal_bo(df, 20, 'AAPL')
        fig3, ax3 = plot_signal_tt(df, 20, 55, 'AAPL')
        fig4, axes = plot_signal_ma(df, 10, 20, 50, 'AAPL')

        assert all(isinstance(f, plt.Figure) for f in [fig1, fig2, fig3, fig4])
