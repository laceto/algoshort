"""
Comprehensive tests for algoshort.trading_summary module.

Tests cover:
- Single symbol trading summary generation
- Multi-symbol summary generation
- Edge cases (empty DataFrame, missing columns, NaN values)
- Signal change detection
- Position sizing integration
- Dashboard creation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from algoshort.trading_summary import (
    get_trading_summary,
    get_multi_symbol_summary,
    print_trading_summary,
    print_multi_symbol_summary,
    create_trading_dashboard,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_df():
    """Create sample DataFrame with all required columns."""
    np.random.seed(42)
    n_days = 20
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    close = 150 + np.cumsum(np.random.randn(n_days) * 2)

    # Create signal that changes on the last day
    signal = np.zeros(n_days)
    signal[5:15] = 1  # Long from day 5-14
    signal[15:] = 0   # Exit on day 15
    signal[-1] = 1    # New entry on last day

    df = pd.DataFrame({
        'date': dates,
        'close': close,
        'hybrid_signal': signal,
        'stop_loss': close * 0.95,
        'shs_eql': (10000 / close).astype(int),
        'shs_fxd': 100,
        'shs_ccv': (8000 / close).astype(int),
        'shs_cvx': (12000 / close).astype(int),
    })
    return df


@pytest.fixture
def sample_df_short():
    """Create sample DataFrame with short position."""
    np.random.seed(123)
    n_days = 20
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    close = 200 + np.cumsum(np.random.randn(n_days) * 2)

    signal = np.zeros(n_days)
    signal[10:] = -1  # Short position

    df = pd.DataFrame({
        'date': dates,
        'close': close,
        'hybrid_signal': signal,
        'stop_loss': close * 1.05,  # Stop above for shorts
    })
    return df


@pytest.fixture
def sample_df_flat():
    """Create sample DataFrame with flat position."""
    n_days = 10
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    close = np.full(n_days, 100.0)

    df = pd.DataFrame({
        'date': dates,
        'close': close,
        'hybrid_signal': np.zeros(n_days),
        'stop_loss': close * 0.95,
    })
    return df


@pytest.fixture
def multi_symbol_data(sample_df, sample_df_short):
    """Create dict of multiple symbol DataFrames."""
    return {
        'AAPL': sample_df,
        'MSFT': sample_df_short,
    }


# =============================================================================
# Test get_trading_summary
# =============================================================================

class TestGetTradingSummary:
    """Tests for get_trading_summary function."""

    def test_basic_summary(self, sample_df):
        """Test basic summary generation."""
        summary = get_trading_summary(sample_df, 'AAPL')

        assert summary['ticker'] == 'AAPL'
        assert 'timestamp' in summary
        assert 'last_date' in summary
        assert 'current_price' in summary
        assert 'current_signal' in summary
        assert 'position_direction' in summary
        assert 'trade_action' in summary
        assert 'stop_loss' in summary
        assert 'position_sizes' in summary
        assert 'recent_history' in summary

    def test_long_position_detected(self, sample_df):
        """Test long position is correctly identified."""
        summary = get_trading_summary(sample_df, 'AAPL')

        assert summary['current_signal'] == 1
        assert summary['position_direction'] == 'LONG'

    def test_short_position_detected(self, sample_df_short):
        """Test short position is correctly identified."""
        summary = get_trading_summary(sample_df_short, 'MSFT')

        assert summary['current_signal'] == -1
        assert summary['position_direction'] == 'SHORT'

    def test_flat_position_detected(self, sample_df_flat):
        """Test flat position is correctly identified."""
        summary = get_trading_summary(sample_df_flat, 'TEST')

        assert summary['current_signal'] == 0
        assert summary['position_direction'] == 'FLAT'
        assert summary['trade_action'] == 'STAY FLAT'

    def test_signal_change_detected(self, sample_df):
        """Test signal change is detected."""
        summary = get_trading_summary(sample_df, 'AAPL')

        # Last bar has signal change (0 -> 1)
        assert summary['signal_changed'] is True
        assert 'ENTER LONG' in summary['trade_action']

    def test_hold_action_no_change(self, sample_df_short):
        """Test hold action when signal doesn't change."""
        summary = get_trading_summary(sample_df_short, 'MSFT')

        # Signal is -1 for multiple bars
        assert summary['signal_changed'] is False
        assert summary['trade_action'] == 'HOLD SHORT'

    def test_stop_loss_included(self, sample_df):
        """Test stop loss is included in summary."""
        summary = get_trading_summary(sample_df, 'AAPL')

        assert summary['stop_loss'] is not None
        assert summary['stop_loss'] > 0

    def test_risk_percentage_calculated(self, sample_df):
        """Test risk percentage is calculated for long position."""
        summary = get_trading_summary(sample_df, 'AAPL')

        # For long position with stop below price
        assert summary['risk_pct'] is not None
        assert summary['risk_pct'] > 0

    def test_position_sizes_included(self, sample_df):
        """Test position sizes are included."""
        summary = get_trading_summary(sample_df, 'AAPL')

        assert len(summary['position_sizes']) > 0
        assert 'equal_weight' in summary['position_sizes']
        assert 'fixed' in summary['position_sizes']

    def test_recent_history_length(self, sample_df):
        """Test recent history has correct length."""
        lookback = 5
        summary = get_trading_summary(sample_df, 'AAPL', lookback=lookback)

        assert len(summary['recent_history']) == lookback

    def test_custom_column_names(self, sample_df):
        """Test with custom column names."""
        df = sample_df.rename(columns={
            'hybrid_signal': 'my_signal',
            'stop_loss': 'my_stop',
        })

        summary = get_trading_summary(
            df, 'AAPL',
            signal_col='my_signal',
            stop_loss_col='my_stop'
        )

        assert summary['current_signal'] == 1

    def test_empty_dataframe_raises(self):
        """Test empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            get_trading_summary(pd.DataFrame(), 'AAPL')

    def test_missing_columns_raises(self, sample_df):
        """Test missing required columns raises ValueError."""
        df = sample_df.drop(columns=['hybrid_signal'])

        with pytest.raises(ValueError, match="Missing required columns"):
            get_trading_summary(df, 'AAPL')

    def test_already_indexed_dataframe(self, sample_df):
        """Test DataFrame already indexed by date works."""
        df = sample_df.set_index('date')
        summary = get_trading_summary(df, 'AAPL')

        assert summary['ticker'] == 'AAPL'
        assert summary['current_price'] > 0

    def test_nan_signal_treated_as_flat(self):
        """Test NaN signal is treated as flat."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [100, 101, 102, 103, 104],
            'hybrid_signal': [1, 1, 0, np.nan, np.nan],
            'stop_loss': [95, 96, 97, 98, 99],
        })

        summary = get_trading_summary(df, 'TEST')

        assert summary['current_signal'] == 0
        assert summary['position_direction'] == 'FLAT'


# =============================================================================
# Test Trade Actions
# =============================================================================

class TestTradeActions:
    """Tests for trade action detection."""

    def test_enter_long(self):
        """Test ENTER LONG action."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [100, 101, 102],
            'hybrid_signal': [0, 0, 1],  # Enter long on last bar
        })

        summary = get_trading_summary(df, 'TEST')
        assert summary['trade_action'] == 'ENTER LONG'
        assert summary['signal_changed'] is True

    def test_enter_short(self):
        """Test ENTER SHORT action."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [100, 101, 102],
            'hybrid_signal': [0, 0, -1],  # Enter short on last bar
        })

        summary = get_trading_summary(df, 'TEST')
        assert summary['trade_action'] == 'ENTER SHORT'

    def test_exit_long(self):
        """Test EXIT LONG action."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [100, 101, 102],
            'hybrid_signal': [1, 1, 0],  # Exit long on last bar
        })

        summary = get_trading_summary(df, 'TEST')
        assert summary['trade_action'] == 'EXIT LONG'

    def test_exit_short(self):
        """Test EXIT SHORT action."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [100, 101, 102],
            'hybrid_signal': [-1, -1, 0],  # Exit short on last bar
        })

        summary = get_trading_summary(df, 'TEST')
        assert summary['trade_action'] == 'EXIT SHORT'

    def test_flip_long_to_short(self):
        """Test FLIP LONG to SHORT action."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [100, 101, 102],
            'hybrid_signal': [1, 1, -1],  # Flip on last bar
        })

        summary = get_trading_summary(df, 'TEST')
        assert 'FLIP' in summary['trade_action']
        assert 'SHORT' in summary['trade_action']

    def test_flip_short_to_long(self):
        """Test FLIP SHORT to LONG action."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [100, 101, 102],
            'hybrid_signal': [-1, -1, 1],  # Flip on last bar
        })

        summary = get_trading_summary(df, 'TEST')
        assert 'FLIP' in summary['trade_action']
        assert 'LONG' in summary['trade_action']

    def test_hold_long(self):
        """Test HOLD LONG action."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [100, 101, 102],
            'hybrid_signal': [1, 1, 1],  # No change
        })

        summary = get_trading_summary(df, 'TEST')
        assert summary['trade_action'] == 'HOLD LONG'
        assert summary['signal_changed'] is False

    def test_stay_flat(self):
        """Test STAY FLAT action."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [100, 101, 102],
            'hybrid_signal': [0, 0, 0],  # No change
        })

        summary = get_trading_summary(df, 'TEST')
        assert summary['trade_action'] == 'STAY FLAT'


# =============================================================================
# Test Multi-Symbol Summary
# =============================================================================

class TestMultiSymbolSummary:
    """Tests for get_multi_symbol_summary function."""

    def test_basic_multi_summary(self, multi_symbol_data):
        """Test basic multi-symbol summary generation."""
        summaries = get_multi_symbol_summary(multi_symbol_data)

        assert len(summaries) == 2
        tickers = [s['ticker'] for s in summaries]
        assert 'AAPL' in tickers
        assert 'MSFT' in tickers

    def test_different_positions(self, multi_symbol_data):
        """Test different positions are correctly identified."""
        summaries = get_multi_symbol_summary(multi_symbol_data)

        aapl = next(s for s in summaries if s['ticker'] == 'AAPL')
        msft = next(s for s in summaries if s['ticker'] == 'MSFT')

        assert aapl['position_direction'] == 'LONG'
        assert msft['position_direction'] == 'SHORT'

    def test_error_handling(self):
        """Test error handling for invalid DataFrames."""
        data = {
            'GOOD': pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=5),
                'close': [100, 101, 102, 103, 104],
                'hybrid_signal': [0, 0, 1, 1, 1],
            }),
            'BAD': pd.DataFrame({'foo': [1, 2, 3]}),  # Missing columns
        }

        summaries = get_multi_symbol_summary(data)

        good = next(s for s in summaries if s['ticker'] == 'GOOD')
        bad = next(s for s in summaries if s['ticker'] == 'BAD')

        assert 'error' not in good
        assert 'error' in bad


# =============================================================================
# Test Dashboard Creation
# =============================================================================

class TestCreateDashboard:
    """Tests for create_trading_dashboard function."""

    def test_dashboard_creation(self, multi_symbol_data):
        """Test dashboard DataFrame is created correctly."""
        dashboard = create_trading_dashboard(multi_symbol_data)

        assert isinstance(dashboard, pd.DataFrame)
        assert len(dashboard) == 2
        assert 'ticker' in dashboard.columns
        assert 'price' in dashboard.columns
        assert 'signal' in dashboard.columns
        assert 'position' in dashboard.columns
        assert 'action' in dashboard.columns

    def test_dashboard_columns(self, multi_symbol_data):
        """Test dashboard has expected columns."""
        dashboard = create_trading_dashboard(multi_symbol_data)

        expected_cols = ['ticker', 'last_date', 'price', 'signal',
                        'position', 'action', 'signal_changed', 'stop_loss']
        for col in expected_cols:
            assert col in dashboard.columns

    def test_dashboard_position_sizes(self, sample_df):
        """Test dashboard includes position size columns."""
        data = {'AAPL': sample_df}
        dashboard = create_trading_dashboard(data)

        # Should have shares columns from the fixture
        shares_cols = [c for c in dashboard.columns if c.startswith('shares_')]
        assert len(shares_cols) > 0


# =============================================================================
# Test Print Functions
# =============================================================================

class TestPrintFunctions:
    """Tests for print functions (mainly testing they don't crash)."""

    def test_print_single_summary(self, sample_df, capsys):
        """Test print_trading_summary runs without error."""
        summary = get_trading_summary(sample_df, 'AAPL')
        print_trading_summary(summary)

        captured = capsys.readouterr()
        assert 'AAPL' in captured.out
        assert 'TRADING SUMMARY' in captured.out

    def test_print_summary_with_error(self, capsys):
        """Test print_trading_summary handles error dict."""
        summary = {'ticker': 'BAD', 'error': 'Test error'}
        print_trading_summary(summary)

        captured = capsys.readouterr()
        assert 'ERROR' in captured.out

    def test_print_multi_summary(self, multi_symbol_data, capsys):
        """Test print_multi_symbol_summary runs without error."""
        summaries = get_multi_symbol_summary(multi_symbol_data)
        print_multi_symbol_summary(summaries)

        captured = capsys.readouterr()
        assert 'TRADING READINESS REPORT' in captured.out
        assert 'AAPL' in captured.out
        assert 'MSFT' in captured.out

    def test_print_detailed_mode(self, sample_df, capsys):
        """Test detailed print mode."""
        summary = get_trading_summary(sample_df, 'AAPL')
        print_trading_summary(summary, detailed=True)

        captured = capsys.readouterr()
        assert 'RECENT HISTORY' in captured.out


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-01')],
            'close': [100.0],
            'hybrid_signal': [1],
        })

        summary = get_trading_summary(df, 'TEST')
        assert summary['current_signal'] == 1

    def test_all_nan_stop_loss(self):
        """Test with all NaN stop loss values."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [100, 101, 102, 103, 104],
            'hybrid_signal': [1, 1, 1, 1, 1],
            'stop_loss': [np.nan] * 5,
        })

        summary = get_trading_summary(df, 'TEST')
        assert summary['stop_loss'] is None
        assert summary['risk_pct'] is None

    def test_no_position_sizing_columns(self):
        """Test with no position sizing columns."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [100, 101, 102, 103, 104],
            'hybrid_signal': [1, 1, 1, 1, 1],
        })

        summary = get_trading_summary(df, 'TEST')
        assert summary['position_sizes'] == {}

    def test_custom_position_cols(self):
        """Test with custom position sizing columns."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [100, 101, 102, 103, 104],
            'hybrid_signal': [1, 1, 1, 1, 1],
            'my_shares': [100, 100, 100, 100, 100],
        })

        summary = get_trading_summary(
            df, 'TEST',
            position_cols={'my_method': 'my_shares'}
        )

        assert 'my_method' in summary['position_sizes']
        assert summary['position_sizes']['my_method'] == 100

    def test_very_large_dataframe(self):
        """Test with large DataFrame for performance."""
        n_days = 10000
        df = pd.DataFrame({
            'date': pd.date_range('2000-01-01', periods=n_days),
            'close': 100 + np.cumsum(np.random.randn(n_days) * 0.5),
            'hybrid_signal': np.random.choice([0, 1, -1], n_days),
        })

        # Should complete without timeout
        summary = get_trading_summary(df, 'TEST', lookback=5)
        assert len(summary['recent_history']) == 5
