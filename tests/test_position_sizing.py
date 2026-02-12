"""
Tests for the position_sizing module.

Covers:
- Constructor validation
- Division by zero edge cases
- Position sizing calculations (long/short)
- Risk appetite calculations
- Float equality edge cases
"""
import pytest
import pandas as pd
import numpy as np
from algoshort.position_sizing import PositionSizing, get_signal_column_names


class TestPositionSizingConstructor:
    """Tests for PositionSizing constructor validation."""

    def test_valid_initialization(self):
        """Test valid parameter initialization."""
        sizer = PositionSizing(
            tolerance=-0.1,
            mn=0.01,
            mx=0.05,
            equal_weight=0.1,
            avg=0.02,
            lot=100,
            initial_capital=100000
        )
        assert sizer.tolerance == -0.1
        assert sizer.mn == 0.01
        assert sizer.mx == 0.05
        assert sizer.lot == 100

    def test_tolerance_must_be_negative(self):
        """Test that tolerance must be negative."""
        with pytest.raises(ValueError, match="tolerance must be negative"):
            PositionSizing(
                tolerance=0.1,  # Invalid: positive
                mn=0.01, mx=0.05, equal_weight=0.1,
                avg=0.02, lot=100
            )

    def test_tolerance_zero_rejected(self):
        """Test that tolerance=0 is rejected."""
        with pytest.raises(ValueError, match="tolerance must be negative"):
            PositionSizing(
                tolerance=0,  # Invalid: zero
                mn=0.01, mx=0.05, equal_weight=0.1,
                avg=0.02, lot=100
            )

    def test_mn_must_be_positive(self):
        """Test that mn must be positive."""
        with pytest.raises(ValueError, match="mn and mx must be positive"):
            PositionSizing(
                tolerance=-0.1,
                mn=0,  # Invalid: zero
                mx=0.05, equal_weight=0.1,
                avg=0.02, lot=100
            )

    def test_mx_must_be_positive(self):
        """Test that mx must be positive."""
        with pytest.raises(ValueError, match="mn and mx must be positive"):
            PositionSizing(
                tolerance=-0.1,
                mn=0.01,
                mx=-0.05,  # Invalid: negative
                equal_weight=0.1,
                avg=0.02, lot=100
            )

    def test_mn_must_be_less_than_mx(self):
        """Test that mn <= mx."""
        with pytest.raises(ValueError, match="mn must be <= mx"):
            PositionSizing(
                tolerance=-0.1,
                mn=0.10,  # Invalid: greater than mx
                mx=0.05,
                equal_weight=0.1,
                avg=0.02, lot=100
            )

    def test_equal_weight_must_be_in_valid_range(self):
        """Test that equal_weight must be in (0, 1]."""
        with pytest.raises(ValueError, match="equal_weight must be in"):
            PositionSizing(
                tolerance=-0.1,
                mn=0.01, mx=0.05,
                equal_weight=0,  # Invalid: zero
                avg=0.02, lot=100
            )

        with pytest.raises(ValueError, match="equal_weight must be in"):
            PositionSizing(
                tolerance=-0.1,
                mn=0.01, mx=0.05,
                equal_weight=1.5,  # Invalid: > 1
                avg=0.02, lot=100
            )

    def test_lot_must_be_positive(self):
        """Test that lot must be positive."""
        with pytest.raises(ValueError, match="lot must be positive"):
            PositionSizing(
                tolerance=-0.1,
                mn=0.01, mx=0.05, equal_weight=0.1,
                avg=0.02,
                lot=0  # Invalid: zero
            )

    def test_initial_capital_must_be_positive(self):
        """Test that initial_capital must be positive."""
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            PositionSizing(
                tolerance=-0.1,
                mn=0.01, mx=0.05, equal_weight=0.1,
                avg=0.02, lot=100,
                initial_capital=-10000  # Invalid: negative
            )


class TestEqtyRiskShares:
    """Tests for eqty_risk_shares method."""

    @pytest.fixture
    def sizer(self):
        """Create a valid PositionSizing instance."""
        return PositionSizing(
            tolerance=-0.1,
            mn=0.01,
            mx=0.05,
            equal_weight=0.1,
            avg=0.02,
            lot=100,
            initial_capital=100000
        )

    def test_basic_long_position(self, sizer):
        """Test basic long position calculation."""
        # Long: price=100, stop=95, risk per share = 5
        # Budget: 100000 * 0.02 * 1 = 2000
        # Shares: 2000 / (5 * 100) * 100 = 400
        shares = sizer.eqty_risk_shares(
            px=100, sl=95, eqty=100000, risk=0.02, fx=1, lot=100
        )
        assert shares == 400

    def test_basic_short_position(self, sizer):
        """Test basic short position calculation (stop above price)."""
        # Short: price=100, stop=105, risk per share = abs(-5) = 5
        # Same calculation as long due to abs()
        shares = sizer.eqty_risk_shares(
            px=100, sl=105, eqty=100000, risk=0.02, fx=1, lot=100
        )
        assert shares == 400

    def test_zero_price_returns_zero(self, sizer):
        """Test that zero price returns 0 shares."""
        shares = sizer.eqty_risk_shares(
            px=0, sl=95, eqty=100000, risk=0.02, fx=1, lot=100
        )
        assert shares == 0

    def test_negative_price_returns_zero(self, sizer):
        """Test that negative price returns 0 shares."""
        shares = sizer.eqty_risk_shares(
            px=-100, sl=95, eqty=100000, risk=0.02, fx=1, lot=100
        )
        assert shares == 0

    def test_zero_lot_returns_zero(self, sizer):
        """Test that zero lot size returns 0 shares."""
        shares = sizer.eqty_risk_shares(
            px=100, sl=95, eqty=100000, risk=0.02, fx=1, lot=0
        )
        assert shares == 0

    def test_negative_equity_returns_zero(self, sizer):
        """Test that negative equity returns 0 shares."""
        shares = sizer.eqty_risk_shares(
            px=100, sl=95, eqty=-10000, risk=0.02, fx=1, lot=100
        )
        assert shares == 0

    def test_stop_equals_price_returns_zero(self, sizer):
        """Test that stop loss equal to price returns 0 shares."""
        shares = sizer.eqty_risk_shares(
            px=100, sl=100, eqty=100000, risk=0.02, fx=1, lot=100
        )
        assert shares == 0

    def test_stop_very_close_to_price_returns_zero(self, sizer):
        """Test stop loss very close to price (near-zero risk)."""
        shares = sizer.eqty_risk_shares(
            px=100.0, sl=100.0000000001, eqty=100000, risk=0.02, fx=1, lot=100
        )
        assert shares == 0

    def test_fx_rate_multiplier(self, sizer):
        """Test FX rate multiplier effect."""
        shares_no_fx = sizer.eqty_risk_shares(
            px=100, sl=95, eqty=100000, risk=0.02, fx=1, lot=100
        )
        shares_with_fx = sizer.eqty_risk_shares(
            px=100, sl=95, eqty=100000, risk=0.02, fx=2, lot=100
        )
        assert shares_with_fx == shares_no_fx * 2

    def test_negative_fx_treated_as_one(self, sizer):
        """Test that negative FX rate is treated as 1."""
        shares_neg_fx = sizer.eqty_risk_shares(
            px=100, sl=95, eqty=100000, risk=0.02, fx=-1, lot=100
        )
        shares_fx_one = sizer.eqty_risk_shares(
            px=100, sl=95, eqty=100000, risk=0.02, fx=1, lot=100
        )
        assert shares_neg_fx == shares_fx_one

    def test_lot_rounding(self, sizer):
        """Test that shares are properly rounded to lot size."""
        # With lot=100, result should always be multiple of 100
        shares = sizer.eqty_risk_shares(
            px=100, sl=99, eqty=100000, risk=0.02, fx=1, lot=100
        )
        assert shares % 100 == 0

    def test_small_budget_rounds_to_zero(self, sizer):
        """Test that small budget rounds to zero lots."""
        # Very small equity -> budget too small for even 1 lot
        shares = sizer.eqty_risk_shares(
            px=100, sl=95, eqty=100, risk=0.02, fx=1, lot=100
        )
        assert shares == 0


class TestRiskAppetite:
    """Tests for risk_appetite method."""

    @pytest.fixture
    def sizer(self):
        """Create a valid PositionSizing instance."""
        return PositionSizing(
            tolerance=-0.1,
            mn=0.01,
            mx=0.05,
            equal_weight=0.1,
            avg=0.02,
            lot=100,
            initial_capital=100000
        )

    def test_basic_risk_appetite(self, sizer):
        """Test basic risk appetite calculation."""
        equity = pd.Series([100000, 101000, 102000, 103000, 104000])
        result = sizer.risk_appetite(
            eqty=equity,
            tolerance=-0.1,
            mn=0.01,
            mx=0.05,
            span=3,
            shape=0
        )
        assert len(result) == len(equity)
        assert result.iloc[-1] >= 0.01
        assert result.iloc[-1] <= 0.05

    def test_risk_appetite_with_drawdown(self, sizer):
        """Test risk appetite decreases during drawdown."""
        # Equity with drawdown
        equity = pd.Series([100000, 105000, 100000, 95000, 90000])
        result = sizer.risk_appetite(
            eqty=equity,
            tolerance=-0.1,
            mn=0.01,
            mx=0.05,
            span=3,
            shape=0
        )
        # During drawdown, risk appetite should decrease
        assert result.iloc[-1] < result.iloc[1]

    def test_tolerance_zero_uses_default(self, sizer):
        """Test that tolerance=0 uses default value instead of crashing."""
        equity = pd.Series([100000, 101000, 102000])
        # Should not raise ZeroDivisionError
        result = sizer.risk_appetite(
            eqty=equity,
            tolerance=0,  # Would cause division by zero
            mn=0.01,
            mx=0.05,
            span=3,
            shape=0
        )
        assert not result.isna().any()

    def test_mn_zero_uses_default(self, sizer):
        """Test that mn=0 uses default value."""
        equity = pd.Series([100000, 101000, 102000])
        result = sizer.risk_appetite(
            eqty=equity,
            tolerance=-0.1,
            mn=0,  # Would cause issues
            mx=0.05,
            span=3,
            shape=1
        )
        assert not result.isna().any()

    def test_mx_zero_uses_default(self, sizer):
        """Test that mx=0 uses default value."""
        equity = pd.Series([100000, 101000, 102000])
        result = sizer.risk_appetite(
            eqty=equity,
            tolerance=-0.1,
            mn=0.01,
            mx=0,  # Would cause issues
            span=3,
            shape=-1
        )
        assert not result.isna().any()

    def test_zero_equity_handled(self, sizer):
        """Test that zero equity values don't cause division by zero."""
        equity = pd.Series([0, 100000, 101000])
        result = sizer.risk_appetite(
            eqty=equity,
            tolerance=-0.1,
            mn=0.01,
            mx=0.05,
            span=3,
            shape=0
        )
        assert not result.isna().all()

    def test_convex_shape(self, sizer):
        """Test convex shape (shape=1)."""
        equity = pd.Series([100000] * 10)
        result = sizer.risk_appetite(
            eqty=equity,
            tolerance=-0.1,
            mn=0.01,
            mx=0.05,
            span=3,
            shape=1
        )
        assert len(result) == 10

    def test_concave_shape(self, sizer):
        """Test concave shape (shape=-1)."""
        equity = pd.Series([100000] * 10)
        result = sizer.risk_appetite(
            eqty=equity,
            tolerance=-0.1,
            mn=0.01,
            mx=0.05,
            span=3,
            shape=-1
        )
        assert len(result) == 10

    def test_empty_equity_series(self, sizer):
        """Test handling of empty equity series."""
        equity = pd.Series([], dtype=float)
        result = sizer.risk_appetite(
            eqty=equity,
            tolerance=-0.1,
            mn=0.01,
            mx=0.05,
            span=3,
            shape=0
        )
        assert len(result) == 0


class TestGetSignalColumnNames:
    """Tests for get_signal_column_names helper function."""

    def test_basic_column_names(self):
        """Test basic column name generation."""
        cols = get_signal_column_names("rsi2")
        assert cols["signal"] == "rsi2"
        assert cols["daily_chg"] == "rsi2_chg1D_fx"
        assert cols["sl"] == "rsi2_stop_loss"
        assert cols["close"] == "close"

    def test_custom_suffixes(self):
        """Test custom suffix parameters."""
        cols = get_signal_column_names(
            signal="ma_cross",
            chg_suffix="_daily_change",
            sl_suffix="_sl",
            close_col="adj_close"
        )
        assert cols["daily_chg"] == "ma_cross_daily_change"
        assert cols["sl"] == "ma_cross_sl"
        assert cols["close"] == "adj_close"

    def test_empty_signal_raises(self):
        """Test that empty signal raises ValueError."""
        with pytest.raises(ValueError, match="Signal must be a non-empty string"):
            get_signal_column_names("")

    def test_none_signal_raises(self):
        """Test that None signal raises ValueError."""
        with pytest.raises(ValueError, match="Signal must be a non-empty string"):
            get_signal_column_names(None)


class TestCalculateSharesEdgeCases:
    """Integration tests for calculate_shares with edge cases."""

    @pytest.fixture
    def sizer(self):
        """Create a valid PositionSizing instance."""
        return PositionSizing(
            tolerance=-0.1,
            mn=0.01,
            mx=0.05,
            equal_weight=0.1,
            avg=0.02,
            lot=100,
            initial_capital=100000
        )

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        n = 20
        np.random.seed(42)
        df = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n)),
            'stop_loss': 95 + np.cumsum(np.random.randn(n) * 0.5),
            'chg1D_fx': np.random.randn(n) * 0.5,
            'signal': [0, 1, 1, 1, 0, -1, -1, -1, 0, 1] * 2
        })
        return df

    def test_calculate_shares_basic(self, sizer, sample_df):
        """Test basic calculate_shares execution."""
        result = sizer.calculate_shares(
            df=sample_df.copy(),
            signal='signal',
            daily_chg='chg1D_fx',
            sl='stop_loss',
            close='close'
        )
        # Should have added equity columns
        assert 'signal_equity_equal' in result.columns
        assert 'signal_equity_constant' in result.columns

    def test_calculate_shares_with_price_equals_stop(self, sizer):
        """Test when price equals stop loss (edge case)."""
        df = pd.DataFrame({
            'close': [100.0, 100.0, 100.0],
            'stop_loss': [100.0, 100.0, 100.0],  # Same as close!
            'chg1D_fx': [0.0, 0.5, -0.5],
            'signal': [0, 1, 1]
        })
        # Should not crash
        result = sizer.calculate_shares(
            df=df.copy(),
            signal='signal',
            daily_chg='chg1D_fx',
            sl='stop_loss',
            close='close'
        )
        assert result is not None
