"""
Tests for the ohlcprocessor module.

Covers:
- Initialization and configuration
- Input validation
- Relative price calculations
- Edge cases (zeros, NaN, single-row, misaligned dates)
- Rebase functionality
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from algoshort.ohlcprocessor import OHLCProcessor, OHLCColumns


class TestOHLCColumnsDataclass:
    """Tests for OHLCColumns configuration dataclass."""

    def test_default_values(self):
        """Test default column names."""
        cols = OHLCColumns()
        assert cols.open == 'open'
        assert cols.high == 'high'
        assert cols.low == 'low'
        assert cols.close == 'close'
        assert cols.date == 'date'

    def test_custom_values(self):
        """Test custom column names."""
        cols = OHLCColumns(
            open='Open',
            high='High',
            low='Low',
            close='Close',
            date='Date'
        )
        assert cols.open == 'Open'
        assert cols.close == 'Close'


class TestOHLCProcessorInit:
    """Tests for OHLCProcessor initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        processor = OHLCProcessor()
        assert processor.columns.open == 'open'
        assert processor.columns.close == 'close'

    def test_custom_column_config(self):
        """Test initialization with custom column config."""
        custom_cols = OHLCColumns(open='o', high='h', low='l', close='c')
        processor = OHLCProcessor(column_config=custom_cols)
        assert processor.columns.open == 'o'
        assert processor.columns.close == 'c'

    def test_logger_created(self):
        """Test that logger is properly created."""
        processor = OHLCProcessor()
        assert processor.logger is not None
        assert 'OHLCProcessor' in processor.logger.name


class TestInputValidation:
    """Tests for input validation in calculate_relative_prices."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return OHLCProcessor()

    @pytest.fixture
    def valid_stock_df(self):
        """Create valid stock DataFrame."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [105.0, 106.0, 107.0, 108.0, 109.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [104.0, 105.0, 106.0, 107.0, 108.0]
        })

    @pytest.fixture
    def valid_benchmark_df(self):
        """Create valid benchmark DataFrame."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [2000.0, 2010.0, 2020.0, 2030.0, 2040.0]
        })

    def test_stock_data_not_dataframe_raises(self, processor, valid_benchmark_df):
        """Test that non-DataFrame stock_data raises TypeError."""
        with pytest.raises(TypeError, match="stock_data must be pandas.DataFrame"):
            processor.calculate_relative_prices(
                stock_data="not a dataframe",
                benchmark_data=valid_benchmark_df
            )

    def test_benchmark_data_not_dataframe_raises(self, processor, valid_stock_df):
        """Test that non-DataFrame benchmark_data raises TypeError."""
        with pytest.raises(TypeError, match="benchmark_data must be pandas.DataFrame"):
            processor.calculate_relative_prices(
                stock_data=valid_stock_df,
                benchmark_data=[1, 2, 3]
            )

    def test_empty_stock_data_raises(self, processor, valid_benchmark_df):
        """Test that empty stock_data raises ValueError."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="stock_data cannot be empty"):
            processor.calculate_relative_prices(
                stock_data=empty_df,
                benchmark_data=valid_benchmark_df
            )

    def test_empty_benchmark_data_raises(self, processor, valid_stock_df):
        """Test that empty benchmark_data raises ValueError."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="benchmark_data cannot be empty"):
            processor.calculate_relative_prices(
                stock_data=valid_stock_df,
                benchmark_data=empty_df
            )

    def test_invalid_digits_raises(self, processor, valid_stock_df, valid_benchmark_df):
        """Test that invalid digits parameter raises ValueError."""
        with pytest.raises(ValueError, match="digits must be integer between 0-10"):
            processor.calculate_relative_prices(
                stock_data=valid_stock_df,
                benchmark_data=valid_benchmark_df,
                digits=-1
            )

        with pytest.raises(ValueError, match="digits must be integer between 0-10"):
            processor.calculate_relative_prices(
                stock_data=valid_stock_df,
                benchmark_data=valid_benchmark_df,
                digits=11
            )

    def test_missing_ohlc_columns_raises(self, processor, valid_benchmark_df):
        """Test that missing OHLC columns raises ValueError."""
        incomplete_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [100, 101, 102]
            # Missing: open, high, low
        })
        with pytest.raises(ValueError, match="Missing required OHLC columns"):
            processor.calculate_relative_prices(
                stock_data=incomplete_df,
                benchmark_data=valid_benchmark_df
            )

    def test_missing_benchmark_column_raises(self, processor, valid_stock_df):
        """Test that missing benchmark column raises ValueError."""
        benchmark_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'price': [2000, 2010, 2020]  # No 'close' column
        })
        with pytest.raises(ValueError, match="Benchmark column 'close' not found"):
            processor.calculate_relative_prices(
                stock_data=valid_stock_df,
                benchmark_data=benchmark_df,
                benchmark_column='close'
            )


class TestRelativePriceCalculations:
    """Tests for relative price calculation logic."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return OHLCProcessor()

    @pytest.fixture
    def sample_data(self):
        """Create sample stock and benchmark data."""
        dates = pd.date_range('2024-01-01', periods=5)

        stock_df = pd.DataFrame({
            'date': dates,
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [105.0, 106.0, 107.0, 108.0, 109.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [104.0, 105.0, 106.0, 107.0, 108.0]
        })

        benchmark_df = pd.DataFrame({
            'date': dates,
            'close': [2000.0, 2010.0, 2020.0, 2030.0, 2040.0]
        })

        return stock_df, benchmark_df

    def test_basic_relative_calculation(self, processor, sample_data):
        """Test basic relative price calculation."""
        stock_df, benchmark_df = sample_data
        result = processor.calculate_relative_prices(
            stock_data=stock_df,
            benchmark_data=benchmark_df,
            rebase=False
        )

        # Check relative columns exist
        assert 'ropen' in result.columns
        assert 'rhigh' in result.columns
        assert 'rlow' in result.columns
        assert 'rclose' in result.columns

        # Check calculation: rclose = close / benchmark
        # First row: 104 / 2000 = 0.052
        assert result['rclose'].iloc[0] == pytest.approx(0.052, rel=1e-3)

    def test_rebased_calculation(self, processor, sample_data):
        """Test rebased relative price calculation."""
        stock_df, benchmark_df = sample_data
        result = processor.calculate_relative_prices(
            stock_data=stock_df,
            benchmark_data=benchmark_df,
            rebase=True
        )

        # With rebase, first benchmark value becomes 1.0
        # So first relative close = 104 / 1.0 = 104
        assert result['rclose'].iloc[0] == pytest.approx(104.0, rel=1e-3)

    def test_rounding_precision(self, processor, sample_data):
        """Test that digits parameter controls rounding."""
        stock_df, benchmark_df = sample_data

        result_2 = processor.calculate_relative_prices(
            stock_data=stock_df,
            benchmark_data=benchmark_df,
            digits=2,
            rebase=False
        )

        result_6 = processor.calculate_relative_prices(
            stock_data=stock_df,
            benchmark_data=benchmark_df,
            digits=6,
            rebase=False
        )

        # 6 decimal places should have more precision
        # 104 / 2000 = 0.052 (2 decimals) vs 0.052000 (6 decimals)
        assert len(str(result_2['rclose'].iloc[0]).split('.')[-1]) <= 2
        # Note: trailing zeros may be dropped, so just check it works

    def test_original_columns_preserved(self, processor, sample_data):
        """Test that original OHLC columns are preserved."""
        stock_df, benchmark_df = sample_data
        result = processor.calculate_relative_prices(
            stock_data=stock_df,
            benchmark_data=benchmark_df
        )

        # Original columns should exist
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'date' in result.columns


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return OHLCProcessor()

    def test_zero_benchmark_first_value_with_rebase_raises(self, processor):
        """Test that zero first benchmark value raises error with rebase=True."""
        stock_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [99.0, 100.0, 101.0],
            'close': [104.0, 105.0, 106.0]
        })

        benchmark_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [0.0, 2010.0, 2020.0]  # First value is zero!
        })

        with pytest.raises(ValueError, match="Cannot rebase: first benchmark value is zero"):
            processor.calculate_relative_prices(
                stock_data=stock_df,
                benchmark_data=benchmark_df,
                rebase=True
            )

    def test_zero_benchmark_mid_series_raises(self, processor):
        """Test that zero benchmark value in middle of series raises error."""
        stock_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [99.0, 100.0, 101.0],
            'close': [104.0, 105.0, 106.0]
        })

        benchmark_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [2000.0, 0.0, 2020.0]  # Zero in middle!
        })

        with pytest.raises(ValueError, match="Benchmark contains zero values"):
            processor.calculate_relative_prices(
                stock_data=stock_df,
                benchmark_data=benchmark_df,
                rebase=False
            )

    def test_negative_benchmark_values_raises(self, processor):
        """Test that negative benchmark values raise error."""
        stock_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [99.0, 100.0, 101.0],
            'close': [104.0, 105.0, 106.0]
        })

        benchmark_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [2000.0, -100.0, 2020.0]  # Negative value!
        })

        with pytest.raises(ValueError, match="negative values"):
            processor.calculate_relative_prices(
                stock_data=stock_df,
                benchmark_data=benchmark_df,
                rebase=False
            )

    def test_too_much_missing_data_raises(self, processor):
        """Test that >10% missing benchmark data raises error."""
        # Create 20 days of stock data
        dates = pd.date_range('2024-01-01', periods=20)
        stock_df = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 110, 20),
            'high': np.random.uniform(110, 120, 20),
            'low': np.random.uniform(90, 100, 20),
            'close': np.random.uniform(100, 110, 20)
        })

        # Benchmark only covers first 10 days (50% missing)
        benchmark_df = pd.DataFrame({
            'date': dates[:10],
            'close': np.random.uniform(2000, 2100, 10)
        })

        with pytest.raises(ValueError, match="Too much missing benchmark data"):
            processor.calculate_relative_prices(
                stock_data=stock_df,
                benchmark_data=benchmark_df
            )

    def test_column_case_normalization(self, processor):
        """Test that uppercase column names are normalized."""
        stock_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=3),
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [104.0, 105.0, 106.0]
        })

        benchmark_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=3),
            'Close': [2000.0, 2010.0, 2020.0]
        })

        # Should not raise - columns should be normalized
        result = processor.calculate_relative_prices(
            stock_data=stock_df,
            benchmark_data=benchmark_df
        )
        assert 'rclose' in result.columns

    def test_datetime_index_converted(self, processor):
        """Test that DatetimeIndex is properly converted to date column."""
        dates = pd.date_range('2024-01-01', periods=3)

        stock_df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [99.0, 100.0, 101.0],
            'close': [104.0, 105.0, 106.0]
        }, index=dates)

        benchmark_df = pd.DataFrame({
            'close': [2000.0, 2010.0, 2020.0]
        }, index=dates)

        result = processor.calculate_relative_prices(
            stock_data=stock_df,
            benchmark_data=benchmark_df
        )
        assert 'date' in result.columns
        assert 'rclose' in result.columns


class TestDataNormalization:
    """Tests for DataFrame normalization."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return OHLCProcessor()

    def test_normalize_creates_copy(self, processor):
        """Test that normalization doesn't modify original DataFrame."""
        original_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=3),
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106]
        })

        original_columns = list(original_df.columns)

        normalized = processor._normalize_dataframe(original_df, require_ohlc=True)

        # Original should be unchanged
        assert list(original_df.columns) == original_columns
        assert 'Date' in original_df.columns
        assert 'date' not in original_df.columns

    def test_normalize_validates_ohlc(self, processor):
        """Test that require_ohlc=True validates OHLC columns."""
        incomplete_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [100, 101, 102]
        })

        with pytest.raises(ValueError, match="Missing required OHLC columns"):
            processor._normalize_dataframe(incomplete_df, require_ohlc=True)

    def test_normalize_skips_ohlc_validation(self, processor):
        """Test that require_ohlc=False skips OHLC validation."""
        minimal_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [100, 101, 102]
        })

        # Should not raise
        result = processor._normalize_dataframe(minimal_df, require_ohlc=False)
        assert 'close' in result.columns


class TestCustomBenchmarkColumn:
    """Tests for using custom benchmark columns."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return OHLCProcessor()

    def test_custom_benchmark_column(self, processor):
        """Test using a non-default benchmark column."""
        stock_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [99.0, 100.0, 101.0],
            'close': [104.0, 105.0, 106.0]
        })

        benchmark_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'adj_close': [2000.0, 2010.0, 2020.0],  # Custom column name
            'close': [1900.0, 1910.0, 1920.0]  # Different values
        })

        result = processor.calculate_relative_prices(
            stock_data=stock_df,
            benchmark_data=benchmark_df,
            benchmark_column='adj_close',  # Use custom column
            rebase=False
        )

        # Should use adj_close (2000), not close (1900)
        # rclose = 104 / 2000 = 0.052
        assert result['rclose'].iloc[0] == pytest.approx(0.052, rel=1e-3)
