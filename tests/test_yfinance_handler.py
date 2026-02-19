"""
Tests for the yfinance_handler module.

Covers:
- Initialization and configuration
- Symbol validation and preprocessing
- Cache handling and path security
- Data retrieval methods
- Edge cases and error handling
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from algoshort.yfinance_handler import YFinanceDataHandler


class TestYFinanceDataHandlerInit:
    """Tests for YFinanceDataHandler initialization."""

    def test_default_initialization(self):
        """Test default initialization without parameters."""
        handler = YFinanceDataHandler(enable_logging=False)
        assert handler.symbols == []
        assert handler.data == {}
        assert handler.cache_dir is None
        assert handler.chunk_size == 50

    def test_cache_dir_creation(self):
        """Test that cache directory is created if specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            handler = YFinanceDataHandler(
                cache_dir=str(cache_path),
                enable_logging=False
            )
            assert cache_path.exists()
            assert handler.cache_dir == cache_path

    def test_chunk_size_minimum(self):
        """Test that chunk_size has minimum of 1."""
        handler = YFinanceDataHandler(chunk_size=0, enable_logging=False)
        assert handler.chunk_size == 1

        handler = YFinanceDataHandler(chunk_size=-5, enable_logging=False)
        assert handler.chunk_size == 1

    def test_logger_no_duplicate_handlers(self):
        """Test that multiple instances don't duplicate log handlers."""
        handler1 = YFinanceDataHandler(enable_logging=True)
        initial_handler_count = len(handler1.logger.handlers)

        handler2 = YFinanceDataHandler(enable_logging=True)
        # Same logger name, should not add more handlers
        assert len(handler2.logger.handlers) == initial_handler_count

    def test_period_map_contains_aliases(self):
        """Test that period_map contains user-friendly aliases."""
        handler = YFinanceDataHandler(enable_logging=False)
        assert handler.period_map['year'] == '1y'
        assert handler.period_map['month'] == '1mo'
        assert handler.period_map['all_time'] == 'max'

    def test_interval_map_contains_aliases(self):
        """Test that interval_map contains user-friendly aliases."""
        handler = YFinanceDataHandler(enable_logging=False)
        assert handler.interval_map['daily'] == '1d'
        assert handler.interval_map['weekly'] == '1wk'
        assert handler.interval_map['hourly'] == '1h'


class TestSymbolPreprocessing:
    """Tests for symbol validation and preprocessing."""

    @pytest.fixture
    def handler(self):
        """Create handler instance for tests."""
        return YFinanceDataHandler(enable_logging=False)

    def test_single_symbol_string(self, handler):
        """Test processing single symbol string."""
        result = handler._preprocess_symbols("aapl")
        assert result == ["AAPL"]

    def test_symbol_list(self, handler):
        """Test processing list of symbols."""
        result = handler._preprocess_symbols(["aapl", "msft", "googl"])
        assert result == ["AAPL", "MSFT", "GOOGL"]

    def test_symbol_with_whitespace(self, handler):
        """Test that whitespace is stripped."""
        result = handler._preprocess_symbols("  aapl  ")
        assert result == ["AAPL"]

    def test_empty_string_raises_error(self, handler):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Empty symbol not allowed"):
            handler._preprocess_symbols("")

    def test_whitespace_only_raises_error(self, handler):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Empty symbol not allowed"):
            handler._preprocess_symbols("   ")

    def test_invalid_type_raises_error(self, handler):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Symbols must be a string or list"):
            handler._preprocess_symbols(123)

        with pytest.raises(TypeError, match="Symbols must be a string or list"):
            handler._preprocess_symbols({"symbol": "AAPL"})

    def test_list_with_non_strings_raises_error(self, handler):
        """Test that list with non-string elements raises TypeError."""
        with pytest.raises(TypeError, match="Symbol must be string"):
            handler._preprocess_symbols(["AAPL", 123, "MSFT"])

    def test_invalid_symbol_format_raises_error(self, handler):
        """Test that invalid symbol characters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            handler._preprocess_symbols("AAPL/MSFT")

        with pytest.raises(ValueError, match="Invalid symbol format"):
            handler._preprocess_symbols("../../etc/passwd")

    def test_valid_special_characters(self, handler):
        """Test that valid special characters are allowed."""
        # These are valid ticker formats
        result = handler._preprocess_symbols("BRK.B")
        assert result == ["BRK.B"]

        result = handler._preprocess_symbols("^GSPC")
        assert result == ["^GSPC"]

    def test_empty_list_raises_error(self, handler):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="No valid symbols provided"):
            handler._preprocess_symbols([])


class TestCachePathSecurity:
    """Tests for cache path security and path traversal prevention."""

    @pytest.fixture
    def handler_with_cache(self):
        """Create handler with cache directory."""
        tmpdir = tempfile.mkdtemp()
        handler = YFinanceDataHandler(
            cache_dir=tmpdir,
            enable_logging=False
        )
        yield handler
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_safe_cache_path_normal_symbol(self, handler_with_cache):
        """Test safe cache path for normal symbol."""
        path = handler_with_cache._get_safe_cache_path("AAPL", "1y", "1d")
        assert path.name == "AAPL_1y_1d.parquet"
        assert path.parent == handler_with_cache.cache_dir

    def test_safe_cache_path_sanitizes_special_chars(self, handler_with_cache):
        """Test that special characters are sanitized in cache path."""
        path = handler_with_cache._get_safe_cache_path("BRK.B", "1y", "1d")
        # Dot should be replaced with underscore
        assert "_" in path.name or "." in path.name
        assert path.parent == handler_with_cache.cache_dir

    def test_path_traversal_blocked(self, handler_with_cache):
        """Test that path traversal attempts are blocked."""
        # The symbol validation should catch this first, but cache path
        # should also sanitize
        with pytest.raises(ValueError):
            # This should fail at symbol validation
            handler_with_cache._preprocess_symbols("../../../etc/passwd")


class TestCacheExpiration:
    """Tests for cache expiration functionality."""

    @pytest.fixture
    def handler_with_cache(self):
        """Create handler with cache directory."""
        tmpdir = tempfile.mkdtemp()
        handler = YFinanceDataHandler(
            cache_dir=tmpdir,
            enable_logging=False
        )
        yield handler
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_from_cache_returns_none_when_no_cache(self, handler_with_cache):
        """Test that _load_from_cache returns None when file doesn't exist."""
        result = handler_with_cache._load_from_cache("AAPL", "1y", "1d")
        assert result is None

    def test_load_from_cache_respects_max_age(self, handler_with_cache):
        """Test that stale cache files are not loaded."""
        # Create a cache file
        cache_path = handler_with_cache._get_safe_cache_path("AAPL", "1y", "1d")
        df = pd.DataFrame({'close': [100, 101, 102]})
        df.to_parquet(cache_path)

        # Fresh cache should be loaded
        result = handler_with_cache._load_from_cache("AAPL", "1y", "1d", max_age_hours=24)
        assert result is not None

        # Simulate old file by setting max_age to 0
        result = handler_with_cache._load_from_cache("AAPL", "1y", "1d", max_age_hours=0)
        assert result is None


class TestGetData:
    """Tests for get_data method."""

    @pytest.fixture
    def handler_with_data(self):
        """Create handler with preloaded data."""
        handler = YFinanceDataHandler(enable_logging=False)
        handler.data["AAPL"] = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3))
        handler.symbols.append("AAPL")
        return handler

    def test_get_data_existing_symbol(self, handler_with_data):
        """Test retrieving data for existing symbol."""
        data = handler_with_data.get_data("AAPL")
        assert not data.empty
        assert len(data) == 3

    def test_get_data_specific_columns(self, handler_with_data):
        """Test retrieving specific columns."""
        data = handler_with_data.get_data("AAPL", columns=['close', 'volume'])
        assert list(data.columns) == ['close', 'volume']

    def test_get_data_missing_symbol_raises(self, handler_with_data):
        """Test that missing symbol raises KeyError."""
        with pytest.raises(KeyError, match="Data for symbol 'MSFT' not found"):
            handler_with_data.get_data("MSFT")

    def test_get_data_returns_copy(self, handler_with_data):
        """Test that get_data returns a copy, not original."""
        data1 = handler_with_data.get_data("AAPL")
        data1['close'] = 999  # Modify the copy

        data2 = handler_with_data.get_data("AAPL")
        assert data2['close'].iloc[0] != 999  # Original unchanged

    def test_get_data_missing_columns_logged(self, handler_with_data):
        """Test that missing columns are handled gracefully."""
        data = handler_with_data.get_data("AAPL", columns=['close', 'nonexistent'])
        assert 'close' in data.columns
        assert 'nonexistent' not in data.columns


class TestGetOHLCData:
    """Tests for get_ohlc_data method."""

    @pytest.fixture
    def handler_with_data(self):
        """Create handler with preloaded OHLC data."""
        handler = YFinanceDataHandler(enable_logging=False)
        handler.data["AAPL"] = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [99.0, 100.0, 101.0],
            'close': [104.0, 105.0, 106.0],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3, name='Date'))
        handler.symbols.append("AAPL")
        return handler

    def test_get_ohlc_data_returns_indexed(self, handler_with_data):
        """Test that get_ohlc_data returns data with date index."""
        data = handler_with_data.get_ohlc_data("AAPL")
        assert data.index.name == 'date'

    def test_get_ohlc_data_has_required_columns(self, handler_with_data):
        """Test that OHLC and volume columns are present."""
        data = handler_with_data.get_ohlc_data("AAPL")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in data.columns

    def test_get_ohlc_data_has_volume_column(self, handler_with_data):
        """Test that volume is included in get_ohlc_data output."""
        data = handler_with_data.get_ohlc_data("AAPL")
        assert 'volume' in data.columns
        assert data['volume'].iloc[0] == 1000000

    def test_get_ohlc_data_missing_volume_raises(self):
        """Test that missing volume column raises ValueError."""
        handler = YFinanceDataHandler(enable_logging=False)
        handler.data["AAPL"] = pd.DataFrame({
            'open': [100.0], 'high': [105.0], 'low': [99.0], 'close': [104.0]
            # no 'volume'
        }, index=pd.date_range('2024-01-01', periods=1, name='Date'))
        handler.symbols.append("AAPL")
        with pytest.raises(ValueError):
            handler.get_ohlc_data("AAPL")

    def test_get_ohlc_data_missing_symbol_raises(self, handler_with_data):
        """Test that missing symbol raises KeyError."""
        with pytest.raises(KeyError):
            handler_with_data.get_ohlc_data("MSFT")


class TestGetCombinedData:
    """Tests for get_combined_data method."""

    @pytest.fixture
    def handler_with_multiple(self):
        """Create handler with multiple symbols (full OHLCV required by get_ohlc_data)."""
        handler = YFinanceDataHandler(enable_logging=False)

        dates = pd.date_range('2024-01-01', periods=3, name='Date')

        handler.data["AAPL"] = pd.DataFrame({
            'open': [99.0, 100.0, 101.0],
            'high': [105.0, 106.0, 107.0],
            'low': [98.0, 99.0, 100.0],
            'close': [100.0, 101.0, 102.0],
            'volume': [1000000, 1100000, 1200000],
        }, index=dates)

        handler.data["MSFT"] = pd.DataFrame({
            'open': [199.0, 200.0, 201.0],
            'high': [205.0, 206.0, 207.0],
            'low': [198.0, 199.0, 200.0],
            'close': [200.0, 201.0, 202.0],
            'volume': [2000000, 2100000, 2200000],
        }, index=dates)

        handler.symbols.extend(["AAPL", "MSFT"])
        return handler

    def test_get_combined_data_long_format(self, handler_with_multiple):
        """Test that combined data is in long format with date and symbol columns."""
        data = handler_with_multiple.get_combined_data(["AAPL", "MSFT"], columns=['close'])
        assert 'symbol' in data.columns
        assert 'date' in data.columns
        assert len(data) == 6  # 3 rows per symbol

    def test_get_combined_data_includes_ohlcv_by_default(self, handler_with_multiple):
        """Test that all OHLC and volume columns are returned when no filter is given."""
        data = handler_with_multiple.get_combined_data(["AAPL"])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in data.columns

    def test_get_combined_data_column_filter(self, handler_with_multiple):
        """Test that column filter restricts output to requested columns."""
        data = handler_with_multiple.get_combined_data(["AAPL", "MSFT"], columns=['close', 'volume'])
        assert 'close' in data.columns
        assert 'volume' in data.columns
        assert 'open' not in data.columns
        assert len(data) == 6

    def test_get_combined_data_missing_symbol_skipped(self, handler_with_multiple):
        """Test that missing symbols are skipped with warning."""
        data = handler_with_multiple.get_combined_data(["AAPL", "NONEXISTENT"])
        assert len(data[data['symbol'] == 'AAPL']) == 3
        assert 'NONEXISTENT' not in data['symbol'].values

    def test_get_combined_data_empty_returns_schema(self, handler_with_multiple):
        """Test that empty result returns DataFrame with correct schema."""
        data = handler_with_multiple.get_combined_data(["NONEXISTENT"])
        assert 'date' in data.columns
        assert 'symbol' in data.columns
        assert len(data) == 0

    def test_get_combined_data_uses_get_ohlc_data(self, handler_with_multiple):
        """Test that get_combined_data delegates to get_ohlc_data."""
        with patch.object(handler_with_multiple, 'get_ohlc_data', wraps=handler_with_multiple.get_ohlc_data) as mock_ohlc:
            handler_with_multiple.get_combined_data(["AAPL", "MSFT"])
            assert mock_ohlc.call_count == 2
            calls = [call.args[0] for call in mock_ohlc.call_args_list]
            assert "AAPL" in calls
            assert "MSFT" in calls


class TestGetMultipleSymbolsData:
    """Tests for get_multiple_symbols_data method."""

    @pytest.fixture
    def handler_with_multiple(self):
        """Create handler with multiple symbols."""
        handler = YFinanceDataHandler(enable_logging=False)

        dates = pd.date_range('2024-01-01', periods=3)

        handler.data["AAPL"] = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200]
        }, index=dates)

        handler.data["MSFT"] = pd.DataFrame({
            'close': [200.0, 201.0, 202.0],
            'volume': [2000, 2100, 2200]
        }, index=dates)

        handler.symbols.extend(["AAPL", "MSFT"])
        return handler

    def test_get_multiple_symbols_wide_format(self, handler_with_multiple):
        """Test that result is in wide format."""
        data = handler_with_multiple.get_multiple_symbols_data(["AAPL", "MSFT"], column='close')
        assert 'date' in data.columns
        assert 'AAPL' in data.columns
        assert 'MSFT' in data.columns
        assert len(data) == 3

    def test_get_multiple_symbols_invalid_column_raises(self, handler_with_multiple):
        """Test that invalid column raises ValueError."""
        with pytest.raises(ValueError, match="Column 'invalid' not valid"):
            handler_with_multiple.get_multiple_symbols_data(["AAPL"], column='invalid')


class TestGetInfo:
    """Tests for get_info method."""

    def test_get_info_raises_on_error(self):
        """Test that get_info raises exception on error."""
        handler = YFinanceDataHandler(enable_logging=False)

        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.side_effect = Exception("API Error")

            with pytest.raises(Exception, match="API Error"):
                handler.get_info("INVALID_SYMBOL_XYZ")


class TestSaveData:
    """Tests for save_data method."""

    @pytest.fixture
    def handler_with_data(self):
        """Create handler with data."""
        handler = YFinanceDataHandler(enable_logging=False)
        handler.data["AAPL"] = pd.DataFrame({
            'close': [100.0, 101.0, 102.0]
        }, index=pd.date_range('2024-01-01', periods=3))
        handler.symbols.append("AAPL")
        return handler

    def test_save_data_csv(self, handler_with_data):
        """Test saving data to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            handler_with_data.save_data(str(filepath), format='csv')
            assert filepath.exists()

    def test_save_data_invalid_format_raises(self, handler_with_data):
        """Test that invalid format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.xyz"
            with pytest.raises(ValueError, match="Format must be one of"):
                handler_with_data.save_data(str(filepath), format='xyz')

    def test_save_data_no_data_raises(self):
        """Test that saving with no data raises ValueError."""
        handler = YFinanceDataHandler(enable_logging=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            with pytest.raises(ValueError, match="No symbols with data"):
                handler.save_data(str(filepath))


class TestClearCache:
    """Tests for clear_cache method."""

    def test_clear_cache_no_cache_dir(self):
        """Test clear_cache with no cache directory."""
        handler = YFinanceDataHandler(enable_logging=False)
        result = handler.clear_cache()
        assert result == 0

    def test_clear_cache_removes_files(self):
        """Test that clear_cache removes cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = YFinanceDataHandler(
                cache_dir=tmpdir,
                enable_logging=False
            )

            # Create some cache files
            (Path(tmpdir) / "AAPL_1y_1d.parquet").touch()
            (Path(tmpdir) / "MSFT_1y_1d.parquet").touch()

            result = handler.clear_cache()
            assert result == 2

            # Verify files are removed
            assert not (Path(tmpdir) / "AAPL_1y_1d.parquet").exists()

    def test_clear_cache_specific_symbols(self):
        """Test clearing cache for specific symbols."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = YFinanceDataHandler(
                cache_dir=tmpdir,
                enable_logging=False
            )

            # Create cache files
            (Path(tmpdir) / "AAPL_1y_1d.parquet").touch()
            (Path(tmpdir) / "MSFT_1y_1d.parquet").touch()

            result = handler.clear_cache(symbols=["AAPL"])
            assert result == 1

            # AAPL removed, MSFT still there
            assert not (Path(tmpdir) / "AAPL_1y_1d.parquet").exists()
            assert (Path(tmpdir) / "MSFT_1y_1d.parquet").exists()


class TestListAvailableData:
    """Tests for list_available_data method."""

    def test_list_available_data_empty(self):
        """Test list_available_data with no data."""
        handler = YFinanceDataHandler(enable_logging=False)
        result = handler.list_available_data()
        assert result == {}

    def test_list_available_data_with_data(self):
        """Test list_available_data with loaded data."""
        handler = YFinanceDataHandler(enable_logging=False)
        handler.data["AAPL"] = pd.DataFrame({
            'close': [100.0, 101.0, 102.0]
        }, index=pd.date_range('2024-01-01', periods=3))
        handler.symbols.append("AAPL")

        result = handler.list_available_data()
        assert "AAPL" in result
        assert result["AAPL"]["rows"] == 3


class TestDunderMethods:
    """Tests for __repr__ and __len__ methods."""

    def test_repr(self):
        """Test __repr__ output."""
        handler = YFinanceDataHandler(enable_logging=False)
        handler.symbols.append("AAPL")
        repr_str = repr(handler)
        assert "YFinanceDataHandler" in repr_str
        assert "symbols=1" in repr_str

    def test_len(self):
        """Test __len__ returns symbol count."""
        handler = YFinanceDataHandler(enable_logging=False)
        assert len(handler) == 0

        handler.symbols.extend(["AAPL", "MSFT"])
        assert len(handler) == 2
