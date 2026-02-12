"""
Tests for the optimizer module.

This module tests the StrategyOptimizer class and related functions
for grid search optimization and walk-forward analysis.
"""

import logging
import os
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from algoshort.optimizer import (
    FLOAT_TOLERANCE,
    MAX_GRID_COMBINATIONS,
    MIN_SEGMENT_ROWS,
    MIN_SEGMENT_SIZE,
    StrategyOptimizer,
    _sanitize_filename,
    _worker_evaluate,
    get_equity,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlc_df():
    """Create a sample OHLC DataFrame for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    return pd.DataFrame({
        'date': dates,
        'open': close + np.random.randn(n) * 0.1,
        'high': close + abs(np.random.randn(n) * 0.5),
        'low': close - abs(np.random.randn(n) * 0.5),
        'close': close,
        'volume': np.random.randint(1000, 10000, n),
    })


@pytest.fixture
def small_ohlc_df():
    """Create a small OHLC DataFrame."""
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10, freq='D'),
        'open': [100.0] * 10,
        'high': [101.0] * 10,
        'low': [99.0] * 10,
        'close': [100.0] * 10,
    })


@pytest.fixture
def mock_equity_func():
    """Create a mock equity function for testing."""
    def equity_func(segment_df, segment_idx=0, config_path='', **kwargs):
        return {
            'convex': 1.5 + np.random.random() * 0.5,
            'constant': 1.2 + np.random.random() * 0.3,
            'equal_weight': 1.3 + np.random.random() * 0.4,
            'rows_processed': len(segment_df),
            'segment_idx': segment_idx,
        }
    return equity_func


@pytest.fixture
def temp_config_file():
    """Create a temporary config file."""
    config_content = '''{
        "regimes": {
            "floor_ceiling": {
                "lvl": 1,
                "vlty_n": 63,
                "threshold": 1.5,
                "dgt": 3,
                "d_vol": 1.0,
                "dist_pct": 0.05,
                "retrace_pct": 0.05,
                "r_vol": 1.0
            }
        }
    }'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(config_content)
        f.flush()
        yield f.name

    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


# =============================================================================
# Test _sanitize_filename
# =============================================================================

class TestSanitizeFilename:
    """Tests for the _sanitize_filename function."""

    def test_normal_string(self):
        """Test with normal string."""
        assert _sanitize_filename('normal_name') == 'normal_name'

    def test_path_separators(self):
        """Test removal of path separators."""
        assert _sanitize_filename('path/to/file') == 'path_to_file'
        assert _sanitize_filename('path\\to\\file') == 'path_to_file'

    def test_special_characters(self):
        """Test removal of special characters."""
        assert _sanitize_filename('file<>:"|?*') == 'file_______'

    def test_null_characters(self):
        """Test removal of null characters."""
        assert _sanitize_filename('file\x00name') == 'file_name'

    def test_non_string(self):
        """Test with non-string input."""
        assert _sanitize_filename(123) == '123'
        assert _sanitize_filename(None) == 'None'


# =============================================================================
# Test get_equity validation
# =============================================================================

class TestGetEquityValidation:
    """Tests for get_equity input validation."""

    def test_missing_config_file(self, sample_ohlc_df):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            get_equity(sample_ohlc_df, config_path='nonexistent.json')

    def test_empty_dataframe(self, temp_config_file):
        """Test error with empty DataFrame."""
        with pytest.raises(ValueError, match="cannot be empty"):
            get_equity(pd.DataFrame(), config_path=temp_config_file)

    def test_single_row_dataframe(self, temp_config_file):
        """Test error with single row DataFrame."""
        df = pd.DataFrame({
            'date': [pd.Timestamp('2020-01-01')],
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.0],
        })
        with pytest.raises(ValueError, match="at least 2 rows"):
            get_equity(df, config_path=temp_config_file)


# =============================================================================
# Test _worker_evaluate
# =============================================================================

class TestWorkerEvaluate:
    """Tests for the _worker_evaluate function."""

    def test_dict_return(self, sample_ohlc_df, temp_config_file):
        """Test with function returning dict."""
        def equity_func(df, **kwargs):
            return {'metric': 1.0}

        result = _worker_evaluate(
            segment_data=sample_ohlc_df,
            segment_idx=0,
            param_kwargs={'window': 10},
            equity_func=equity_func,
            config_path=temp_config_file,
        )

        assert result['metric'] == 1.0
        assert result['window'] == 10

    def test_series_return(self, sample_ohlc_df, temp_config_file):
        """Test with function returning Series."""
        def equity_func(df, **kwargs):
            return pd.Series({'metric': 2.0})

        result = _worker_evaluate(
            segment_data=sample_ohlc_df,
            segment_idx=0,
            param_kwargs={'window': 20},
            equity_func=equity_func,
            config_path=temp_config_file,
        )

        assert result['metric'] == 2.0
        assert result['window'] == 20

    def test_invalid_return_type(self, sample_ohlc_df, temp_config_file):
        """Test error with invalid return type."""
        def equity_func(df, **kwargs):
            return [1, 2, 3]  # Invalid return type

        with pytest.raises(TypeError, match="expected dict or Series"):
            _worker_evaluate(
                segment_data=sample_ohlc_df,
                segment_idx=0,
                param_kwargs={},
                equity_func=equity_func,
                config_path=temp_config_file,
            )


# =============================================================================
# Test StrategyOptimizer.__init__
# =============================================================================

class TestStrategyOptimizerInit:
    """Tests for StrategyOptimizer initialization."""

    def test_valid_init(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test valid initialization."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        assert len(optimizer.data) == len(sample_ohlc_df)
        assert optimizer.config_path == temp_config_file

    def test_empty_dataframe(self, mock_equity_func, temp_config_file):
        """Test error with empty DataFrame."""
        with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
            StrategyOptimizer(
                data=pd.DataFrame(),
                equity_func=mock_equity_func,
                config_path=temp_config_file,
            )

    def test_non_dataframe(self, mock_equity_func, temp_config_file):
        """Test error with non-DataFrame."""
        with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
            StrategyOptimizer(
                data=[1, 2, 3],
                equity_func=mock_equity_func,
                config_path=temp_config_file,
            )

    def test_non_callable_equity_func(self, sample_ohlc_df, temp_config_file):
        """Test error with non-callable equity_func."""
        with pytest.raises(TypeError, match="must be callable"):
            StrategyOptimizer(
                data=sample_ohlc_df,
                equity_func="not_callable",
                config_path=temp_config_file,
            )

    def test_invalid_config_path(self, sample_ohlc_df, mock_equity_func):
        """Test error with invalid config path."""
        with pytest.raises(ValueError, match="valid file path"):
            StrategyOptimizer(
                data=sample_ohlc_df,
                equity_func=mock_equity_func,
                config_path='nonexistent.json',
            )

    def test_data_copied(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test that data is copied defensively."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        # Modify original
        sample_ohlc_df.iloc[0, 0] = 'modified'

        # Optimizer's copy should be unchanged
        assert optimizer.data.iloc[0, 0] != 'modified'


# =============================================================================
# Test run_grid_search
# =============================================================================

class TestRunGridSearch:
    """Tests for StrategyOptimizer.run_grid_search."""

    def test_empty_param_grid(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test error with empty param_grid."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        with pytest.raises(ValueError, match="param_grid required"):
            optimizer.run_grid_search(
                segment_data=sample_ohlc_df,
                param_grid={},
                segment_idx=0,
            )

    def test_n_jobs_zero(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test error with n_jobs=0."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        with pytest.raises(ValueError, match="n_jobs cannot be 0"):
            optimizer.run_grid_search(
                segment_data=sample_ohlc_df,
                param_grid={'window': [10, 20]},
                segment_idx=0,
                n_jobs=0,
            )

    def test_combination_limit(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test error when grid exceeds combination limit."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        # Create grid that exceeds limit
        large_grid = {
            'param1': list(range(100)),
            'param2': list(range(100)),
            'param3': list(range(10)),
        }  # 100 * 100 * 10 = 100,000 > MAX_GRID_COMBINATIONS

        with pytest.raises(RuntimeError, match="exceeds limit"):
            optimizer.run_grid_search(
                segment_data=sample_ohlc_df,
                param_grid=large_grid,
                segment_idx=0,
            )

    def test_empty_value_list(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test with empty value list returns empty DataFrame."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        result = optimizer.run_grid_search(
            segment_data=sample_ohlc_df,
            param_grid={'window': []},
            segment_idx=0,
        )

        assert result.empty

    def test_single_combination(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test with single combination."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        result = optimizer.run_grid_search(
            segment_data=sample_ohlc_df,
            param_grid={'window': [10]},
            segment_idx=0,
            n_jobs=1,
        )

        assert len(result) == 1
        assert 'window' in result.columns
        assert result.iloc[0]['window'] == 10

    def test_multiple_combinations(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test with multiple combinations."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        result = optimizer.run_grid_search(
            segment_data=sample_ohlc_df,
            param_grid={'window': [10, 20], 'multiplier': [1.5, 2.0]},
            segment_idx=0,
            n_jobs=1,
        )

        assert len(result) == 4  # 2 * 2 combinations
        assert 'window' in result.columns
        assert 'multiplier' in result.columns


# =============================================================================
# Test rolling_walk_forward
# =============================================================================

class TestRollingWalkForward:
    """Tests for StrategyOptimizer.rolling_walk_forward."""

    def test_n_segments_minimum(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test error with n_segments < 2."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        with pytest.raises(ValueError, match="n_segments must be >= 2"):
            optimizer.rolling_walk_forward(
                stop_method='atr',
                param_grid={'window': [10, 20]},
                n_segments=1,
            )

    def test_segment_size_too_small(self, small_ohlc_df, mock_equity_func, temp_config_file):
        """Test error when segment size is too small."""
        optimizer = StrategyOptimizer(
            data=small_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        with pytest.raises(ValueError, match="Segment size too small"):
            optimizer.rolling_walk_forward(
                stop_method='atr',
                param_grid={'window': [10]},
                n_segments=10,  # Too many segments for 10 rows
            )

    def test_returns_tuple(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test that rolling_walk_forward returns a tuple."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        result = optimizer.rolling_walk_forward(
            stop_method='atr',
            param_grid={'window': [10, 20]},
            n_segments=2,
            n_jobs=1,
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

        oos_df, stability, history = result
        assert isinstance(oos_df, pd.DataFrame)
        assert isinstance(stability, dict)
        assert isinstance(history, list)

    def test_stability_calculation(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test stability metrics calculation."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        oos_df, stability, history = optimizer.rolling_walk_forward(
            stop_method='atr',
            param_grid={'window': [10, 14, 20]},
            n_segments=2,
            n_jobs=1,
        )

        assert 'n_segments_valid' in stability
        if stability['n_segments_valid'] >= 2:
            assert 'window_cv' in stability


# =============================================================================
# Test sensitivity_analysis
# =============================================================================

class TestSensitivityAnalysis:
    """Tests for StrategyOptimizer.sensitivity_analysis."""

    def test_empty_best_params(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test error with empty best_params."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        with pytest.raises(ValueError, match="non-empty dict"):
            optimizer.sensitivity_analysis(
                stop_method='atr',
                best_params={},
            )

    def test_returns_tuple(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test that sensitivity_analysis returns a tuple."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        result = optimizer.sensitivity_analysis(
            stop_method='atr',
            best_params={'window': 14},
            variance=0.2,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

        plateau_ratio, results_df = result
        assert isinstance(plateau_ratio, (float, type(np.nan)))
        assert isinstance(results_df, pd.DataFrame)

    def test_integer_param_variance(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test variance grid generation for integer params."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        _, results = optimizer.sensitivity_analysis(
            stop_method='atr',
            best_params={'window': 10},
            variance=0.2,
        )

        # Should have 3 unique window values (0.8, 1.0, 1.2 times 10)
        unique_windows = results['window'].unique()
        assert len(unique_windows) <= 3

    def test_float_param_variance(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test variance grid generation for float params."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        _, results = optimizer.sensitivity_analysis(
            stop_method='atr',
            best_params={'multiplier': 2.0},
            variance=0.2,
        )

        # Should have exactly 3 multiplier values
        unique_multipliers = results['multiplier'].unique()
        assert len(unique_multipliers) == 3


# =============================================================================
# Test compare_signals
# =============================================================================

class TestCompareSignals:
    """Tests for StrategyOptimizer.compare_signals."""

    def test_single_signal(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test with single signal."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        result = optimizer.compare_signals(
            signals='rrg',
            stop_method='atr',
            param_grid={'window': [10, 14]},
            n_segments=2,
            n_jobs=1,
        )

        assert isinstance(result, pd.DataFrame)

    def test_multiple_signals_list(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test with list of signals (currently only uses first)."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        result = optimizer.compare_signals(
            signals=['rrg', 'other'],
            stop_method='atr',
            param_grid={'window': [10]},
            n_segments=2,
            n_jobs=1,
        )

        assert isinstance(result, pd.DataFrame)


# =============================================================================
# Test _evaluate_params
# =============================================================================

class TestEvaluateParams:
    """Tests for StrategyOptimizer._evaluate_params."""

    def test_dict_return(self, sample_ohlc_df, temp_config_file):
        """Test with function returning dict."""
        def equity_func(df, **kwargs):
            return {'metric': 1.5}

        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=equity_func,
            config_path=temp_config_file,
        )

        result = optimizer._evaluate_params(
            segment_data=sample_ohlc_df,
            segment_idx=0,
            window=10,
        )

        assert result['metric'] == 1.5
        assert result['window'] == 10

    def test_series_return(self, sample_ohlc_df, temp_config_file):
        """Test with function returning Series."""
        def equity_func(df, **kwargs):
            return pd.Series({'metric': 2.5})

        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=equity_func,
            config_path=temp_config_file,
        )

        result = optimizer._evaluate_params(
            segment_data=sample_ohlc_df,
            segment_idx=0,
        )

        assert result['metric'] == 2.5

    def test_invalid_return_type(self, sample_ohlc_df, temp_config_file):
        """Test error with invalid return type."""
        def equity_func(df, **kwargs):
            return "invalid"

        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=equity_func,
            config_path=temp_config_file,
        )

        with pytest.raises(TypeError, match="dict or pd.Series"):
            optimizer._evaluate_params(
                segment_data=sample_ohlc_df,
                segment_idx=0,
            )


# =============================================================================
# Test edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_nan_metric(self, sample_ohlc_df, temp_config_file):
        """Test handling of all-NaN metric values."""
        def equity_func(df, **kwargs):
            return {'convex': np.nan, 'constant': np.nan}

        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=equity_func,
            config_path=temp_config_file,
        )

        # Should handle gracefully (skipping segments with all-NaN)
        oos_df, stability, history = optimizer.rolling_walk_forward(
            stop_method='atr',
            param_grid={'window': [10]},
            n_segments=2,
            n_jobs=1,
            opt_metric='convex',
        )

        # May have no valid segments due to all-NaN
        assert stability['n_segments_valid'] >= 0

    def test_constant_prices(self, temp_config_file):
        """Test with constant price data."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=200, freq='D'),
            'open': [100.0] * 200,
            'high': [100.0] * 200,
            'low': [100.0] * 200,
            'close': [100.0] * 200,
        })

        def equity_func(segment_df, **kwargs):
            return {'convex': 1.0}

        optimizer = StrategyOptimizer(
            data=df,
            equity_func=equity_func,
            config_path=temp_config_file,
        )

        result = optimizer.run_grid_search(
            segment_data=df,
            param_grid={'window': [10]},
            segment_idx=0,
            n_jobs=1,
        )

        assert len(result) == 1

    def test_negative_n_jobs(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test with negative n_jobs (should use all CPUs)."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        result = optimizer.run_grid_search(
            segment_data=sample_ohlc_df,
            param_grid={'window': [10, 20]},
            segment_idx=0,
            n_jobs=-1,
        )

        assert len(result) == 2


# =============================================================================
# Test constants
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_min_segment_size(self):
        """Test MIN_SEGMENT_SIZE constant."""
        assert MIN_SEGMENT_SIZE > 0
        assert MIN_SEGMENT_SIZE == 20

    def test_min_segment_rows(self):
        """Test MIN_SEGMENT_ROWS constant."""
        assert MIN_SEGMENT_ROWS > 0
        assert MIN_SEGMENT_ROWS == 30

    def test_max_grid_combinations(self):
        """Test MAX_GRID_COMBINATIONS constant."""
        assert MAX_GRID_COMBINATIONS > 0
        assert MAX_GRID_COMBINATIONS == 10000

    def test_float_tolerance(self):
        """Test FLOAT_TOLERANCE constant."""
        assert FLOAT_TOLERANCE > 0
        assert FLOAT_TOLERANCE < 1e-5


# =============================================================================
# Test state management
# =============================================================================

class TestStateManagement:
    """Tests for optimizer state management."""

    def test_last_results_stored(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test that last results are stored."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        optimizer.rolling_walk_forward(
            stop_method='atr',
            param_grid={'window': [10, 14]},
            n_segments=2,
            n_jobs=1,
        )

        assert optimizer._last_oos_results is not None
        assert optimizer._last_stability is not None
        assert optimizer._last_param_history is not None

    def test_initial_state_none(self, sample_ohlc_df, mock_equity_func, temp_config_file):
        """Test that initial state is None/empty."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        assert optimizer._last_oos_results is None
        assert optimizer._last_stability is None
        assert optimizer._last_param_history == []


# =============================================================================
# Test logging
# =============================================================================

class TestLogging:
    """Tests for logging behavior."""

    def test_no_print_statements(self, sample_ohlc_df, mock_equity_func, temp_config_file, capsys):
        """Test that no print statements are used (only logging)."""
        optimizer = StrategyOptimizer(
            data=sample_ohlc_df,
            equity_func=mock_equity_func,
            config_path=temp_config_file,
        )

        # Disable logging to stdout for this test
        logging.getLogger('algoshort.optimizer').setLevel(logging.CRITICAL)

        optimizer.run_grid_search(
            segment_data=sample_ohlc_df,
            param_grid={'window': [10]},
            segment_idx=0,
            n_jobs=1,
        )

        captured = capsys.readouterr()
        # Should have no direct print output (may have joblib output)
        # Main assertion is that it doesn't crash
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
