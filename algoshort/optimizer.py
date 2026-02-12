"""
Strategy optimization module for parameter tuning and walk-forward analysis.

This module provides tools for grid search optimization, rolling walk-forward
validation, and sensitivity analysis of trading strategies.

Example:
    >>> from algoshort.optimizer import StrategyOptimizer, get_equity
    >>> optimizer = StrategyOptimizer(data, get_equity, 'config_regime.json')
    >>> results = optimizer.rolling_walk_forward(
    ...     stop_method='atr',
    ...     param_grid={'window': [10, 14, 20], 'multiplier': [1.5, 2.0, 2.5]},
    ...     n_segments=4
    ... )
"""

# Standard library
import itertools
import logging
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Third-party
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Local imports
from algoshort.position_sizing import PositionSizing
from algoshort.regime_fc import RegimeFC
from algoshort.returns import ReturnsCalculator
from algoshort.stop_loss import StopLossCalculator
from algoshort.utils import load_config


# Module logger
logger = logging.getLogger(__name__)

# Constants
MIN_SEGMENT_SIZE = 20
MIN_SEGMENT_ROWS = 30
MAX_GRID_COMBINATIONS = 10000
FLOAT_TOLERANCE = 1e-10

__all__ = [
    'get_equity',
    'StrategyOptimizer',
]


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use in filenames."""
    # Remove path separators and invalid characters
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(name))


def get_equity(
    segment_df: pd.DataFrame,
    segment_idx: int = 0,
    config_path: str = 'config_regime.json',
    price_col: str = 'close',
    stop_method: str = 'atr',
    inplace: bool = False,
    save_output: bool = False,
    **stop_kwargs: Any
) -> Dict[str, Any]:
    """
    Process one segment independently with flexible stop-loss method.

    Args:
        segment_df: Raw price data for this segment only (must have OHLC columns)
        segment_idx: Segment index for logging/filename
        config_path: Path to regime config JSON file
        price_col: Main price column ('close', 'rclose', etc.)
        stop_method: Stop-loss method to use. Options: 'atr', 'fixed_percentage',
                     'breakout_channel', 'moving_average', 'volatility_std',
                     'support_resistance', 'classified_pivot'
        inplace: If True, modify segment_df; else work on copy (recommended False)
        save_output: If True, save output to Excel file (default False)
        **stop_kwargs: Passed directly to the chosen stop-loss method
                       e.g. window=14, multiplier=2.0, percentage=0.05

    Returns:
        Dict containing final equity metrics and metadata:
            - Equity curve values (constant, concave, convex, equal_weight)
            - segment_idx, start_date, end_date
            - signal, stop_method, stop_kwargs, rows_processed

    Raises:
        FileNotFoundError: If config_path doesn't exist
        ValueError: If segment_df is empty or has insufficient rows
        KeyError: If required columns are missing
    """
    # Validate inputs
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if segment_df.empty:
        raise ValueError("segment_df cannot be empty")

    if len(segment_df) < 2:
        raise ValueError(f"segment_df must have at least 2 rows, got {len(segment_df)}")

    logger.info(
        "Processing segment %d | stop_method='%s' | rows=%d",
        segment_idx, stop_method, len(segment_df)
    )
    logger.debug("Input columns: %s", segment_df.columns.tolist())
    if stop_kwargs:
        logger.debug("Stop-loss kwargs: %s", stop_kwargs)

    df = segment_df if inplace else segment_df.copy()

    # 1. Compute signal column 'rrg'
    config = load_config(config_path)
    logger.info("Computing signal 'rrg'...")
    regime_fc = RegimeFC(df, logging.WARNING)
    df = regime_fc.compute_regime(
        relative=True,
        lvl=config['regimes']['floor_ceiling']['lvl'],
        vlty_n=config['regimes']['floor_ceiling']['vlty_n'],
        threshold=config['regimes']['floor_ceiling']['threshold'],
        dgt=config['regimes']['floor_ceiling']['dgt'],
        d_vol=config['regimes']['floor_ceiling']['d_vol'],
        dist_pct=config['regimes']['floor_ceiling']['dist_pct'],
        retrace_pct=config['regimes']['floor_ceiling']['retrace_pct'],
        r_vol=config['regimes']['floor_ceiling']['r_vol']
    )

    signal = 'rrg'

    # 2. Compute returns
    logger.info("Computing returns...")
    returns_calc = ReturnsCalculator(ohlc_stock=df)
    df = returns_calc.get_returns(
        df=df,
        signal=signal,
        relative=True,
        inplace=False
    )

    # 3. Compute stop-loss using selected method + kwargs
    logger.info("Computing stop-loss using method '%s'...", stop_method)
    sl_calc = StopLossCalculator(df)

    df = sl_calc.get_stop_loss(
        signal=signal,
        method=stop_method,
        **stop_kwargs
    )

    stop_loss_col = f"{signal}_stop_loss"
    if stop_loss_col not in df.columns:
        raise KeyError(
            f"Stop-loss column '{stop_loss_col}' not created by method '{stop_method}'"
        )

    # Reset index to ensure sequential 0-based indexing
    df = df.reset_index(drop=True)

    # 4. Compute equity curves
    logger.info("Computing equity curves...")
    pos = PositionSizing(
        tolerance=-0.2,
        mn=0.01,
        mx=0.10,
        equal_weight=0.25,
        avg=0.05,
        lot=100,
        initial_capital=100000
    )

    # Calculate shares with custom column names
    df = pos.calculate_shares(
        df=df,
        daily_chg=signal + '_chg1D_fx',
        sl=stop_loss_col,
        signal=signal,
        close=price_col
    )

    # Save full output for inspection (optional)
    if save_output:
        safe_method = _sanitize_filename(stop_method)
        output_file = f"segment_{segment_idx}_{signal}_{safe_method}_output.xlsx"
        try:
            df.to_excel(output_file)
            logger.info("Saved full segment output: %s", output_file)
        except (OSError, PermissionError) as e:
            logger.warning("Failed to save output file: %s", e)

    # Extract final equity metrics (last row)
    # Try both naming conventions for compatibility
    metrics_cols_v1 = [
        'date',
        signal + '_constant',
        signal + '_concave',
        signal + '_convex',
        signal + '_equal_weight'
    ]
    metrics_cols_v2 = [
        'date',
        signal + '_equity_constant',
        signal + '_equity_concave',
        signal + '_equity_convex',
        signal + '_equity_equal'
    ]

    available_cols = [col for col in metrics_cols_v1 if col in df.columns]
    if not available_cols:
        available_cols = [col for col in metrics_cols_v2 if col in df.columns]

    if not available_cols:
        logger.warning(
            "No equity curve columns found. Available columns: %s",
            df.columns.tolist()
        )
        raise ValueError("No equity curve columns found after position sizing")

    last_row = df[available_cols].iloc[-1].to_dict()

    # Normalize column names for consistent output
    normalized_row = {}
    for key, value in last_row.items():
        # Remove signal prefix and _equity_ infix for cleaner keys
        clean_key = key.replace(f'{signal}_equity_', '').replace(f'{signal}_', '')
        normalized_row[clean_key] = value

    # Add useful metadata
    result = normalized_row.copy()
    result['segment_idx'] = segment_idx
    result['rows_processed'] = len(df)
    result['signal'] = signal
    result['stop_method'] = stop_method
    result['stop_kwargs'] = stop_kwargs.copy() if stop_kwargs else {}

    # Add dates if available
    if 'date' in df.columns:
        result['start_date'] = df['date'].iloc[0]
        result['end_date'] = df['date'].iloc[-1]

    logger.info(
        "Segment %d completed | rows=%d | stop_method=%s",
        segment_idx, len(df), stop_method
    )

    return result


def _worker_evaluate(
    segment_data: pd.DataFrame,
    segment_idx: int,
    param_kwargs: Dict[str, Any],
    equity_func: Callable[..., Dict[str, Any]],
    config_path: str,
) -> Dict[str, Any]:
    """
    Standalone worker for parallel grid search evaluation.

    Args:
        segment_data: DataFrame segment to evaluate
        segment_idx: Index of the segment
        param_kwargs: Parameters to pass to equity_func
        equity_func: Callable that returns performance metrics
        config_path: Path to configuration file

    Returns:
        Dict containing metrics merged with param_kwargs

    Raises:
        TypeError: If equity_func doesn't return dict or Series
    """
    metrics = equity_func(
        segment_data,
        segment_idx=segment_idx,
        config_path=config_path,
        **param_kwargs
    )

    if isinstance(metrics, pd.Series):
        metrics = metrics.to_dict()
    elif not isinstance(metrics, dict):
        raise TypeError(
            f"equity_func returned {type(metrics).__name__}, expected dict or Series. "
            f"Got: {metrics}"
        )

    metrics.update(param_kwargs)
    return metrics


class StrategyOptimizer:
    """
    Optimizer for parameter tuning using grid search and walk-forward analysis.

    This class provides methods for:
    - Grid search optimization across parameter combinations
    - Rolling walk-forward validation with in-sample/out-of-sample splits
    - Sensitivity analysis around optimal parameters

    Attributes:
        data (pd.DataFrame): Historical OHLC DataFrame
        equity_func (Callable): Function to compute equity metrics
        config_path (str): Path to regime configuration JSON

    Example:
        >>> optimizer = StrategyOptimizer(data, get_equity, 'config.json')
        >>> oos_df, stability, history = optimizer.rolling_walk_forward(
        ...     stop_method='atr',
        ...     param_grid={'window': [10, 14, 20], 'multiplier': [1.5, 2.0]},
        ...     n_segments=4,
        ...     opt_metric='convex'
        ... )
    """

    def __init__(
        self,
        data: pd.DataFrame,
        equity_func: Callable[..., Dict[str, Any]],
        config_path: str,
    ) -> None:
        """
        Initialize the StrategyOptimizer.

        Args:
            data: Full historical OHLC DataFrame
            equity_func: Callable that takes (segment_df, segment_idx, config_path, **params)
                         and returns dict with performance metrics
            config_path: Path to the regime configuration JSON file

        Raises:
            ValueError: If data is empty or config_path is invalid
            TypeError: If equity_func is not callable
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("data must be a non-empty pandas DataFrame")
        if not callable(equity_func):
            raise TypeError("equity_func must be callable")
        if not isinstance(config_path, str) or not os.path.isfile(config_path):
            raise ValueError(f"config_path must be a valid file path: {config_path}")

        self.data = data.copy()
        self.equity_func = equity_func
        self.config_path = config_path

        self._last_oos_results: Optional[pd.DataFrame] = None
        self._last_stability: Optional[Dict[str, float]] = None
        self._last_param_history: List[Dict[str, Any]] = []

        logger.info(
            "StrategyOptimizer initialized with %d rows",
            len(self.data)
        )

    def _evaluate_params(
        self,
        segment_data: pd.DataFrame,
        segment_idx: int,
        **param_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate parameters on a data segment.

        Args:
            segment_data: DataFrame segment to evaluate
            segment_idx: Index of the segment
            **param_kwargs: Parameters to pass to equity_func

        Returns:
            Dict containing metrics merged with parameters

        Raises:
            TypeError: If equity_func doesn't return dict or Series
        """
        metrics = self.equity_func(
            segment_data,
            segment_idx=segment_idx,
            config_path=self.config_path,
            **param_kwargs
        )
        logger.debug("Segment %d | Raw metrics: %s", segment_idx, metrics)

        if isinstance(metrics, pd.Series):
            metrics = metrics.to_dict()
        elif not isinstance(metrics, dict):
            raise TypeError(
                f"equity_func must return dict or pd.Series, got {type(metrics).__name__}"
            )

        metrics.update(param_kwargs)
        return metrics

    def run_grid_search(
        self,
        segment_data: pd.DataFrame,
        param_grid: Dict[str, Iterable[Any]],
        segment_idx: int,
        n_jobs: int = 1,
        progress: bool = False,
        backend: str = "loky",
        prefer: str = "processes",
    ) -> pd.DataFrame:
        """
        Run grid search over parameter combinations.

        Args:
            segment_data: DataFrame segment to evaluate
            param_grid: Dict mapping parameter names to lists of values
            segment_idx: Index of the segment
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            progress: Whether to show progress (not implemented)
            backend: Joblib backend ('loky', 'threading', 'multiprocessing')
            prefer: Joblib prefer setting ('processes', 'threads')

        Returns:
            DataFrame with one row per parameter combination, containing
            metrics and parameter values

        Raises:
            ValueError: If param_grid is empty or n_jobs is 0
            RuntimeError: If grid has too many combinations
        """
        if not param_grid:
            raise ValueError("param_grid required")
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        if n_jobs < 0:
            n_jobs = -1

        param_names = list(param_grid.keys())
        param_values_lists = [list(v) for v in param_grid.values()]

        # Calculate total combinations before creating
        n_combinations = 1
        for values in param_values_lists:
            n_combinations *= len(values)

        if n_combinations > MAX_GRID_COMBINATIONS:
            raise RuntimeError(
                f"Grid search has {n_combinations} combinations, "
                f"exceeds limit of {MAX_GRID_COMBINATIONS}. "
                "Reduce parameter ranges or use random search."
            )

        all_combos = list(itertools.product(*param_values_lists))

        if not all_combos:
            logger.warning("Grid search produced no combinations")
            return pd.DataFrame()

        logger.info(
            "Running grid search with %d combinations on segment %d",
            len(all_combos), segment_idx
        )

        tasks = [
            delayed(_worker_evaluate)(
                segment_data=segment_data,
                segment_idx=segment_idx,
                param_kwargs=dict(zip(param_names, combo)),
                equity_func=self.equity_func,
                config_path=self.config_path,
            )
            for combo in all_combos
        ]

        effective_backend = backend

        try:
            results = Parallel(
                n_jobs=n_jobs,
                backend=effective_backend,
                prefer=prefer,
                verbose=10 if n_jobs > 1 else 0,
            )(tasks)
        except Exception as e:
            if "pickle" in str(e).lower() and effective_backend == "loky":
                logger.error(
                    "Pickling error with loky backend. "
                    "Try backend='threading' or ensure equity_func is picklable."
                )
            raise

        if not results:
            logger.warning("Grid search returned no results")
            return pd.DataFrame()

        return pd.DataFrame(results)

    def rolling_walk_forward(
        self,
        stop_method: str,
        param_grid: Dict[str, Iterable[Any]],
        close_col: str = "close",
        n_segments: int = 4,
        n_jobs: int = 1,
        progress: bool = False,
        verbose: bool = False,
        opt_metric: str = "convex",
    ) -> Tuple[pd.DataFrame, Dict[str, float], List[Dict[str, Any]]]:
        """
        Perform rolling walk-forward optimization.

        This method splits data into segments, optimizes on in-sample data,
        and validates on out-of-sample data.

        Args:
            stop_method: Stop-loss method to use
            param_grid: Dict mapping parameter names to lists of values
            close_col: Name of the close price column
            n_segments: Number of walk-forward segments (minimum 2)
            n_jobs: Number of parallel jobs for grid search
            progress: Whether to show progress
            verbose: Whether to print detailed output
            opt_metric: Metric to optimize (column name in results)

        Returns:
            Tuple of:
                - oos_df: DataFrame with out-of-sample results
                - stability: Dict with parameter stability metrics (CV values)
                - history: List of dicts with per-segment best parameters

        Raises:
            ValueError: If n_segments < 2 or segment size too small
        """
        oos_df, stability, history = self._single_rolling_walk_forward(
            stop_method=stop_method,
            param_grid=param_grid,
            close_col=close_col,
            n_segments=n_segments,
            n_jobs=n_jobs,
            progress=progress,
            verbose=verbose,
            opt_metric=opt_metric,
        )

        if verbose:
            logger.info("Stability: %s", stability)
            if oos_df is not None and not oos_df.empty and opt_metric in oos_df.columns:
                logger.info(
                    "OOS rows: %d, mean %s: %.4f",
                    len(oos_df), opt_metric, oos_df[opt_metric].mean()
                )

        return oos_df, stability, history

    def _single_rolling_walk_forward(
        self,
        stop_method: str,
        param_grid: Dict[str, Iterable[Any]],
        close_col: str = "close",
        n_segments: int = 4,
        n_jobs: int = 1,
        progress: bool = False,
        verbose: bool = False,
        opt_metric: str = "convex",
    ) -> Tuple[pd.DataFrame, Dict[str, float], List[Dict[str, Any]]]:
        """
        Internal walk-forward logic for a single run.

        Args:
            stop_method: Stop-loss method to use
            param_grid: Parameter grid for optimization
            close_col: Close price column name
            n_segments: Number of segments
            n_jobs: Parallel jobs
            progress: Show progress
            verbose: Verbose output
            opt_metric: Optimization metric

        Returns:
            Tuple of (oos_df, stability_dict, param_history)
        """
        if n_segments < 2:
            raise ValueError("n_segments must be >= 2")

        n = len(self.data)
        segment_size = n // (n_segments + 1)

        if segment_size < MIN_SEGMENT_SIZE:
            raise ValueError(
                f"Segment size too small ({segment_size} bars) — "
                f"increase data or reduce n_segments. Minimum is {MIN_SEGMENT_SIZE}."
            )

        if verbose:
            logger.info(
                "Total bars: %d | Segments: %d | Approx bars/segment: %d",
                n, n_segments, segment_size
            )

        oos_rows = []
        param_history = []

        for i in range(n_segments):
            train_start = i * segment_size
            train_end = (i + 1) * segment_size
            test_start = train_end
            test_end = (i + 2) * segment_size

            is_data = self.data.iloc[train_start:train_end]
            oos_data = self.data.iloc[test_start:test_end]

            if len(is_data) < MIN_SEGMENT_ROWS or len(oos_data) < MIN_SEGMENT_ROWS:
                logger.warning(
                    "Skipping segment %d: IS=%d rows, OOS=%d rows (minimum %d)",
                    i, len(is_data), len(oos_data), MIN_SEGMENT_ROWS
                )
                continue

            # In-sample optimization
            is_results = self.run_grid_search(
                segment_data=is_data,
                param_grid=param_grid,
                segment_idx=i,
                n_jobs=n_jobs,
                progress=progress,
                backend="loky",
                prefer="processes",
            )

            logger.debug("IS results for segment %d: %d rows", i, len(is_results))

            if is_results.empty:
                logger.warning("Segment %d: Grid search returned empty results", i)
                continue

            # Validate opt_metric column exists
            if opt_metric not in is_results.columns:
                available = is_results.columns.tolist()
                raise KeyError(
                    f"Optimization metric '{opt_metric}' not found in results. "
                    f"Available columns: {available}"
                )

            # Check for all-NaN metric
            if is_results[opt_metric].isna().all():
                logger.warning(
                    "Segment %d: All values for '%s' are NaN, skipping",
                    i, opt_metric
                )
                continue

            # Pick best by opt_metric
            best = is_results.loc[is_results[opt_metric].idxmax()]
            best_params = {p: best[p] for p in param_grid}

            param_history.append({
                "segment": i + 1,
                "params": best_params,
                "is_metric": best[opt_metric],
            })

            # Out-of-sample evaluation with best params
            oos_row = self._evaluate_params(
                segment_data=oos_data,
                stop_method=stop_method,
                close_col=close_col,
                segment_idx=i,
                **best_params
            )
            oos_row["segment"] = i + 1
            oos_rows.append(oos_row)

        oos_df = pd.DataFrame(oos_rows)

        # Calculate stability metrics
        if len(param_history) >= 2:
            hist_df = pd.json_normalize(param_history)

            stability = {}
            for param in param_grid:
                col = f"params.{param}"
                if col in hist_df.columns:
                    mean_val = hist_df[col].mean()
                    if abs(mean_val) > FLOAT_TOLERANCE:
                        stability[f"{param}_cv"] = hist_df[col].std() / abs(mean_val)
                    else:
                        stability[f"{param}_cv"] = np.nan
                else:
                    stability[f"{param}_cv"] = np.nan

            stability["n_segments_valid"] = len(param_history)
        else:
            stability = {"n_segments_valid": len(param_history)}

        self._last_oos_results = oos_df
        self._last_stability = stability
        self._last_param_history = param_history

        return oos_df, stability, param_history

    def sensitivity_analysis(
        self,
        stop_method: str,
        best_params: Dict[str, Any],
        close_col: str = "close",
        variance: float = 0.20,
        opt_metric: str = "convex",
        extra_grids: Optional[Dict[str, List[Any]]] = None,
    ) -> Tuple[float, pd.DataFrame]:
        """
        Evaluate robustness around best parameters.

        Creates a grid around the best parameters (default +/- 20%) and
        evaluates performance across the neighborhood.

        Args:
            stop_method: Stop-loss method to use
            best_params: Dict of best parameter values
            close_col: Close price column name
            variance: Fraction to vary parameters (default 0.20 = +/- 20%)
            opt_metric: Metric to evaluate (column name)
            extra_grids: Optional additional grid values to include

        Returns:
            Tuple of:
                - plateau_ratio_pct: (avg performance / peak performance) * 100
                - results_df: Full grid search results

        Raises:
            ValueError: If best_params is empty or not found in grid
            RuntimeError: If grid search returns no results
        """
        if not best_params:
            raise ValueError("best_params must be non-empty dict")

        # Generate tight grid around best values
        param_grid = extra_grids.copy() if extra_grids else {}
        for param, best_val in best_params.items():
            if param not in param_grid:
                if isinstance(best_val, (int, float)):
                    factors = [1 - variance, 1, 1 + variance]
                    if isinstance(best_val, int):
                        param_grid[param] = sorted(set(
                            int(best_val * f) for f in factors
                        ))
                    else:
                        param_grid[param] = [round(best_val * f, 6) for f in factors]
                else:
                    param_grid[param] = [best_val]

        results = self.run_grid_search(
            segment_data=self.data,
            param_grid=param_grid,
            segment_idx=-1,
        )

        if results.empty:
            raise RuntimeError("No valid results from sensitivity grid")

        # Validate opt_metric exists
        if opt_metric not in results.columns:
            raise KeyError(
                f"Metric '{opt_metric}' not found. Available: {results.columns.tolist()}"
            )

        # Find peak using tolerance-based comparison for floats
        def matches_params(row):
            for k, v in best_params.items():
                if k not in row:
                    return False
                if isinstance(v, float):
                    if not np.isclose(row[k], v, rtol=1e-5):
                        return False
                elif row[k] != v:
                    return False
            return True

        mask = results.apply(matches_params, axis=1)
        peak = results.loc[mask, opt_metric]

        if peak.empty:
            raise ValueError(
                f"Best parameters {best_params} not found in sensitivity grid. "
                "Check variance/rounding or provide exact values in extra_grids."
            )

        peak_val = peak.iloc[0]
        avg_val = results[opt_metric].mean()

        if abs(peak_val) > FLOAT_TOLERANCE:
            plateau_ratio_pct = (avg_val / peak_val) * 100
        else:
            plateau_ratio_pct = np.nan

        return plateau_ratio_pct, results

    def compare_signals(
        self,
        signals: Union[str, Sequence[str]],
        stop_method: str,
        param_grid: Dict[str, Iterable[Any]],
        key_metrics: Sequence[str] = ("convex", "sharpe", "sortino", "profit_factor", "win_rate"),
        sort_by: str = "convex_mean",
        ascending: bool = False,
        **walk_forward_kwargs: Any
    ) -> pd.DataFrame:
        """
        Compare optimization results across multiple signals.

        Note: This method is currently disabled as rolling_walk_forward
        does not support multiple signals. Use rolling_walk_forward
        directly for single-signal optimization.

        Args:
            signals: Signal column name(s) - currently not used
            stop_method: Stop-loss method to use
            param_grid: Parameter grid for optimization
            key_metrics: Metrics to include in comparison
            sort_by: Column to sort results by
            ascending: Sort order
            **walk_forward_kwargs: Additional args for rolling_walk_forward

        Returns:
            DataFrame with comparison results (empty if no valid results)
        """
        if isinstance(signals, str):
            signals = [signals]

        logger.warning(
            "compare_signals currently runs single optimization. "
            "Signal comparison not yet implemented."
        )

        # Run optimization (single signal only for now)
        oos_df, stability, param_hist = self.rolling_walk_forward(
            stop_method=stop_method,
            param_grid=param_grid,
            **walk_forward_kwargs
        )

        if oos_df.empty:
            return pd.DataFrame()

        # Build comparison row
        agg = oos_df.mean(numeric_only=True).to_dict()
        agg_median = oos_df.median(numeric_only=True).to_dict()

        row = {
            "signal": signals[0] if signals else "default",
            "n_segments": stability.get("n_segments_valid", 0),
        }

        # Core metrics - mean and median
        for m in key_metrics:
            if m in agg:
                row[f"{m}_mean"] = agg[m]
            if m in agg_median:
                row[f"{m}_median"] = agg_median[m]

        # Stability (coefficient of variation for params)
        for param in param_grid.keys():
            cv_key = f"{param}_cv"
            if cv_key in stability:
                row[cv_key] = stability[cv_key]

        # Best params from last segment
        if param_hist:
            last_best = param_hist[-1]["params"]
            for param, value in last_best.items():
                row[f"last_{param}"] = value

        df = pd.DataFrame([row])

        if not df.empty:
            df = df.set_index("signal")
            numeric_cols = df.select_dtypes(include="number").columns
            df[numeric_cols] = df[numeric_cols].round(4)

            if sort_by in df.columns:
                df = df.sort_values(sort_by, ascending=ascending)

        return df
