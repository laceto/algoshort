import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Iterable, Union, Sequence
import itertools
from joblib import Parallel, delayed
import os
from tqdm import tqdm

import logging
from algoshort.regime_fc import RegimeFC
from algoshort.returns import ReturnsCalculator
from algoshort.stop_loss import StopLossCalculator   # your stop-loss module
from algoshort.position_sizing import PositionSizing  # your position sizing module
from algoshort.utils import load_config

def get_equity(
    segment_df: pd.DataFrame,
    signal: str = 'rrg',
    segment_idx: int = 0,
    config_path: str = 'config_regime.json',
    price_col: str = 'close',
    stop_method: str = 'atr',               # ← NEW: choose any supported method
    inplace: bool = False,
    **stop_kwargs: Any                      # ← NEW: method-specific kwargs
) -> dict:
    """
    Process one segment independently with flexible stop-loss method.
    
    Args:
        segment_df: Raw price data for this segment only
        signal: Name of the signal column to create/use ('rrg')
        segment_idx: For logging / filename
        config_path: Path to regime config JSON
        price_col: Main price column ('close', 'rclose', etc.)
        stop_method: Stop-loss method to use (see StopLossCalculator for supported names)
                     e.g. 'atr', 'fixed_percentage', 'breakout_channel', 'moving_average',
                     'volatility_std', 'support_resistance', 'classified_pivot'
        inplace: If True, modify segment_df; else work on copy (recommended False)
        **stop_kwargs: Passed directly to the chosen stop-loss method
                       e.g. window=14, multiplier=2.0, percentage=0.05, etc.
    
    Returns:
        dict: Final equity metrics + metadata
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"\n=== Processing segment {segment_idx} | signal='{signal}' | stop_method='{stop_method}' ===")
    print(f"Input columns: {segment_df.columns.tolist()}")
    print(f"Rows: {len(segment_df)}")
    if stop_kwargs:
        print(f"Stop-loss kwargs: {stop_kwargs}")

    df = segment_df if inplace else segment_df.copy()

    # 1. Compute signal column 'rrg'
    config = load_config(config_path)
    print("→ Computing signal 'rrg'...")
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

    # if signal not in df.columns:
    #     raise KeyError(f"Signal column '{signal}' was not created by RegimeFC")

    # print(f"Signal '{signal}' value counts:\n{df[signal].value_counts()}")

    # 2. Compute returns
    print("→ Computing returns...")
    returns_calc = ReturnsCalculator(ohlc_stock=df)
    df = returns_calc.get_returns(
        df=df,
        signal='rrg',
        relative=config['returns']['relative'],
        inplace=False
    )

    # daily_change_col = f"{signal}_chg1D_fx"
    # if daily_change_col not in df.columns:
    #     raise KeyError(f"Daily change column '{daily_change_col}' not created")

    # 3. Compute stop-loss using selected method + kwargs
    print(f"→ Computing stop-loss using method '{stop_method}'...")
    sl_calc = StopLossCalculator(df)
    
    df = sl_calc.get_stop_loss(
        signal=signal,
        method=stop_method,
        **stop_kwargs  # ← passes all kwargs to the chosen method
    )

    stop_loss_col = f"{signal}_stop_loss"
    if stop_loss_col not in df.columns:
        raise KeyError(f"Stop-loss column '{stop_loss_col}' not created by method '{stop_method}'")
    
    # Reset index to ensure sequential 0-based indexing
    df = df.reset_index(drop=True)  # ← ADD THIS LINE
    # 4. Compute equity curves
    print("→ Computing equity curves...")

    pos = PositionSizing(
        tolerance=-0.2,
        mn=0.01,
        mx=0.10,
        equal_weight=0.25,
        avg=0.05,
        lot=100,
        initial_capital=100000  # Optional, defaults to 100000
    )

# Calculate shares with custom column names
    df = pos.calculate_shares(
        df=df,
        daily_chg=signal + '_chg1D_fx',
        sl=stop_loss_col,
        signal=signal,
        close=price_col
    )

    # Save full output for inspection
    output_file = f"segment_{segment_idx}_{signal}_{stop_method}_output.xlsx"
    df.to_excel(output_file)
    print(f"Saved full segment output: {output_file}")

    # Extract final equity metrics (last row)
    metrics_cols = ['date', signal + '_constant', signal + '_concave', signal + '_convex', signal + '_equal_weight']
    available_cols = [col for col in metrics_cols if col in df.columns]

    if not available_cols:
        raise ValueError("No equity curve columns found after position sizing")

    last_row = df[available_cols].iloc[-1].to_dict()

    # Add useful metadata
    last_row['segment_idx'] = segment_idx
    last_row['start_date'] = df['date'].iloc[0]
    last_row['end_date'] = df['date'].iloc[-1]
    last_row['signal'] = signal
    last_row['stop_method'] = stop_method
    last_row['stop_kwargs'] = stop_kwargs
    last_row['rows_processed'] = len(df)

    print(f"Segment {segment_idx} final equity ({stop_method}):")
    for k, v in last_row.items():
        if k not in ['date', 'start_date', 'end_date', 'stop_kwargs']:
            if isinstance(v, (int, float)):
                print(f"  {k:12}: {v:,.2f}")
            else:
                print(f"  {k:12}: {v}")

    return last_row


def _worker_evaluate(
    segment_data: pd.DataFrame,
    signal: str,
    segment_idx: int,
    param_kwargs: dict,
    equity_func: callable,
    config_path: str,  # ← NEW: pass path instead of dict
) -> dict:
    """Standalone worker — passes config_path to equity_func."""
    metrics = equity_func(
        segment_data,
        signal=signal,
        segment_idx=segment_idx,
        config_path=config_path,   # ← passed here
        **param_kwargs
    )

    if isinstance(metrics, pd.Series):
        metrics = metrics.to_dict()
    elif not isinstance(metrics, dict):
        raise TypeError(f"equity_func returned {type(metrics)}, expected dict/Series")

    metrics.update(param_kwargs)
    return metrics


class StrategyOptimizer:
    """
    Optimizer for parameter tuning — now uses config_path (JSON file path) instead of loaded dict.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        equity_func: callable,
        config_path: str,  # ← NEW: required path to config JSON
    ):
        """
        Args:
            data: Full historical OHLC DataFrame (index = time)
            equity_func: Callable that takes (segment_df, signal, segment_idx, config_path, **param_kwargs)
                         and returns dict with performance metrics + equity curves
            config_path: Path to the regime configuration JSON file
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("data must be a non-empty pandas DataFrame")
        if not callable(equity_func):
            raise TypeError("equity_func must be callable")
        if not isinstance(config_path, str) or not os.path.isfile(config_path):
            raise ValueError(f"config_path must be a valid file path: {config_path}")

        self.data = data.copy()  # defensive
        self.equity_func = equity_func
        self.config_path = config_path  # ← stored path

        self._last_oos_results: Optional[pd.DataFrame] = None
        self._last_stability: Optional[Dict[str, float]] = None
        self._last_param_history: List[Dict[str, Any]] = []

    def _evaluate_params(
        self,
        segment_data: pd.DataFrame,
        signal: str,
        segment_idx: int,
        **param_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Core evaluation: call equity_func with config_path.
        """
        metrics = self.equity_func(
            segment_data,
            signal=signal,
            segment_idx=segment_idx,
            config_path=self.config_path,   # ← passed here
            **param_kwargs
        )
        print(f"Signal: {signal} | Segment: {segment_idx} | Raw metrics from equity_func: {metrics}")

        if isinstance(metrics, pd.Series):
            metrics = metrics.to_dict()
        elif not isinstance(metrics, dict):
            raise TypeError("equity_func must return dict or pd.Series")

        metrics.update(param_kwargs)
        return metrics

    def run_grid_search(
        self,
        segment_data: pd.DataFrame,
        signal: str,
        param_grid: Dict[str, Iterable[Any]],
        segment_idx: int,
        n_jobs: int = 1,
        progress: bool = False,
        backend: str = "loky",
        prefer: str = "processes",
    ) -> pd.DataFrame:
        """
        Grid search — passes config_path to worker.
        """
        if not param_grid:
            raise ValueError("param_grid required")
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        if n_jobs < 0:
            n_jobs = -1

        param_names = list(param_grid.keys())
        param_values_lists = [list(v) for v in param_grid.values()]

        all_combos = list(itertools.product(*param_values_lists))

        if not all_combos:
            return pd.DataFrame()

        tasks = [
            delayed(_worker_evaluate)(
                segment_data=segment_data,
                signal=signal,
                segment_idx=segment_idx,
                param_kwargs=dict(zip(param_names, combo)),
                equity_func=self.equity_func,
                config_path=self.config_path,   # ← passed to worker
            )
            for combo in all_combos
        ]

        effective_backend = backend
        if os.name == "nt" and backend == "loky":
            pass  # keep or switch to threading if needed

        try:
            results = Parallel(
                n_jobs=n_jobs,
                backend=effective_backend,
                prefer=prefer,
                verbose=10 if n_jobs > 1 else 0,
            )(tasks)
        except Exception as e:
            if "pickle" in str(e).lower() and effective_backend == "loky":
                print("Pickling error on Windows. Try backend='threading'.")
            raise

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def rolling_walk_forward(
        self,
        signals: Union[str, Sequence[str]],
        stop_method: str,
        param_grid: Dict[str, Iterable[Any]],
        close_col: str = "close",
        n_segments: int = 4,
        n_jobs: int = 1,
        progress: bool = False,
        verbose: bool = False,
        opt_metric: str = "convex",
    ) -> Dict[str, Tuple[pd.DataFrame, Dict[str, float], List[Dict[str, Any]]]]:
        """
        Extended walk-forward that runs the full procedure once per signal column.

        Args:
            signals: Single signal column name or list of column names

        Returns:
            Dict[sig_name → (oos_df, stability_dict, param_history_list)]
        """
        if isinstance(signals, str):
            signals = [signals]

        if not all(isinstance(s, str) for s in signals):
            raise TypeError("signals must be str or sequence of str")

        missing = [s for s in signals if s not in self.data.columns]
        if missing:
            raise KeyError(f"Signal columns not found in data: {missing}")

        results = {}

        for sig in signals:
            if verbose:
                print(f"\n=== Processing signal: {sig} ===")

            # Inner call to perform single-signal WFA
            oos_df, stability, history = self._single_rolling_walk_forward(
                signal=sig,
                stop_method=stop_method,
                param_grid=param_grid,
                close_col=close_col,
                n_segments=n_segments,
                n_jobs=n_jobs,
                progress=progress,
                verbose=verbose,
                opt_metric=opt_metric,
            )

            results[sig] = (oos_df, stability, history)

            if verbose:
                print(f"  → Stability: {stability}")
                if oos_df is not None and not oos_df.empty:
                    print(f"  → OOS rows: {len(oos_df)}, mean {opt_metric}: {oos_df[opt_metric].mean():.4f}")

        return results

    def _single_rolling_walk_forward(
        self,
        signal: str,
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
        Internal single-signal walk-forward logic (extracted for DRY).
        """
        if n_segments < 2:
            raise ValueError("n_segments must be >= 2")

        n = len(self.data)
        segment_size = n // (n_segments + 1)
        if segment_size < 20:
            raise ValueError(f"Segment size too small ({segment_size} bars) — increase data or reduce n_segments")

        if verbose:
            print(f"Total bars: {n}  |  Segments: {n_segments}  |  Approx bars/segment: {segment_size}")

        oos_rows = []
        param_history = []

        for i in range(n_segments):
            train_start = i * segment_size
            train_end = (i + 1) * segment_size
            test_start = train_end
            test_end = (i + 2) * segment_size

            is_data = self.data.iloc[train_start:train_end]
            oos_data = self.data.iloc[test_start:test_end]

            if len(is_data) < 30 or len(oos_data) < 30:
                continue  # skip degenerate segments

            # In-sample optimization
            is_results = self.run_grid_search(
                segment_data=is_data,
                signal=signal,
                # stop_method=stop_method,
                param_grid=param_grid,
                segment_idx=i,
                # close_col=close_col,
                n_jobs=n_jobs,
                progress=progress,
                backend="loky",
                prefer="processes",
            )
            
            print(f"\n=== IS results for signal {signal} segment {i} ===")
            print(is_results[["window", "multiplier", signal + "_convex"]].sort_values(signal + "_convex", ascending=False).head(5))
            print("Convex values unique:", is_results[signal + "_convex"].unique())

            if is_results.empty:
                continue

            # Pick best by opt_metric (ascending assumed)
            best = is_results.loc[is_results[opt_metric].idxmax()]
            best_params = {p: best[p] for p in param_grid}  # extract only grid params
            print('qui')
            param_history.append({
                "segment": i + 1,
                "params": best_params,
                "is_metric": best[opt_metric],
            })

            # Out-of-sample evaluation with best params
            oos_row = self._evaluate_params(
                segment_data=oos_data,
                signal=signal,
                stop_method=stop_method,
                close_col=close_col,
                segment_idx=i,
                **best_params
            )
            oos_row["segment"] = i + 1
            oos_rows.append(oos_row)

        oos_df = pd.DataFrame(oos_rows)

        if len(param_history) >= 2:
            # Flatten without record_path
            hist_df = pd.json_normalize(param_history)
            
            stability = {}
            for param in param_grid:                   # param_grid.keys() = ['window', 'multiplier']
                col = f"params.{param}"                # how json_normalize names nested keys
                if col in hist_df.columns:
                    mean_val = hist_df[col].mean()
                    if mean_val != 0:
                        stability[f"{param}_cv"] = hist_df[col].std() / mean_val
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
        signal: str,
        stop_method: str,
        best_params: Dict[str, Any],
        close_col: str = "close",
        variance: float = 0.20,
        extra_grids: Optional[Dict[str, List[Any]]] = None,
    ) -> Tuple[float, pd.DataFrame]:
        """
        Evaluate robustness around best parameters for any method.

        Returns:
            (plateau_ratio_pct, full_grid_results_df)
            plateau_ratio = (avg performance / peak performance) × 100
        """
        if not best_params:
            raise ValueError("best_params must be non-empty dict")

        # Generate tight grid around bests
        param_grid = extra_grids or {}
        for param, best_val in best_params.items():
            if isinstance(best_val, (int, float)):
                factors = [1 - variance, 1, 1 + variance]
                if isinstance(best_val, int):
                    param_grid[param] = [int(best_val * f) for f in factors]
                else:
                    param_grid[param] = [round(best_val * f, 3) for f in factors]
            else:
                param_grid[param] = [best_val]  # non-numeric: just center

        results = self.run_grid_search(
            segment_data=self.data,
            signal=signal,
            stop_method=stop_method,
            param_grid=param_grid,
            close_col=close_col,
            segment_idx=-1,  # full data
        )

        if results.empty:
            raise RuntimeError("No valid results from sensitivity grid")

        # Find peak at exact best_params
        mask = np.all([results[k] == v for k, v in best_params.items()], axis=0)
        peak = results.loc[mask, signal + "_convex"]

        if peak.empty:
            raise ValueError("Best parameters not found in sensitivity grid — check variance/rounding")

        peak_val = peak.iloc[0]
        avg_val = results[signal + "_convex"].mean()

        plateau_ratio_pct = (avg_val / peak_val) * 100 if peak_val != 0 else np.nan

        return plateau_ratio_pct, results

    def compare_signals(
        self,
        signals: Union[str, Sequence[str]],
        stop_method: str,
        param_grid: Dict[str, Iterable[Any]],
        key_metrics: Sequence[str] = ("convex", "sharpe", "sortino", "profit_factor", "win_rate"),
        sort_by: str = "convex",
        ascending: bool = False,
        **walk_forward_kwargs
    ) -> pd.DataFrame:
        """
        Run optimization for multiple signals and return a clean comparison table.

        Returns:
            DataFrame with one row per signal, columns = metrics + stability info
        """
        if isinstance(signals, str):
            signals = [signals]

        multi_results = self.rolling_walk_forward(
            signals=signals,
            stop_method=stop_method,
            param_grid=param_grid,
            **walk_forward_kwargs
        )

        rows = []

        for sig, (oos_df, stability, param_hist) in multi_results.items():
            if oos_df.empty:
                continue

            agg = oos_df.mean(numeric_only=True).to_dict()
            agg_median = oos_df.median(numeric_only=True).to_dict()

            row = {
                "signal": sig,
                "n_segments": stability.get("n_segments_valid", 0),
            }

            # Core metrics - mean
            for m in key_metrics:
                if m in agg:
                    row[f"{m}_mean"]   = agg[m]
                if m in agg_median:
                    row[f"{m}_median"] = agg_median[m]

            # Stability (coefficient of variation for params)
            for param, cv_key in [("window", "window_cv"), ("multiplier", "multiplier_cv")]:
                if cv_key in stability:
                    row[f"{param}_cv"] = stability[cv_key]

            # Best param from last (or most frequent) segment - optional
            if param_hist:
                last_best = param_hist[-1]["params"]
                row["last_window"]     = last_best.get("window")
                row["last_multiplier"] = last_best.get("multiplier")

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("signal")

        # Sort & round for readability
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)

        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].round(4)

        return df

