import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Iterable, Union, Sequence
import itertools
from joblib import Parallel, delayed
import os
from tqdm import tqdm  # optional for progress


def _worker_evaluate(
    segment_data: pd.DataFrame,
    signal: str,
    stop_method: str,
    close_col: str,
    segment_idx: int,
    param_kwargs: dict,
    equity_func: callable,
    calc: 'StopLossCalculator',
) -> dict:
    """Standalone worker — avoids capturing class state (Windows pickling savior)."""
    calc.data = segment_data.copy()  # defensive — segment might be view

    df_with_stops = calc.get_stop_loss(
        signal=signal,
        method=stop_method,
        price_col=close_col,
        **param_kwargs,
    )

    metrics = equity_func(df_with_stops, signal, segment_idx, close_col)

    if isinstance(metrics, pd.Series):
        metrics = metrics.to_dict()
    elif not isinstance(metrics, dict):
        raise TypeError(f"equity_func returned {type(metrics)}, expected dict/Series")

    metrics.update(param_kwargs)
    return metrics


class StrategyOptimizer:
    """
    Optimizer for parameter tuning of stop-loss strategies from StopLossCalculator
    using in-sample grid search + out-of-sample walk-forward analysis.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        calculator: 'StopLossCalculator',
        equity_func: callable,
    ):
        """
        Args:
            data: Full historical OHLC + signal DataFrame (index = time)
            calculator: Initialized StopLossCalculator instance
            equity_func: Callable that takes (df_with_stops, signal_col, segment_idx, close_col)
                         and returns pd.Series or dict with performance metrics
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("data must be a non-empty pandas DataFrame")
        if not callable(equity_func):
            raise TypeError("equity_func must be callable")

        self.data = data.copy()  # defensive copy
        self.calc = calculator
        self.equity_func = equity_func

        self._last_oos_results: Optional[pd.DataFrame] = None
        self._last_stability: Optional[Dict[str, float]] = None
        self._last_param_history: List[Dict[str, Any]] = []

    def _evaluate_params(
        self,
        segment_data: pd.DataFrame,
        signal: str,
        stop_method: str,
        close_col: str,
        segment_idx: int = -1,
        **param_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Core evaluation step: set data → compute stops via get_stop_loss → run equity function.
        """
        if not stop_method:
            raise ValueError("stop_method must be specified")

        self.calc.data = segment_data
        print(f"Calling {stop_method} with kwargs: {param_kwargs}")
        df_with_stops = self.calc.get_stop_loss(
            signal=signal,
            method=stop_method,
            price_col=close_col if stop_method in {"atr", "moving_average", "volatility_std", "fixed_percentage"} else None,
            **param_kwargs,
        )
        metrics = self.equity_func(df_with_stops, signal, segment_idx, close_col)
        print(f"Signal: {signal} | Segment: {segment_idx} | Raw metrics from equity_func: {metrics}")
        
        # Normalize return type to dict
        if isinstance(metrics, pd.Series):
            metrics = metrics.to_dict()
        elif not isinstance(metrics, dict):
            raise TypeError("equity_func must return dict or pd.Series")

        metrics.update(param_kwargs)  # add all params for tracking
        return metrics

    def run_grid_search(
        self,
        segment_data: pd.DataFrame,
        signal: str,
        stop_method: str,
        param_grid: Dict[str, Iterable[Any]],
        segment_idx: int = -1,
        close_col: str = "close",
        n_jobs: int = 1,
        progress: bool = False,
        backend: str = "loky",               # NEW: configurable
        prefer: str = "processes",           # NEW
    ) -> pd.DataFrame:
        """
        Grid search with Windows-friendly defaults.
        """
        if not param_grid:
            raise ValueError("param_grid required")
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        if n_jobs < 0:
            n_jobs = -1

        param_names = list(param_grid.keys())
        param_values_lists = [list(v) for v in param_grid.values()]  # materialize

        all_combos = list(itertools.product(*param_values_lists))

        if not all_combos:
            return pd.DataFrame()

        # Prepare tasks — pass self.calc explicitly
        tasks = [
            delayed(_worker_evaluate)(
                segment_data=segment_data,
                signal=signal,
                stop_method=stop_method,
                close_col=close_col,
                segment_idx=segment_idx,
                param_kwargs=dict(zip(param_names, combo)),
                equity_func=self.equity_func,      # function reference — usually picklable
                calc=self.calc,                    # pass instance explicitly
            )
            for combo in all_combos
        ]

        # Windows tip: try "threading" first if you get pickling errors
        effective_backend = backend
        if os.name == "nt" and backend == "loky":
            # You can force threading here for debugging
            # effective_backend = "threading"   # uncomment to test GIL-safe fallback
            pass

        try:
            results = Parallel(
                n_jobs=n_jobs,
                backend=effective_backend,
                prefer=prefer,
                verbose=10 if n_jobs > 1 else 0,
                # timeout=600,                  # optional: kill hung workers after 10 min
            )(tasks)
        except Exception as e:
            if "pickle" in str(e).lower() and effective_backend == "loky":
                print("Pickling error detected on Windows. Try backend='threading' or simplify equity_func.")
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
                stop_method=stop_method,
                param_grid=param_grid,
                segment_idx=i,
                close_col=close_col,
                n_jobs=n_jobs,
                progress=progress,
                backend="loky",
                prefer="processes",
            )
            
            print(f"\n=== IS results for signal {signal} segment {i} ===")
            print(is_results[["window", "multiplier", "convex"]].sort_values("convex", ascending=False).head(5))
            print("Convex values unique:", is_results["convex"].unique())

            if is_results.empty:
                continue

            # Pick best by opt_metric (ascending assumed)
            best = is_results.loc[is_results[opt_metric].idxmax()]
            best_params = {p: best[p] for p in param_grid}  # extract only grid params

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
        peak = results.loc[mask, "convex"]

        if peak.empty:
            raise ValueError("Best parameters not found in sensitivity grid — check variance/rounding")

        peak_val = peak.iloc[0]
        avg_val = results["convex"].mean()

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

# class StrategyOptimizer:
#     def __init__(self, data: pd.DataFrame, calculator: StopLossCalculator, equity_func):
#         """
#         Args:
#             data: The full OHLC DataFrame.
#             calculator: An instance of StopLossCalculator.
#             equity_func: Your custom function that returns a 1-row, 4-col DataFrame.
#         """
#         self.data = data
#         self.calc = calculator
#         self.equity_func = equity_func
#         self.optimization_results = pd.DataFrame()
#         self.best_params = {}

   
#     def run_grid_search(self, is_data, signal, windows, multipliers, i, price_col = 'close'):
#         """Performs a standard grid search on a specific data segment."""
#         results = []
#         self.calc.data = is_data
        
#         for w, m in itertools.product(windows, multipliers):
#             temp_df = self.calc.atr_stop_loss(signal=signal, window=w, multiplier=m, price_col=price_col)
#             row = self.equity_func(temp_df, signal, i, price_col = price_col)
#             row.update({'window': w, 'multiplier': m})
#             results.append(row)
            
#         return pd.DataFrame(results)

#     def rolling_walk_forward(self, signal, close_col, windows, multipliers, n_segments=4):
#         """Executes a rolling WFA and returns OOS metrics and parameter stability."""
#         segment_size = len(self.data) // (n_segments + 1)
#         print(f"Debug: Data Length: {len(self.data)}, Segments: {n_segments}")
#         oos_results = []
#         param_history = []

#         for i in range(n_segments):
#             # Define Splits
#             is_data = self.data.iloc[i * segment_size : (i + 1) * segment_size]
#             oos_data = self.data.iloc[(i + 1) * segment_size : (i + 2) * segment_size]

#             # In-Sample Optimization
#             is_df = self.run_grid_search(is_data, signal, windows, multipliers, i, price_col = close_col)
#             best_row = is_df.sort_values('convex', ascending=False).iloc[0]
            
#             w_best, m_best = int(best_row['window']), best_row['multiplier']
#             param_history.append({'segment': i+1, 'window': w_best, 'multiplier': m_best})

#             # Out-of-Sample Validation
#             self.calc.data = oos_data
#             final_oos = self.calc.atr_stop_loss(signal, window=w_best, multiplier=m_best, price_col=close_col)
#             oos_metrics = self.equity_func(final_oos, signal, i, close_col)
#             oos_metrics['segment'] = i + 1
#             # oos_metrics.to_excel(str(i+'output.xlsx'))
#             # oos_metrics['w_best'] = w_best
#             # oos_metrics['m_best'] = m_best
#             oos_results.append(oos_metrics)

#             # # Update the placeholder every loop
#             # self.final_best_params = {
#             #     'window': w_best,
#             #     'multiplier': m_best,
#             #     'segment_index': i
#             # }

#         # Calculate Stability
#         history_df = pd.DataFrame(param_history)
#         stability = {
#             'window_cv': history_df['window'].std() / history_df['window'].mean(),
#             'multiplier_cv': history_df['multiplier'].std() / history_df['multiplier'].mean()
#         }
        
#         # return pd.DataFrame(oos_results)
#         return pd.DataFrame(oos_results), stability, param_history

#     def sensitivity_analysis(self, signal, best_w, best_m, variance=0.2):
#         """Tests the 'plateau' around the optimal parameters."""
#         w_range = [int(best_w * r) for r in [1-variance, 1, 1+variance]]
#         m_range = [round(best_m * r, 2) for r in [1-variance, 1, 1+variance]]
        
#         # Test on full data
#         self.calc.data = self.data
#         results = self.run_grid_search(self.data, signal, w_range, m_range)
        
#         peak_equity = results[(results['window']==best_w) & (results['multiplier']==best_m)]['convex'].iloc[0]
#         avg_equity = results['convex'].mean()
        
#         return (avg_equity / peak_equity) * 100, results