# import pandas as pd
# from typing import Dict, Tuple, List, Optional
# import numpy as np
# from joblib import Parallel, delayed
# from tqdm import tqdm  # optional - nice progress bar


# class ReturnsCalculator:
#     """
#     A class to calculate returns and equity curves for trading strategies based on OHLC data and regime signals.

#     Args:
#         ohlc_stock (pd.DataFrame): DataFrame with OHLC columns ['open', 'high', 'low', 'close']
#             or ['Open', 'High', 'Low', 'Close'] for absolute mode, and
#             ['r_open', 'r_high', 'r_low', 'r_close'] or ['rOpen', 'rHigh', 'rLow', 'rClose']
#             for relative mode.

#     Raises:
#         KeyError: If required OHLC columns are not found.
#     """

#     def __init__(self, ohlc_stock: pd.DataFrame):
#         self.ohlc_stock = ohlc_stock
#         self._cache: Dict[str, pd.Series] = {}  # Cache for rolling calculations

#     @staticmethod
#     def _lower_upper_OHLC(data: pd.DataFrame, relative: bool = False) -> Tuple[str, str, str, str]:
#         """
#         Determines OHLC column names based on DataFrame columns and relative flag.

#         Args:
#             data (pd.DataFrame): DataFrame to check for OHLC columns.
#             relative (bool): If True, use relative OHLC columns (e.g., 'r_open'); else absolute (e.g., 'open').

#         Returns:
#             Tuple[str, str, str, str]: Column names for open, high, low, close.

#         Raises:
#             KeyError: If required OHLC columns are not found.
#         """
#         rel = 'r' if relative else ''
#         if 'Open' in data.columns:
#             ohlc = [f'{rel}Open', f'{rel}High', f'{rel}Low', f'{rel}Close']
#         elif 'open' in data.columns:
#             ohlc = [f'{rel}open', f'{rel}high', f'{rel}low', f'{rel}close']
#         else:
#             raise KeyError("No 'Open' or 'open' column found in DataFrame.")

#         for col in ohlc:
#             if col not in data.columns:
#                 raise KeyError(f"Column '{col}' not found in DataFrame.")
        
#         return tuple(ohlc)

#     def get_returns(self, df: pd.DataFrame, signal: str, relative: bool = False, inplace: bool = False) -> pd.DataFrame:
#         """
#         Calculates returns based on a regime signal, including daily changes, cumulative returns, and stop-loss levels.

#         Args:
#             df (pd.DataFrame): DataFrame containing OHLC data and the regime signal column.
#             signal (str): Name of the signal column (e.g., 'bo_5', 'tt_52', 'sma_358').
#             relative (bool): If True, use relative OHLC columns ('r_close', etc.); else absolute ('close', etc.). Defaults to False.
#             inplace (bool): If True, modify input DataFrame; else return a new one. Defaults to False.

#         Returns:
#             pd.DataFrame: DataFrame with additional columns:
#                 - '<signal>_chg1D': Daily price change based on lagged signal.
#                 - '<signal>_chg1D_fx': Duplicate of daily price change (for compatibility).
#                 - '<signal>_PL_cum': Cumulative sum of daily price changes.
#                 - '<signal>_PL_cum_fx': Duplicate of cumulative price changes (for compatibility).
#                 - '<signal>_returns': Daily percentage returns based on lagged signal.
#                 - '<signal>_log_returns': Daily log returns based on lagged signal.
#                 - '<signal>_cumul': Cumulative returns from log returns (exp(cumsum(log_returns)) - 1).

#         Raises:
#             KeyError: If required columns ('close'/'r_close', 'high'/'r_high', 'low'/'r_low', signal) are missing.
#             ValueError: If DataFrame is empty, or data types are non-numeric.
#         """
#         try:
#             # Validate inputs
#             if df.empty:
#                 raise ValueError("Input DataFrame is empty.")
            
#             # Get OHLC column names
#             _o, _h, _l, _c = self._lower_upper_OHLC(data=df, relative=relative)
            
#             # Validate signal column
#             if signal not in df.columns:
#                 raise KeyError(f"Signal column '{signal}' not found in DataFrame.")
            
#             # Validate numeric data
#             if not np.issubdtype(df[_c].dtype, np.number) or \
#             not np.issubdtype(df[_h].dtype, np.number) or \
#             not np.issubdtype(df[_l].dtype, np.number) or \
#             not np.issubdtype(df[signal].dtype, np.number):
#                 raise ValueError("OHLC and signal columns must contain numeric data.")

#             # Work on copy if not inplace
#             result_df = df if inplace else df.copy()
            
#             # Fill NaN in signal column with 0
#             signal_filled = result_df[signal].fillna(0)
            
#             # Pre-compute reusable intermediate values
#             close_prices = result_df[_c]
#             price_diff = close_prices.diff()
#             lagged_signal = signal_filled.shift()
            
#             # Calculate daily price changes
#             chg1D = price_diff * lagged_signal
            
#             # Calculate returns
#             pct_returns = close_prices.pct_change() * lagged_signal
#             log_returns = np.log(close_prices / close_prices.shift()) * lagged_signal
            
#             # Build new columns dictionary (avoids fragmentation)
#             new_columns = {
#                 f'{signal}_chg1D': chg1D,
#                 f'{signal}_chg1D_fx': chg1D,  # Duplicate for compatibility
#                 f'{signal}_PL_cum': chg1D.cumsum(),
#                 f'{signal}_PL_cum_fx': chg1D.cumsum(),  # Duplicate for compatibility
#                 f'{signal}_returns': pct_returns,
#                 f'{signal}_log_returns': log_returns,
#                 f'{signal}_cumul': np.exp(log_returns.cumsum()) - 1  # Vectorized exp, no apply
#             }
            
#             # Single assignment operation using assign (prevents fragmentation)
#             if inplace:
#                 # For inplace, update signal column first, then assign new columns
#                 result_df[signal] = signal_filled
#                 for col_name, col_data in new_columns.items():
#                     result_df[col_name] = col_data
#             else:
#                 # For non-inplace, use assign for cleaner code
#                 result_df = result_df.assign(**{signal: signal_filled, **new_columns})

#             return result_df

#         except Exception as e:
#             raise ValueError(f"Error computing returns: {str(e)}")
        
#     def get_returns_multiple(
#             self,
#             df: pd.DataFrame,
#             signals: List[str],
#             relative: bool = False,
#             n_jobs: int = -1,           # -1 = use all cores
#             verbose: bool = True,
#             inplace: bool = False,
#         ) -> pd.DataFrame:
#             """
#             Calculate returns for multiple signals in parallel.

#             Args:
#                 df: Input OHLC + signals DataFrame
#                 signals: List of signal column names
#                 relative: Use relative OHLC columns?
#                 n_jobs: Number of parallel jobs (-1 = all cores)
#                 verbose: Show progress bar
#                 inplace: Modify input df instead of returning new one

#             Returns:
#                 DataFrame with all original columns + new columns for every signal
#             """
#             if not signals:
#                 return df if inplace else df.copy()

#             # Fail fast - validate all signals exist
#             missing = [s for s in signals if s not in df.columns]
#             if missing:
#                 raise KeyError(f"Missing signal columns: {missing}")

#             # We'll collect all new columns from parallel workers
#             def _compute_one_signal(sig: str) -> dict:
#                 # Create minimal working copy (only needed columns)
#                 # This reduces memory pressure in parallel workers
#                 working_df = df[[self._lower_upper_OHLC(df, relative)[3], sig]].copy()

#                 result = self.get_returns(
#                     df=working_df,
#                     signal=sig,
#                     relative=relative,
#                     inplace=False,  # always return new → safer in parallel
#                 )

#                 # Return only the columns we actually generated
#                 prefix = f"{sig}_"
#                 new_cols = [c for c in result.columns if c.startswith(prefix)]
#                 return {c: result[c] for c in new_cols}

#             # ── Parallel execution ───────────────────────────────────────
#             parallel = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)

#             if verbose:
#                 results = parallel(
#                     delayed(_compute_one_signal)(sig)
#                     for sig in tqdm(signals, desc="Computing signals", unit="sig")
#                 )
#             else:
#                 results = parallel(delayed(_compute_one_signal)(sig) for sig in signals)

#             # Merge all results efficiently
#             all_new_columns = {}
#             for res in results:
#                 all_new_columns.update(res)

#             # Final assembly - single assignment
#             if inplace:
#                 df = df.assign(**all_new_columns)
#                 return df
#             else:
#                 return df.assign(**all_new_columns)


# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple, List, Optional
# from joblib import Parallel, delayed
# from tqdm import tqdm  # optional: nice progress bar


# class ReturnsCalculator:
#     """
#     Calculates returns and equity curves for trading strategies based on OHLC data and regime signals.

#     Supports both absolute and relative OHLC columns, controlled by the `relative` flag.

#     Args:
#         ohlc_stock: DataFrame containing OHLC and signal columns
#         open_col: Base name of the open column (default: 'open')
#         high_col: Base name of the high column (default: 'high')
#         low_col: Base name of the low column (default: 'low')
#         close_col: Base name of the close column (default: 'close')
#         relative_prefix: Prefix used for relative columns (default: 'r')

#     Raises:
#         KeyError: If any required OHLC columns are missing
#     """

#     def __init__(
#         self,
#         ohlc_stock: pd.DataFrame,
#         open_col: str = "open",
#         high_col: str = "high",
#         low_col: str = "low",
#         close_col: str = "close",
#         relative_prefix: str = "r",
#     ):
#         self.ohlc_stock = ohlc_stock
#         self._cache: Dict[str, pd.Series] = {}

#         self._base_cols = (open_col, high_col, low_col, close_col)
#         self._relative_prefix = relative_prefix

#         # Pre-build and validate both mappings
#         self._col_mappings: Dict[bool, Tuple[str, str, str, str]] = {
#             False: self._base_cols,
#             True: tuple(f"{relative_prefix}{base}" for base in self._base_cols),
#         }

#         # Fail fast: validate all possible columns exist
#         for relative, cols in self._col_mappings.items():
#             missing = [c for c in cols if c not in ohlc_stock.columns]
#             if missing:
#                 raise KeyError(
#                     f"Missing OHLC columns for relative={relative}: {missing}\n"
#                     f"Available columns: {list(ohlc_stock.columns)}\n"
#                     f"Expected (base): {self._base_cols}"
#                 )

#     def _get_ohlc_columns(self, relative: bool = False) -> Tuple[str, str, str, str]:
#         """Return pre-validated OHLC column names based on relative flag."""
#         return self._col_mappings[relative]

#     def get_returns(
#         self,
#         df: pd.DataFrame,
#         signal: str,
#         relative: bool = False,
#         inplace: bool = False,
#     ) -> pd.DataFrame:
#         """
#         Calculate returns, cumulative PL, log returns, etc. for a single signal.

#         Args:
#             df: DataFrame with OHLC and signal column
#             signal: Name of the signal column
#             relative: If True, use relative-prefixed OHLC columns
#             inplace: Modify df in place instead of returning new copy

#         Returns:
#             DataFrame with added return-related columns
#         """
#         if df.empty:
#             raise ValueError("Input DataFrame is empty.")

#         # Retrieve validated column names
#         _o, _h, _l, _c = self._get_ohlc_columns(relative=relative)

#         if signal not in df.columns:
#             raise KeyError(f"Signal column '{signal}' not found in DataFrame.")

#         # Basic type checks (fail fast)
#         required_cols = [_c, signal]
#         if not all(np.issubdtype(df[col].dtype, np.number) for col in required_cols):
#             raise ValueError("Close and signal columns must be numeric.")

#         result_df = df if inplace else df.copy()

#         # Fill NaN signals with 0 (neutral / no position)
#         signal_filled = result_df[signal].fillna(0)

#         close_prices = result_df[_c]
#         price_diff = close_prices.diff()
#         lagged_signal = signal_filled.shift()

#         chg1D = price_diff * lagged_signal
#         pct_returns = close_prices.pct_change() * lagged_signal
#         log_returns = np.log1p(close_prices.pct_change()) * lagged_signal   # safer than log(c / c.shift())

#         new_columns = {
#             f"{signal}_chg1D": chg1D,
#             f"{signal}_chg1D_fx": chg1D,  # legacy compatibility
#             f"{signal}_PL_cum": chg1D.cumsum(),
#             f"{signal}_PL_cum_fx": chg1D.cumsum(),
#             f"{signal}_returns": pct_returns,
#             f"{signal}_log_returns": log_returns,
#             f"{signal}_cumul": np.exp(log_returns.cumsum()) - 1,
#         }

#         if inplace:
#             result_df[signal] = signal_filled
#             result_df.assign(**new_columns, inplace=True)
#         else:
#             result_df = result_df.assign(**{signal: signal_filled, **new_columns})

#         return result_df

#     def get_returns_multiple(
#         self,
#         df: pd.DataFrame,
#         signals: List[str],
#         relative: bool = False,
#         n_jobs: int = -1,
#         verbose: bool = True,
#         inplace: bool = False,
#     ) -> pd.DataFrame:
#         """
#         Compute returns for multiple signals in parallel.

#         Args:
#             signals: List of signal column names
#             relative: Use relative OHLC columns for all signals
#             n_jobs: Parallel jobs (-1 = all cores)
#             verbose: Show progress bar
#             inplace: Modify input df

#         Returns:
#             DataFrame with original + all new signal columns
#         """
#         if not signals:
#             return df if inplace else df.copy()

#         missing_signals = [s for s in signals if s not in df.columns]
#         if missing_signals:
#             raise KeyError(f"Missing signal columns: {missing_signals}")

#         def _compute_one_signal(sig: str) -> Dict[str, pd.Series]:
#             # Minimal working copy → reduces memory pressure in parallel
#             close_col = self._get_ohlc_columns(relative)[3]
#             working_df = df[[close_col, sig]].copy()

#             result = self.get_returns(
#                 df=working_df,
#                 signal=sig,
#                 relative=relative,  # same mode for all
#                 inplace=False,
#             )

#             prefix = f"{sig}_"
#             new_cols = {c: result[c] for c in result.columns if c.startswith(prefix)}
#             return new_cols

#         # ── Parallel execution ───────────────────────────────────────────────
#         parallel = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)

#         if verbose:
#             results = parallel(
#                 delayed(_compute_one_signal)(sig)
#                 for sig in tqdm(signals, desc="Computing signals", unit="sig")
#             )
#         else:
#             results = parallel(delayed(_compute_one_signal)(sig) for sig in signals)

#         # Merge all generated columns
#         all_new_columns = {}
#         for res in results:
#             all_new_columns.update(res)

#         if inplace:
#             df = df.assign(**all_new_columns)
#             return df
#         else:
#             return df.assign(**all_new_columns)


import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from joblib import Parallel, delayed
import logging

# Module-level logger (best practice)
logger = logging.getLogger(__name__)


class ReturnsCalculator:
    """
    Calculates returns and equity curves for trading strategies based on OHLC data and regime signals.

    Supports both absolute and relative OHLC columns, controlled by the `relative` flag.

    Args:
        ohlc_stock: DataFrame containing OHLC and signal columns
        open_col: Base name of the open column (default: 'open')
        high_col: Base name of the high column (default: 'high')
        low_col: Base name of the low column (default: 'low')
        close_col: Base name of the close column (default: 'close')
        relative_prefix: Prefix used for relative columns (default: 'r')
        logger: Optional logger instance (defaults to module logger)

    Raises:
        KeyError: If any required OHLC columns are missing
    """

    def __init__(
            self,
            ohlc_stock: pd.DataFrame,
            open_col: str = "open",
            high_col: str = "high",
            low_col: str = "low",
            close_col: str = "close",
            relative_prefix: str = "r",
            logger: Optional[logging.Logger] = None,  # ← new optional param
        ):
            self.ohlc_stock = ohlc_stock
            self._cache: Dict[str, pd.Series] = {}

            self._base_cols = (open_col, high_col, low_col, close_col)
            self._relative_prefix = relative_prefix

            # Set logger: prefer injected one, fall back to module logger
            self.logger = logger if logger is not None else logging.getLogger(__name__)

            # Pre-build and validate column mappings (your existing code)
            self._col_mappings: Dict[bool, Tuple[str, str, str, str]] = {
                False: self._base_cols,
                True: tuple(f"{relative_prefix}{base}" for base in self._base_cols),
            }

            # Fail-fast column validation (your existing code)
            for relative, cols in self._col_mappings.items():
                missing = [c for c in cols if c not in ohlc_stock.columns]
                if missing:
                    raise KeyError(
                        f"Missing OHLC columns for relative={relative}: {missing}\n"
                        f"Available: {list(ohlc_stock.columns)}\n"
                        f"Expected base: {self._base_cols}"
                    )

            self.logger.debug("ReturnsCalculator initialized – %d rows", len(ohlc_stock))

    def _get_ohlc_columns(self, relative: bool = False) -> Tuple[str, str, str, str]:
        """Return pre-validated OHLC column names based on relative flag."""
        return self._col_mappings[relative]

    def get_returns(
        self,
        df: pd.DataFrame,
        signal: str,
        relative: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate returns, cumulative PL, log returns, etc. for a single signal.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        _o, _h, _l, _c = self._get_ohlc_columns(relative=relative)

        if signal not in df.columns:
            raise KeyError(f"Signal column '{signal}' not found in DataFrame.")

        required_cols = [_c, signal]
        if not all(np.issubdtype(df[col].dtype, np.number) for col in required_cols):
            raise ValueError("Close and signal columns must be numeric.")

        result_df = df if inplace else df.copy()

        signal_filled = result_df[signal].fillna(0)

        close_prices = result_df[_c]
        price_diff = close_prices.diff()
        lagged_signal = signal_filled.shift()

        chg1D = price_diff * lagged_signal
        pct_returns = close_prices.pct_change() * lagged_signal
        log_returns = np.log1p(close_prices.pct_change()) * lagged_signal

        new_columns = {
            f"{signal}_chg1D": chg1D,
            f"{signal}_chg1D_fx": chg1D,           # legacy
            f"{signal}_PL_cum": chg1D.cumsum(),
            f"{signal}_PL_cum_fx": chg1D.cumsum(),
            f"{signal}_returns": pct_returns,
            f"{signal}_log_returns": log_returns,
            f"{signal}_cumul": np.exp(log_returns.cumsum()) - 1,
        }

        if inplace:
            result_df[signal] = signal_filled
            result_df.assign(**new_columns, inplace=True)
        else:
            result_df = result_df.assign(**{signal: signal_filled, **new_columns})

        self.logger.debug("Computed returns for signal '%s' (rows: %d)", signal, len(result_df))

        return result_df

    def get_returns_multiple(
        self,
        df: pd.DataFrame,
        signals: List[str],
        relative: bool = False,
        n_jobs: int = -1,
        verbose: bool = True,           # kept for compatibility — prefer logger level
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Compute returns for multiple signals in parallel using joblib.
        Progress reporting now uses logging instead of tqdm.
        """
        if not signals:
            self.logger.info("No signals provided — returning input DataFrame")
            return df if inplace else df.copy()

        missing_signals = [s for s in signals if s not in df.columns]
        if missing_signals:
            raise KeyError(f"Missing signal columns: {missing_signals}")

        def _compute_one_signal(sig: str) -> Dict[str, pd.Series]:
            close_col = self._get_ohlc_columns(relative)[3]
            # Minimal slice — helps memory in parallel workers
            working_df = df[[close_col, sig]].copy()

            result = self.get_returns(
                df=working_df,
                signal=sig,
                relative=relative,
                inplace=False,
            )

            prefix = f"{sig}_"
            new_cols = {c: result[c] for c in result.columns if c.startswith(prefix)}
            return new_cols

        # ── Parallel execution ───────────────────────────────────────────────
        parallel = Parallel(n_jobs=n_jobs, verbose=0)  # disable joblib built-in verbose

        if verbose and self.logger.isEnabledFor(logging.INFO):
            self.logger.info("Computing returns for %d signals (parallel, n_jobs=%d)", 
                            len(signals), n_jobs if n_jobs > 0 else "all cores")

            results = []
            total = len(signals)
            for i, res in enumerate(parallel(delayed(_compute_one_signal)(sig) for sig in signals), 1):
                results.append(res)
                if i % max(1, total // 20) == 0 or i == total:  # update ~5% steps
                    self.logger.info("Progress: %d/%d signals processed", i, total)
        else:
            results = parallel(delayed(_compute_one_signal)(sig) for sig in signals)

        # Merge results
        all_new_columns = {}
        for res in results:
            all_new_columns.update(res)

        if inplace:
            df = df.assign(**all_new_columns)
            result_df = df
        else:
            result_df = df.assign(**all_new_columns)

        self.logger.info("Finished computing %d signals → %d new columns added", 
                         len(signals), len(all_new_columns))

        return result_df