import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

class TripleMACrossoverRegime:
    """
    A class to detect stock regimes based on triple moving average (MA) crossover strategies using OHLC data.
    Supports SMA and EMA crossovers with three windows: short (st), medium (mt), long (lt).
    Computes pairwise crossovers (st vs mt, mt vs lt) and multiplies signals for final regime:
    - 1: Both crossovers bullish (st >= mt and mt >= lt).
    - -1: Both crossovers bearish (st <= mt and mt <= lt).
    - 0: Mixed, equal, or insufficient data.
    Includes calculated MA values in output DataFrame (e.g., sma_short_5, sma_medium_10, sma_long_20).

    Args:
        ohlc_stock (pd.DataFrame): DataFrame with OHLC columns ['open', 'high', 'low', 'close']
            or ['Open', 'High', 'Low', 'Close'] for absolute mode, and
            ['r_open', 'r_high', 'r_low', 'r_close'] or ['rOpen', 'rHigh', 'rLow', 'rClose']
            for relative mode.

    Raises:
        KeyError: If required OHLC columns are not found.
    """

    def __init__(self, ohlc_stock: pd.DataFrame):
        self.ohlc_stock = ohlc_stock
        self._cache: Dict[str, pd.Series] = {}  # Cache for MAs and crossover signals

    def _lower_upper_OHLC(self, relative: bool = False) -> Tuple[str, str, str, str]:
        """
        Determines OHLC column names based on DataFrame columns and relative flag.

        Args:
            relative (bool): If True, use relative OHLC columns (e.g., 'rclose'); else absolute (e.g., 'close').

        Returns:
            Tuple[str, str, str, str]: Column names for open, high, low, close.

        Raises:
            KeyError: If required OHLC columns are not found.
        """
        rel = 'r' if relative else ''
        if 'Open' in self.ohlc_stock.columns:
            ohlc = [f'{rel}Open', f'{rel}High', f'{rel}Low', f'{rel}Close']
        elif 'open' in self.ohlc_stock.columns:
            ohlc = [f'{rel}open', f'{rel}high', f'{rel}low', f'{rel}close']
        else:
            raise KeyError("No 'Open' or 'open' column found in DataFrame.")

        for col in ohlc:
            if col not in self.ohlc_stock.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        
        return tuple(ohlc)

    def _compute_ma_crossover(self, close_col: str, short_window: int, long_window: int, ma_type: str = 'sma') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Computes MA crossover regime for two windows and returns MA series.

        Args:
            close_col (str): Column name for close price.
            short_window (int): Short MA window.
            long_window (int): Long MA window.
            ma_type (str): 'sma' for simple MA, 'ema' for exponential MA.

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Short MA, long MA, and crossover regime
                (1 for short MA >= long MA, -1 for short MA < long MA, 0 if equal or NaN).
        """
        cache_key = f"{ma_type}_crossover_{close_col}_{short_window}_{long_window}"
        cache_key_short = f"{ma_type}_{close_col}_{short_window}"
        cache_key_long = f"{ma_type}_{close_col}_{long_window}"

        close = self.ohlc_stock[close_col]

        if cache_key_short in self._cache:
            short_ma = self._cache[cache_key_short]
        else:
            if ma_type == 'sma':
                short_ma = close.rolling(short_window, min_periods=short_window).mean()
            else:  # ema
                short_ma = close.ewm(span=short_window, min_periods=short_window, adjust=False).mean()
            self._cache[cache_key_short] = short_ma

        if cache_key_long in self._cache:
            long_ma = self._cache[cache_key_long]
        else:
            if ma_type == 'sma':
                long_ma = close.rolling(long_window, min_periods=long_window).mean()
            else:  # ema
                long_ma = close.ewm(span=long_window, min_periods=long_window, adjust=False).mean()
            self._cache[cache_key_long] = long_ma

        if cache_key in self._cache:
            regime = self._cache[cache_key]
        else:
            regime = np.sign(short_ma - long_ma)
            regime_series = pd.Series(regime, index=self.ohlc_stock.index)
            self._cache[cache_key] = regime_series
            regime = regime_series

        return short_ma, long_ma, regime

    def compute_ma_regime(self, ma_type: str, short_window: int, medium_window: int, long_window: int, 
                          relative: bool = False, inplace: bool = False) -> pd.DataFrame:
        """
        Computes triple MA crossover regime and includes MA values.

        Args:
            ma_type (str): 'sma' or 'ema'.
            short_window (int): Short MA window.
            medium_window (int): Medium MA window.
            long_window (int): Long MA window.
            relative (bool): If True, use relative close; else absolute.
            inplace (bool): If True, modify input DataFrame; else return a new one.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - SMA: 'sma_short_<short>', 'sma_medium_<medium>', 'sma_long_<long>',
                       'sma_<short><medium>', 'sma_<medium><long>', 'sma_<short><medium><long>' (or 'rsma_').
                - EMA: 'ema_short_<short>', 'ema_medium_<medium>', 'ema_long_<long>',
                       'ema_<short><medium>', 'ema_<medium><long>', 'ema_<short><medium><long>' (or 'rema_').

        Raises:
            ValueError: If ma_type or windows are invalid, or DataFrame is too short.
            KeyError: If required OHLC columns are missing.
        """
        if ma_type not in ['sma', 'ema']:
            raise ValueError("ma_type must be 'sma' or 'ema'")
        if short_window < 1 or medium_window < 1 or long_window < 1:
            raise ValueError("Windows must be positive.")
        if not (short_window < medium_window < long_window):
            raise ValueError("Windows must satisfy short < medium < long.")
        if len(self.ohlc_stock) < long_window:
            raise ValueError(f"DataFrame length ({len(self.ohlc_stock)}) must be >= long_window ({long_window}).")
        
        _o, _h, _l, _c = self._lower_upper_OHLC(relative=relative)
        df = self.ohlc_stock if inplace else self.ohlc_stock.copy()
        
        prefix = 'rsma_' if relative and ma_type == 'sma' else 'rema_' if relative and ma_type == 'ema' else \
                 'sma_' if ma_type == 'sma' else 'ema_'

        # Short vs Medium crossover
        short_ma, medium_ma, sm_crossover = self._compute_ma_crossover(_c, short_window, medium_window, ma_type)
        df[f'{prefix}short_{short_window}'] = short_ma
        df[f'{prefix}medium_{medium_window}'] = medium_ma
        df[f'{prefix}{short_window}{medium_window}'] = sm_crossover

        # Medium vs Long crossover
        medium_ma_reused, long_ma, ml_crossover = self._compute_ma_crossover(_c, medium_window, long_window, ma_type)
        df[f'{prefix}long_{long_window}'] = long_ma
        df[f'{prefix}{medium_window}{long_window}'] = ml_crossover

        # Final triple crossover signal (product of pairwise signals)
        final_regime = sm_crossover * ml_crossover
        df[f'{prefix}{short_window}{medium_window}{long_window}'] = final_regime
        
        return df

    def signal_ma(self, ma_type: str = 'sma', short_window: int = 5, medium_window: int = 10, 
                  long_window: int = 20, relative: bool = False) -> pd.DataFrame:
        """
        Computes MA crossover regime with MA values, absolute or relative.

        Args:
            ma_type (str): 'sma' or 'ema'. Defaults to 'sma'.
            short_window (int): Short MA window. Defaults to 5.
            medium_window (int): Medium MA window. Defaults to 10.
            long_window (int): Long MA window. Defaults to 20.
            relative (bool): If True, use relative close ('rclose'); else absolute ('close').

        Returns:
            pd.DataFrame: DataFrame with columns:
                - SMA: 'sma_short_<short>', 'sma_medium_<medium>', 'sma_long_<long>',
                       'sma_<short><medium>', 'sma_<medium><long>', 'sma_<short><medium><long>' (or 'rsma_').
                - EMA: 'ema_short_<short>', 'ema_medium_<medium>', 'ema_long_<long>',
                       'ema_<short><medium>', 'ema_<medium><long>', 'ema_<short><medium><long>' (or 'rema_').

        Raises:
            ValueError: If ma_type or windows are invalid, or DataFrame is too short.
            KeyError: If required OHLC columns are missing.
        """
        return self.compute_ma_regime(ma_type=ma_type, short_window=short_window, medium_window=medium_window, 
                                      long_window=long_window, relative=relative, inplace=False)