import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

class RegimeBO:
    """
    A class to detect stock regimes based on OHLC data, supporting:
    - 'breakout': Single-window breakout (1 for high == rolling max high, -1 for low == rolling min low).
    - 'turtle': Dual-window Turtle Trader (1 for long, -1 for short, 0 for neutral).
    Breakout is a special case of Turtle Trader where slow_window == fast_window.
    Includes rolling max high and min low values in output for both regimes.

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
        self._cache: Dict[str, pd.Series] = {}  # Cache for regimes and rolling calculations

    def _lower_upper_OHLC(self, relative: bool = False) -> Tuple[str, str, str, str]:
        """
        Determines OHLC column names based on DataFrame columns and relative flag.

        Args:
            relative (bool): If True, use relative OHLC columns (e.g., 'r_open'); else absolute (e.g., 'open').

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

    def _compute_breakout(self, high_col: str, low_col: str, window: int) -> pd.Series:
        """
        Computes breakout regime for a given window.

        Args:
            high_col (str): Column name for high price.
            low_col (str): Column name for low price.
            window (int): Lookback window for rolling calculations.

        Returns:
            pd.Series: Regime values (1, -1, or NaN, forward-filled).
        """
        cache_key = f"breakout_{high_col}_{low_col}_{window}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        high = self.ohlc_stock[high_col]
        low = self.ohlc_stock[low_col]
        rolling_max = high.rolling(window, min_periods=window).max()
        rolling_min = low.rolling(window, min_periods=window).min()
        
        hl = np.where(high == rolling_max, 1,
                      np.where(low == rolling_min, -1, np.nan))
        
        regime = pd.Series(hl, index=self.ohlc_stock.index).ffill()
        self._cache[cache_key] = regime
        return regime

    def _get_rolling_stats(self, high_col: str, low_col: str, window: int) -> Tuple[pd.Series, pd.Series]:
        """
        Computes rolling max high and min low, with caching.

        Args:
            high_col (str): Column name for high price.
            low_col (str): Column name for low price.
            window (int): Lookback window for rolling calculations.

        Returns:
            Tuple[pd.Series, pd.Series]: Rolling max high and min low.
        """
        cache_key_high = f"rollmax_{high_col}_{window}"
        cache_key_low = f"rollmin_{low_col}_{window}"
        
        if cache_key_high in self._cache:
            rolling_max = self._cache[cache_key_high]
        else:
            rolling_max = self.ohlc_stock[high_col].rolling(window, min_periods=window).max()
            self._cache[cache_key_high] = rolling_max
        
        if cache_key_low in self._cache:
            rolling_min = self._cache[cache_key_low]
        else:
            rolling_min = self.ohlc_stock[low_col].rolling(window, min_periods=window).min()
            self._cache[cache_key_low] = rolling_min
        
        return rolling_max, rolling_min

    def compute_regime(self, regime_type: str, window: int = 20, fast_window: Optional[int] = None, 
                       relative: bool = False, inplace: bool = False) -> pd.DataFrame:
        """
        Computes regime based on specified type and OHLC data.

        Args:
            regime_type (str): Type of regime ('breakout' or 'turtle').
            window (int): Window for breakout or slow window for turtle regime. Defaults to 20.
            fast_window (int, optional): Fast window for turtle regime. Defaults to None (breakout).
            relative (bool): If True, use relative OHLC columns (e.g., 'r_open'); else absolute.
            inplace (bool): If True, modify input DataFrame; else return a new one.

        Returns:
            pd.DataFrame: DataFrame with regime columns:
                - Breakout: 'hi_<window>', 'lo_<window>', 'bo_<window>' (or 'rhi_', 'rlo_', 'rbo_').
                - Turtle: 'hi_<window>', 'lo_<window>', 'hi_<fast_window>', 'lo_<fast_window>',
                          'bo_<window>', 'bo_<fast_window>', 'tt_<window><fast_window>' (or 'rhi_', 'rlo_', 'rbo_', 'rtt_').

        Raises:
            ValueError: If regime_type is invalid, windows are invalid, or DataFrame is too short.
            KeyError: If required OHLC columns are missing.
        """
        if regime_type not in ['breakout', 'turtle']:
            raise ValueError("regime_type must be 'breakout' or 'turtle'")
        if window < 1:
            raise ValueError("window must be positive.")
        if fast_window is not None and (fast_window < 1 or fast_window >= window):
            raise ValueError("fast_window must be positive and less than window.")
        if regime_type == 'turtle' and fast_window is None:
            raise ValueError("fast_window is required for turtle regime.")
        if len(self.ohlc_stock) < window:
            raise ValueError(f"DataFrame length ({len(self.ohlc_stock)}) must be >= window ({window}).")
        
        _o, _h, _l, _c = self._lower_upper_OHLC(relative=relative)
        df = self.ohlc_stock if inplace else self.ohlc_stock.copy()
        
        prefix_bo = 'rbo_' if relative else 'bo_'
        prefix_h = 'rhi_' if relative else 'hi_'
        prefix_l = 'rlo_' if relative else 'lo_'
        prefix_tt = 'rtt_' if relative else 'tt_'

        if regime_type == 'breakout':
            regime = self._compute_breakout(_h, _l, window)
            rolling_max, rolling_min = self._get_rolling_stats(_h, _l, window)
            df[f'{prefix_h}{window}'] = rolling_max
            df[f'{prefix_l}{window}'] = rolling_min
            df[f'{prefix_bo}{window}'] = regime
        else:  # turtle
            slow_regime = self._compute_breakout(_h, _l, window)
            fast_regime = self._compute_breakout(_h, _l, fast_window)
            slow_max, slow_min = self._get_rolling_stats(_h, _l, window)
            fast_max, fast_min = self._get_rolling_stats(_h, _l, fast_window)
            df[f'{prefix_h}{window}'] = slow_max
            df[f'{prefix_l}{window}'] = slow_min
            df[f'{prefix_h}{fast_window}'] = fast_max
            df[f'{prefix_l}{fast_window}'] = fast_min
            df[f'{prefix_bo}{window}'] = slow_regime
            df[f'{prefix_bo}{fast_window}'] = fast_regime
            turtle = np.where(slow_regime == 1, np.where(fast_regime == 1, 1, 0),
                              np.where(slow_regime == -1, np.where(fast_regime == -1, -1, 0), 0))
            df[f'{prefix_tt}{window}{fast_window}'] = pd.Series(turtle, index=df.index)
        
        return df

    def signal_bo(self, window: int = 20, relative: bool = False) -> pd.DataFrame:
        """
        Computes breakout regime, absolute or relative.

        Args:
            window (int): Window for breakout calculation. Defaults to 20.
            relative (bool): If True, use relative OHLC columns ('r_open', etc.); else absolute ('open', etc.).

        Returns:
            pd.DataFrame: DataFrame with 'hi_<window>', 'lo_<window>', 'bo_<window>' (or 'rhi_', 'rlo_', 'rbo_').

        Raises:
            ValueError: If window is invalid or DataFrame is too short.
            KeyError: If required OHLC columns are missing.
        """
        return self.compute_regime(regime_type='breakout', window=window, relative=relative, inplace=False)

    def signal_tt(self, slow_window: int = 20, fast_window: int = 10, relative: bool = False) -> pd.DataFrame:
        """
        Computes Turtle Trader regime, absolute or relative.

        Args:
            slow_window (int): Slow window for turtle regime. Defaults to 20.
            fast_window (int): Fast window for turtle regime. Defaults to 10.
            relative (bool): If True, use relative OHLC columns ('r_open', etc.); else absolute.

        Returns:
            pd.DataFrame: DataFrame with 'hi_<slow_window>', 'lo_<slow_window>', 'hi_<fast_window>',
                          'lo_<fast_window>', 'bo_<slow_window>', 'bo_<fast_window>',
                          'tt_<slow_window><fast_window>' (or 'rhi_', 'rlo_', 'rbo_', 'rtt_' for relative).

        Raises:
            ValueError: If windows are invalid or DataFrame is too short.
            KeyError: If required OHLC columns are missing.
        """
        return self.compute_regime(regime_type='turtle', window=slow_window, fast_window=fast_window, relative=relative, inplace=False)
    
