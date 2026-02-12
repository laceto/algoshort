"""
Breakout and Turtle Trader Regime Detection.

This module provides regime detection based on price breakouts:
- Breakout: Single-window breakout detection
- Turtle: Dual-window Turtle Trader strategy

Signals:
    - 1: Bullish (breakout above rolling high)
    - -1: Bearish (breakdown below rolling low)
    - 0: Neutral (for Turtle only)

Example:
    >>> from algoshort.regimes.breakout import BreakoutRegime
    >>> detector = BreakoutRegime(df)
    >>> result = detector.breakout(window=20)
    >>> result = detector.turtle(slow=50, fast=20)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from algoshort.regimes.base import BaseRegimeDetector


logger = logging.getLogger(__name__)


class BreakoutRegime(BaseRegimeDetector):
    """
    Regime detection based on price breakouts and Turtle Trader strategy.

    Breakout Logic:
        - Signal = 1 when high == rolling max high
        - Signal = -1 when low == rolling min low
        - Forward-filled to maintain position

    Turtle Trader Logic:
        - Long = 1 when both slow and fast breakouts are bullish
        - Short = -1 when both slow and fast breakouts are bearish
        - Neutral = 0 otherwise

    Attributes:
        df (pd.DataFrame): OHLC DataFrame
        _cache (Dict[str, pd.Series]): Cache for rolling calculations
    """

    def __init__(self, df: pd.DataFrame, log_level: int = logging.INFO):
        """
        Initialize the Breakout Regime detector.

        Args:
            df: DataFrame with OHLC columns
            log_level: Logging level

        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If DataFrame is empty or missing OHLC columns
        """
        super().__init__(df, log_level)
        self.logger.info(f"BreakoutRegime initialized with {len(self._df)} rows")

    def _get_rolling_stats(
        self,
        high_col: str,
        low_col: str,
        window: int
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute rolling max high and min low with caching.

        Args:
            high_col: High price column name
            low_col: Low price column name
            window: Lookback window

        Returns:
            Tuple of (rolling_max_high, rolling_min_low)
        """
        # Check cache for rolling max
        cache_key_high = f"rollmax_{high_col}_{window}"
        rolling_max = self._get_cached(cache_key_high)

        if rolling_max is None:
            rolling_max = self._df[high_col].rolling(
                window, min_periods=window
            ).max()
            self._set_cached(cache_key_high, rolling_max)

        # Check cache for rolling min
        cache_key_low = f"rollmin_{low_col}_{window}"
        rolling_min = self._get_cached(cache_key_low)

        if rolling_min is None:
            rolling_min = self._df[low_col].rolling(
                window, min_periods=window
            ).min()
            self._set_cached(cache_key_low, rolling_min)

        return rolling_max, rolling_min

    def _compute_breakout(
        self,
        high_col: str,
        low_col: str,
        window: int
    ) -> pd.Series:
        """
        Compute breakout regime signal.

        Args:
            high_col: High price column name
            low_col: Low price column name
            window: Lookback window

        Returns:
            Regime signal series (1, -1, or forward-filled)
        """
        cache_key = f"breakout_{high_col}_{low_col}_{window}"
        cached = self._get_cached(cache_key)

        if cached is not None:
            return cached

        high = self._df[high_col]
        low = self._df[low_col]
        rolling_max, rolling_min = self._get_rolling_stats(high_col, low_col, window)

        # Signal: 1 at new high, -1 at new low, NaN otherwise
        signal = np.where(
            high == rolling_max, 1,
            np.where(low == rolling_min, -1, np.nan)
        )

        # Forward-fill to maintain position
        regime = pd.Series(signal, index=self._df.index).ffill()

        self._set_cached(cache_key, regime)
        return regime

    def compute(
        self,
        regime_type: str = 'breakout',
        window: int = 20,
        fast_window: Optional[int] = None,
        relative: bool = False,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Compute breakout or turtle regime.

        Args:
            regime_type: 'breakout' or 'turtle'
            window: Window for breakout, or slow window for turtle
            fast_window: Fast window for turtle (required for turtle)
            relative: If True, use relative OHLC columns
            inplace: If True, modify internal DataFrame; else return copy

        Returns:
            DataFrame with regime columns:
                Breakout:
                    - hi_{window}: Rolling max high
                    - lo_{window}: Rolling min low
                    - bo_{window}: Breakout signal

                Turtle:
                    - hi_{slow}, lo_{slow}: Slow rolling stats
                    - hi_{fast}, lo_{fast}: Fast rolling stats
                    - bo_{slow}, bo_{fast}: Individual breakout signals
                    - tt_{slow}{fast}: Combined turtle signal

        Raises:
            ValueError: If regime_type or windows are invalid
            KeyError: If required OHLC columns are missing
        """
        self.logger.info(f"Computing {regime_type} regime with window={window}")

        # Validate parameters
        if regime_type not in ['breakout', 'turtle']:
            raise ValueError("regime_type must be 'breakout' or 'turtle'")

        self._validate_window(window, "window")

        if regime_type == 'turtle':
            if fast_window is None:
                raise ValueError("fast_window is required for turtle regime")
            self._validate_window(fast_window, "fast_window")
            if fast_window >= window:
                raise ValueError(
                    f"fast_window ({fast_window}) must be less than window ({window})"
                )

        # Get OHLC columns
        _o, _h, _l, _c = self._get_ohlc_columns(relative=relative)

        # Prepare output DataFrame
        df = self._df if inplace else self._df.copy()

        # Get column prefixes
        prefixes = self.get_column_prefix(relative)
        prefix_h = prefixes['high']
        prefix_l = prefixes['low']
        prefix_bo = prefixes['breakout']
        prefix_tt = prefixes['turtle']

        if regime_type == 'breakout':
            regime = self._compute_breakout(_h, _l, window)
            rolling_max, rolling_min = self._get_rolling_stats(_h, _l, window)

            df[f'{prefix_h}{window}'] = rolling_max
            df[f'{prefix_l}{window}'] = rolling_min
            df[f'{prefix_bo}{window}'] = regime

            self.logger.info(
                f"Breakout computed: "
                f"{(regime == 1).sum()} bullish, {(regime == -1).sum()} bearish"
            )

        else:  # turtle
            slow_window = window

            # Compute both breakout signals
            slow_regime = self._compute_breakout(_h, _l, slow_window)
            fast_regime = self._compute_breakout(_h, _l, fast_window)

            # Get rolling stats
            slow_max, slow_min = self._get_rolling_stats(_h, _l, slow_window)
            fast_max, fast_min = self._get_rolling_stats(_h, _l, fast_window)

            # Add columns
            df[f'{prefix_h}{slow_window}'] = slow_max
            df[f'{prefix_l}{slow_window}'] = slow_min
            df[f'{prefix_h}{fast_window}'] = fast_max
            df[f'{prefix_l}{fast_window}'] = fast_min
            df[f'{prefix_bo}{slow_window}'] = slow_regime
            df[f'{prefix_bo}{fast_window}'] = fast_regime

            # Compute turtle signal
            # Long: both bullish, Short: both bearish, else neutral
            turtle = np.where(
                slow_regime == 1,
                np.where(fast_regime == 1, 1, 0),
                np.where(
                    slow_regime == -1,
                    np.where(fast_regime == -1, -1, 0),
                    0
                )
            )
            df[f'{prefix_tt}{slow_window}{fast_window}'] = pd.Series(
                turtle, index=df.index
            )

            self.logger.info(
                f"Turtle computed: "
                f"{(turtle == 1).sum()} long, "
                f"{(turtle == -1).sum()} short, "
                f"{(turtle == 0).sum()} neutral"
            )

        return df

    def breakout(
        self,
        window: int = 20,
        relative: bool = False
    ) -> pd.DataFrame:
        """
        Compute single-window breakout regime.

        Args:
            window: Lookback window (default: 20)
            relative: If True, use relative OHLC columns

        Returns:
            DataFrame with breakout columns

        Example:
            >>> detector = BreakoutRegime(df)
            >>> result = detector.breakout(window=50)
            >>> signal = result['bo_50']  # Breakout signal
        """
        return self.compute(
            regime_type='breakout',
            window=window,
            relative=relative,
            inplace=False
        )

    def turtle(
        self,
        slow: int = 50,
        fast: int = 20,
        relative: bool = False
    ) -> pd.DataFrame:
        """
        Compute Turtle Trader dual-window regime.

        Args:
            slow: Slow window (default: 50)
            fast: Fast window (default: 20)
            relative: If True, use relative OHLC columns

        Returns:
            DataFrame with turtle trader columns

        Example:
            >>> detector = BreakoutRegime(df)
            >>> result = detector.turtle(slow=55, fast=20)
            >>> signal = result['tt_5520']  # Turtle signal
        """
        return self.compute(
            regime_type='turtle',
            window=slow,
            fast_window=fast,
            relative=relative,
            inplace=False
        )

    def get_signal_column(
        self,
        regime_type: str = 'breakout',
        window: int = 20,
        fast_window: Optional[int] = None,
        relative: bool = False
    ) -> str:
        """
        Get the signal column name for given parameters.

        Args:
            regime_type: 'breakout' or 'turtle'
            window: Window or slow window
            fast_window: Fast window (for turtle)
            relative: If True, return relative column name

        Returns:
            Signal column name (e.g., 'bo_20' or 'tt_5020')
        """
        prefixes = self.get_column_prefix(relative)

        if regime_type == 'breakout':
            return f"{prefixes['breakout']}{window}"
        else:  # turtle
            return f"{prefixes['turtle']}{window}{fast_window}"
