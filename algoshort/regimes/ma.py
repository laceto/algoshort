"""
Moving Average Crossover Regime Detection.

This module provides regime detection based on triple moving average crossovers
using Simple Moving Average (SMA) or Exponential Moving Average (EMA).

Signals:
    - 1: Bullish (short MA >= medium MA >= long MA)
    - -1: Bearish (short MA <= medium MA <= long MA)
    - 0: Mixed/neutral

Example:
    >>> from algoshort.regimes.ma import MovingAverageCrossover
    >>> detector = MovingAverageCrossover(df)
    >>> result = detector.sma_crossover(short=5, medium=10, long=20)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from algoshort.regimes.base import BaseRegimeDetector, validate_window_order


logger = logging.getLogger(__name__)


class MovingAverageCrossover(BaseRegimeDetector):
    """
    Regime detection based on triple moving average crossovers.

    Supports SMA and EMA crossovers with three windows (short, medium, long).
    Computes pairwise crossovers and multiplies signals for final regime.

    Attributes:
        df (pd.DataFrame): OHLC DataFrame
        _cache (Dict[str, pd.Series]): Cache for MAs and crossover signals
    """

    def __init__(self, df: pd.DataFrame, log_level: int = logging.INFO):
        """
        Initialize the Moving Average Crossover detector.

        Args:
            df: DataFrame with OHLC columns
            log_level: Logging level

        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If DataFrame is empty or missing OHLC columns
        """
        super().__init__(df, log_level)
        self.logger.info(f"MovingAverageCrossover initialized with {len(self._df)} rows")

    def _compute_ma(
        self,
        close_col: str,
        window: int,
        ma_type: str = 'sma'
    ) -> pd.Series:
        """
        Compute moving average with caching.

        Args:
            close_col: Close price column name
            window: MA window
            ma_type: 'sma' or 'ema'

        Returns:
            Moving average series
        """
        cache_key = f"{ma_type}_{close_col}_{window}"

        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        close = self._df[close_col]

        if ma_type == 'sma':
            ma = close.rolling(window, min_periods=window).mean()
        else:  # ema
            ma = close.ewm(span=window, min_periods=window, adjust=False).mean()

        self._set_cached(cache_key, ma)
        return ma

    def _compute_crossover(
        self,
        close_col: str,
        short_window: int,
        long_window: int,
        ma_type: str = 'sma'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute MA crossover regime for two windows.

        Args:
            close_col: Close price column name
            short_window: Short MA window
            long_window: Long MA window
            ma_type: 'sma' or 'ema'

        Returns:
            Tuple of (short_ma, long_ma, crossover_signal)
            Signal: 1 for short >= long, -1 for short < long
        """
        short_ma = self._compute_ma(close_col, short_window, ma_type)
        long_ma = self._compute_ma(close_col, long_window, ma_type)

        # Compute crossover signal
        cache_key = f"{ma_type}_crossover_{close_col}_{short_window}_{long_window}"
        cached = self._get_cached(cache_key)

        if cached is not None:
            return short_ma, long_ma, cached

        regime = np.sign(short_ma - long_ma)
        regime_series = pd.Series(regime, index=self._df.index)

        self._set_cached(cache_key, regime_series)
        return short_ma, long_ma, regime_series

    def compute(
        self,
        ma_type: str = 'sma',
        short_window: int = 5,
        medium_window: int = 10,
        long_window: int = 20,
        relative: bool = False,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Compute triple MA crossover regime.

        Args:
            ma_type: 'sma' for Simple MA, 'ema' for Exponential MA
            short_window: Short MA window
            medium_window: Medium MA window
            long_window: Long MA window
            relative: If True, use relative OHLC columns
            inplace: If True, modify internal DataFrame; else return copy

        Returns:
            DataFrame with columns:
                - {prefix}short_{short}: Short MA values
                - {prefix}medium_{medium}: Medium MA values
                - {prefix}long_{long}: Long MA values
                - {prefix}{short}{medium}: Short vs Medium crossover
                - {prefix}{medium}{long}: Medium vs Long crossover
                - {prefix}{short}{medium}{long}: Triple crossover signal

        Raises:
            ValueError: If ma_type or windows are invalid
            KeyError: If required OHLC columns are missing
        """
        self.logger.info(
            f"Computing {ma_type.upper()} crossover: "
            f"{short_window}/{medium_window}/{long_window}"
        )

        # Validate parameters
        if ma_type not in ['sma', 'ema']:
            raise ValueError("ma_type must be 'sma' or 'ema'")

        validate_window_order(
            short_window, medium_window, long_window,
            names=['short_window', 'medium_window', 'long_window']
        )

        self._validate_window(long_window, "long_window")

        # Get OHLC columns
        _o, _h, _l, _c = self._get_ohlc_columns(relative=relative)

        # Prepare output DataFrame
        df = self._df if inplace else self._df.copy()

        # Determine column prefix
        prefix = f'r{ma_type}_' if relative else f'{ma_type}_'

        # Compute short vs medium crossover
        short_ma, medium_ma, sm_crossover = self._compute_crossover(
            _c, short_window, medium_window, ma_type
        )
        df[f'{prefix}short_{short_window}'] = short_ma
        df[f'{prefix}medium_{medium_window}'] = medium_ma
        df[f'{prefix}{short_window}{medium_window}'] = sm_crossover

        # Compute medium vs long crossover
        _, long_ma, ml_crossover = self._compute_crossover(
            _c, medium_window, long_window, ma_type
        )
        df[f'{prefix}long_{long_window}'] = long_ma
        df[f'{prefix}{medium_window}{long_window}'] = ml_crossover

        # Final triple crossover signal (product of pairwise signals)
        final_regime = sm_crossover * ml_crossover
        df[f'{prefix}{short_window}{medium_window}{long_window}'] = final_regime

        self.logger.info(
            f"MA crossover computed: "
            f"{(final_regime == 1).sum()} bullish, "
            f"{(final_regime == -1).sum()} bearish, "
            f"{(final_regime == 0).sum()} neutral"
        )

        return df

    def sma_crossover(
        self,
        short: int = 5,
        medium: int = 10,
        long: int = 20,
        relative: bool = False
    ) -> pd.DataFrame:
        """
        Compute Simple Moving Average crossover regime.

        Args:
            short: Short SMA window (default: 5)
            medium: Medium SMA window (default: 10)
            long: Long SMA window (default: 20)
            relative: If True, use relative OHLC columns

        Returns:
            DataFrame with SMA crossover columns

        Example:
            >>> detector = MovingAverageCrossover(df)
            >>> result = detector.sma_crossover(short=10, medium=20, long=50)
            >>> signal = result['sma_102050']  # Triple crossover signal
        """
        return self.compute(
            ma_type='sma',
            short_window=short,
            medium_window=medium,
            long_window=long,
            relative=relative,
            inplace=False
        )

    def ema_crossover(
        self,
        short: int = 5,
        medium: int = 10,
        long: int = 20,
        relative: bool = False
    ) -> pd.DataFrame:
        """
        Compute Exponential Moving Average crossover regime.

        Args:
            short: Short EMA window (default: 5)
            medium: Medium EMA window (default: 10)
            long: Long EMA window (default: 20)
            relative: If True, use relative OHLC columns

        Returns:
            DataFrame with EMA crossover columns

        Example:
            >>> detector = MovingAverageCrossover(df)
            >>> result = detector.ema_crossover(short=12, medium=26, long=50)
            >>> signal = result['ema_122650']  # Triple crossover signal
        """
        return self.compute(
            ma_type='ema',
            short_window=short,
            medium_window=medium,
            long_window=long,
            relative=relative,
            inplace=False
        )

    def get_signal_column(
        self,
        ma_type: str = 'sma',
        short: int = 5,
        medium: int = 10,
        long: int = 20,
        relative: bool = False
    ) -> str:
        """
        Get the signal column name for given parameters.

        Args:
            ma_type: 'sma' or 'ema'
            short: Short window
            medium: Medium window
            long: Long window
            relative: If True, return relative column name

        Returns:
            Signal column name (e.g., 'sma_51020' or 'rsma_51020')
        """
        prefix = f'r{ma_type}_' if relative else f'{ma_type}_'
        return f'{prefix}{short}{medium}{long}'
