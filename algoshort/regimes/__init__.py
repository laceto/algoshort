"""
Unified Regime Detection Module.

This module provides a unified interface for all regime detection methods:
- Moving Average Crossover (SMA, EMA)
- Breakout and Turtle Trader
- Floor/Ceiling Swing Analysis

Example:
    >>> from algoshort.regimes import RegimeDetector
    >>> detector = RegimeDetector(df)
    >>>
    >>> # Moving Average regimes
    >>> df_ma = detector.sma_crossover(short=5, medium=10, long=20)
    >>> df_ema = detector.ema_crossover(short=12, medium=26, long=50)
    >>>
    >>> # Breakout regimes
    >>> df_bo = detector.breakout(window=20)
    >>> df_tt = detector.turtle(slow=50, fast=20)
    >>>
    >>> # Floor/Ceiling regime
    >>> df_fc = detector.floor_ceiling(lvl=1, threshold=1.5)
    >>>
    >>> # Or use the generic interface
    >>> df = detector.compute('sma_crossover', short=10, medium=20, long=50)

Available Methods:
    - sma_crossover: Triple SMA crossover
    - ema_crossover: Triple EMA crossover
    - breakout: Single-window breakout
    - turtle: Dual-window Turtle Trader
    - floor_ceiling: Floor/Ceiling swing analysis
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from algoshort.regimes.base import BaseRegimeDetector, calculate_atr, validate_window_order
from algoshort.regimes.breakout import BreakoutRegime
from algoshort.regimes.floor_ceiling import FloorCeilingRegime
from algoshort.regimes.ma import MovingAverageCrossover


logger = logging.getLogger(__name__)


__all__ = [
    # Main unified interface
    'RegimeDetector',
    # Individual detector classes
    'MovingAverageCrossover',
    'BreakoutRegime',
    'FloorCeilingRegime',
    # Base class and utilities
    'BaseRegimeDetector',
    'calculate_atr',
    'validate_window_order',
]


class RegimeDetector:
    """
    Unified interface for all regime detection methods.

    This class provides a single entry point for computing regime signals
    using various methods: moving average crossovers, breakouts, turtle trader,
    and floor/ceiling swing analysis.

    Attributes:
        df (pd.DataFrame): OHLC DataFrame
        _ma (MovingAverageCrossover): MA detector (lazy-loaded)
        _bo (BreakoutRegime): Breakout detector (lazy-loaded)
        _fc (FloorCeilingRegime): Floor/Ceiling detector (lazy-loaded)

    Example:
        >>> detector = RegimeDetector(df)
        >>> result = detector.sma_crossover(short=10, medium=20, long=50)
        >>> signal = result['sma_102050']
    """

    # Available regime methods
    METHODS = {
        'sma_crossover': 'Moving Average - SMA triple crossover',
        'ema_crossover': 'Moving Average - EMA triple crossover',
        'breakout': 'Breakout - single window',
        'turtle': 'Turtle Trader - dual window',
        'floor_ceiling': 'Floor/Ceiling - swing analysis'
    }

    # Minimum required rows
    MIN_ROWS = 2

    def __init__(self, df: pd.DataFrame, log_level: int = logging.INFO):
        """
        Initialize the unified RegimeDetector.

        Args:
            df: DataFrame with OHLC columns
            log_level: Logging level (default: logging.INFO)

        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If DataFrame is empty or has insufficient rows
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        if len(df) < self.MIN_ROWS:
            raise ValueError(f"DataFrame must have at least {self.MIN_ROWS} rows")

        self._df = df
        self._log_level = log_level

        # Lazy-loaded detectors
        self._ma: Optional[MovingAverageCrossover] = None
        self._bo: Optional[BreakoutRegime] = None
        self._fc: Optional[FloorCeilingRegime] = None

        self.logger.info(f"RegimeDetector initialized with {len(df)} rows")

    @property
    def df(self) -> pd.DataFrame:
        """Get the underlying DataFrame."""
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        """Set DataFrame and reset all detectors."""
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self._df = value
        # Reset lazy-loaded detectors
        self._ma = None
        self._bo = None
        self._fc = None
        self.logger.debug("DataFrame updated, detectors reset")

    @property
    def ma_detector(self) -> MovingAverageCrossover:
        """Get or create Moving Average detector."""
        if self._ma is None:
            self._ma = MovingAverageCrossover(self._df, log_level=self._log_level)
        return self._ma

    @property
    def bo_detector(self) -> BreakoutRegime:
        """Get or create Breakout detector."""
        if self._bo is None:
            self._bo = BreakoutRegime(self._df, log_level=self._log_level)
        return self._bo

    @property
    def fc_detector(self) -> FloorCeilingRegime:
        """Get or create Floor/Ceiling detector."""
        if self._fc is None:
            self._fc = FloorCeilingRegime(self._df, log_level=self._log_level)
        return self._fc

    def sma_crossover(
        self,
        short: int = 5,
        medium: int = 10,
        long: int = 20,
        relative: bool = False
    ) -> pd.DataFrame:
        """
        Compute Simple Moving Average triple crossover regime.

        Args:
            short: Short SMA window (default: 5)
            medium: Medium SMA window (default: 10)
            long: Long SMA window (default: 20)
            relative: If True, use relative OHLC columns

        Returns:
            DataFrame with SMA crossover columns:
                - sma_short_{short}: Short SMA values
                - sma_medium_{medium}: Medium SMA values
                - sma_long_{long}: Long SMA values
                - sma_{short}{medium}{long}: Triple crossover signal

        Example:
            >>> result = detector.sma_crossover(short=10, medium=20, long=50)
            >>> signal = result['sma_102050']
        """
        self.logger.info(f"Computing SMA crossover: {short}/{medium}/{long}")
        return self.ma_detector.sma_crossover(
            short=short, medium=medium, long=long, relative=relative
        )

    def ema_crossover(
        self,
        short: int = 5,
        medium: int = 10,
        long: int = 20,
        relative: bool = False
    ) -> pd.DataFrame:
        """
        Compute Exponential Moving Average triple crossover regime.

        Args:
            short: Short EMA window (default: 5)
            medium: Medium EMA window (default: 10)
            long: Long EMA window (default: 20)
            relative: If True, use relative OHLC columns

        Returns:
            DataFrame with EMA crossover columns:
                - ema_short_{short}: Short EMA values
                - ema_medium_{medium}: Medium EMA values
                - ema_long_{long}: Long EMA values
                - ema_{short}{medium}{long}: Triple crossover signal

        Example:
            >>> result = detector.ema_crossover(short=12, medium=26, long=50)
            >>> signal = result['ema_122650']
        """
        self.logger.info(f"Computing EMA crossover: {short}/{medium}/{long}")
        return self.ma_detector.ema_crossover(
            short=short, medium=medium, long=long, relative=relative
        )

    def breakout(
        self,
        window: int = 20,
        relative: bool = False
    ) -> pd.DataFrame:
        """
        Compute single-window breakout regime.

        Signal Logic:
            - 1: High equals rolling max high (bullish breakout)
            - -1: Low equals rolling min low (bearish breakdown)
            - Forward-filled to maintain position

        Args:
            window: Lookback window (default: 20)
            relative: If True, use relative OHLC columns

        Returns:
            DataFrame with breakout columns:
                - hi_{window}: Rolling max high
                - lo_{window}: Rolling min low
                - bo_{window}: Breakout signal

        Example:
            >>> result = detector.breakout(window=50)
            >>> signal = result['bo_50']
        """
        self.logger.info(f"Computing breakout: window={window}")
        return self.bo_detector.breakout(window=window, relative=relative)

    def turtle(
        self,
        slow: int = 50,
        fast: int = 20,
        relative: bool = False
    ) -> pd.DataFrame:
        """
        Compute Turtle Trader dual-window regime.

        Signal Logic:
            - 1 (Long): Both slow and fast breakouts are bullish
            - -1 (Short): Both slow and fast breakouts are bearish
            - 0 (Neutral): Mixed signals

        Args:
            slow: Slow window (default: 50)
            fast: Fast window (default: 20)
            relative: If True, use relative OHLC columns

        Returns:
            DataFrame with turtle columns:
                - hi_{slow}, lo_{slow}: Slow rolling stats
                - hi_{fast}, lo_{fast}: Fast rolling stats
                - bo_{slow}, bo_{fast}: Individual breakout signals
                - tt_{slow}{fast}: Combined turtle signal

        Example:
            >>> result = detector.turtle(slow=55, fast=20)
            >>> signal = result['tt_5520']
        """
        self.logger.info(f"Computing turtle: slow={slow}, fast={fast}")
        return self.bo_detector.turtle(slow=slow, fast=fast, relative=relative)

    def floor_ceiling(
        self,
        lvl: int = 1,
        threshold: float = 1.5,
        vlty_n: int = 63,
        relative: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute Floor/Ceiling swing-based regime.

        This method performs sophisticated swing analysis to identify:
        - Floor and ceiling price levels
        - Regime changes based on breakout/breakdown
        - Multiple swing levels

        Args:
            lvl: Swing level (default: 1)
            threshold: Floor/ceiling discovery threshold (default: 1.5)
            vlty_n: Volatility window for ATR (default: 63)
            relative: If True, use relative OHLC columns
            **kwargs: Additional parameters (d_vol, dist_pct, retrace_pct, r_vol)

        Returns:
            DataFrame with floor/ceiling columns:
                - hi{n}, lo{n}: Swing high/low levels
                - rg: Regime signal (1, -1, 0)
                - flr: Floor levels
                - clg: Ceiling levels
                - rg_ch: Regime change levels

        Example:
            >>> result = detector.floor_ceiling(lvl=1, threshold=2.0)
            >>> signal = result['rg']
        """
        self.logger.info(f"Computing floor_ceiling: lvl={lvl}, threshold={threshold}")
        return self.fc_detector.floor_ceiling(
            lvl=lvl, threshold=threshold, vlty_n=vlty_n, relative=relative, **kwargs
        )

    def compute(
        self,
        method: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generic regime computation by method name.

        Args:
            method: Method name (see METHODS class attribute)
            **kwargs: Parameters for the specific method

        Returns:
            DataFrame with regime columns

        Raises:
            ValueError: If method is not recognized

        Example:
            >>> result = detector.compute('sma_crossover', short=10, medium=20, long=50)
            >>> result = detector.compute('turtle', slow=55, fast=20)
        """
        method = method.lower().replace('-', '_')

        method_map = {
            'sma_crossover': self.sma_crossover,
            'sma': self.sma_crossover,
            'ema_crossover': self.ema_crossover,
            'ema': self.ema_crossover,
            'breakout': self.breakout,
            'bo': self.breakout,
            'turtle': self.turtle,
            'tt': self.turtle,
            'floor_ceiling': self.floor_ceiling,
            'fc': self.floor_ceiling,
        }

        if method not in method_map:
            available = ', '.join(sorted(set(method_map.keys())))
            raise ValueError(
                f"Unknown method: '{method}'. Available: {available}"
            )

        self.logger.info(f"Computing regime using method: {method}")
        return method_map[method](**kwargs)

    def compute_all(
        self,
        sma_params: Optional[Dict[str, Any]] = None,
        ema_params: Optional[Dict[str, Any]] = None,
        breakout_params: Optional[Dict[str, Any]] = None,
        turtle_params: Optional[Dict[str, Any]] = None,
        floor_ceiling_params: Optional[Dict[str, Any]] = None,
        relative: bool = False
    ) -> pd.DataFrame:
        """
        Compute all regime signals and merge into a single DataFrame.

        Args:
            sma_params: Parameters for SMA crossover (default: {short=5, medium=10, long=20})
            ema_params: Parameters for EMA crossover (default: {short=5, medium=10, long=20})
            breakout_params: Parameters for breakout (default: {window=20})
            turtle_params: Parameters for turtle (default: {slow=50, fast=20})
            floor_ceiling_params: Parameters for floor_ceiling (default: {lvl=1, threshold=1.5})
            relative: If True, use relative OHLC columns for all methods

        Returns:
            DataFrame with all regime columns merged

        Example:
            >>> result = detector.compute_all(
            ...     sma_params={'short': 10, 'medium': 20, 'long': 50},
            ...     turtle_params={'slow': 55, 'fast': 20}
            ... )
        """
        self.logger.info("Computing all regime signals")

        # Default parameters
        sma_defaults = {'short': 5, 'medium': 10, 'long': 20}
        ema_defaults = {'short': 5, 'medium': 10, 'long': 20}
        bo_defaults = {'window': 20}
        tt_defaults = {'slow': 50, 'fast': 20}
        fc_defaults = {'lvl': 1, 'threshold': 1.5}

        # Merge with provided params
        sma_params = {**sma_defaults, **(sma_params or {}), 'relative': relative}
        ema_params = {**ema_defaults, **(ema_params or {}), 'relative': relative}
        breakout_params = {**bo_defaults, **(breakout_params or {}), 'relative': relative}
        turtle_params = {**tt_defaults, **(turtle_params or {}), 'relative': relative}
        fc_params = {**fc_defaults, **(floor_ceiling_params or {}), 'relative': relative}

        # Start with original DataFrame
        result = self._df.copy()

        # Compute each regime and merge new columns
        for name, method, params in [
            ('SMA', self.sma_crossover, sma_params),
            ('EMA', self.ema_crossover, ema_params),
            ('Breakout', self.breakout, breakout_params),
            ('Turtle', self.turtle, turtle_params),
            ('Floor/Ceiling', self.floor_ceiling, fc_params),
        ]:
            try:
                df_regime = method(**params)
                # Add only new columns
                new_cols = [c for c in df_regime.columns if c not in result.columns]
                if new_cols:
                    result[new_cols] = df_regime[new_cols]
                self.logger.debug(f"{name}: added {len(new_cols)} columns")
            except Exception as e:
                self.logger.warning(f"Failed to compute {name}: {e}")

        self.logger.info(f"All regimes computed: {len(result.columns)} total columns")
        return result

    def get_signal_columns(
        self,
        sma_params: Optional[Dict[str, int]] = None,
        ema_params: Optional[Dict[str, int]] = None,
        breakout_window: int = 20,
        turtle_params: Optional[Dict[str, int]] = None,
        relative: bool = False
    ) -> Dict[str, str]:
        """
        Get signal column names for all methods.

        Args:
            sma_params: {short, medium, long} for SMA
            ema_params: {short, medium, long} for EMA
            breakout_window: Window for breakout
            turtle_params: {slow, fast} for turtle
            relative: If True, return relative column names

        Returns:
            Dictionary mapping method names to signal column names
        """
        sma = sma_params or {'short': 5, 'medium': 10, 'long': 20}
        ema = ema_params or {'short': 5, 'medium': 10, 'long': 20}
        turtle = turtle_params or {'slow': 50, 'fast': 20}

        prefix = 'r' if relative else ''

        return {
            'sma_crossover': f"{prefix}sma_{sma['short']}{sma['medium']}{sma['long']}",
            'ema_crossover': f"{prefix}ema_{ema['short']}{ema['medium']}{ema['long']}",
            'breakout': f"{prefix}bo_{breakout_window}",
            'turtle': f"{prefix}tt_{turtle['slow']}{turtle['fast']}",
            'floor_ceiling': f"{prefix}rg" if not relative else 'rrg',
        }

    @classmethod
    def available_methods(cls) -> Dict[str, str]:
        """
        Get available regime detection methods.

        Returns:
            Dictionary of method names and descriptions
        """
        return cls.METHODS.copy()

    def __repr__(self) -> str:
        return f"RegimeDetector(rows={len(self._df)}, methods={list(self.METHODS.keys())})"
