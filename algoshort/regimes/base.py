"""
Base class and common utilities for regime detection modules.

This module provides:
- BaseRegimeDetector: Abstract base class with shared functionality
- Common column detection and validation utilities
- Caching mechanisms for computed values
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class BaseRegimeDetector(ABC):
    """
    Abstract base class for all regime detection implementations.

    Provides common functionality:
    - OHLC column detection (absolute and relative)
    - DataFrame validation
    - Caching mechanism for computed values
    - Logging setup

    Attributes:
        df (pd.DataFrame): OHLC DataFrame
        _cache (Dict[str, pd.Series]): Cache for computed values
        logger (logging.Logger): Logger instance
    """

    # Minimum required rows for regime detection
    MIN_ROWS = 2

    def __init__(self, df: pd.DataFrame, log_level: int = logging.INFO):
        """
        Initialize the base regime detector.

        Args:
            df: DataFrame with OHLC columns
            log_level: Logging level (default: logging.INFO)

        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If DataFrame is empty or missing OHLC columns
        """
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        # Only add handler if none exist to prevent duplicates
        if not self.logger.handlers and not self.logger.parent.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Validate DataFrame
        self._validate_dataframe(df)

        # Store a copy to prevent external modifications
        self._df = df.copy()
        self._cache: Dict[str, pd.Series] = {}

        self.logger.debug(
            f"Initialized {self.__class__.__name__} with DataFrame of shape {self._df.shape}"
        )

    @property
    def df(self) -> pd.DataFrame:
        """Get the underlying DataFrame."""
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        """Set the DataFrame and clear the cache."""
        self._validate_dataframe(value)
        self._df = value.copy()
        self._cache.clear()
        self.logger.debug(f"DataFrame updated: {len(self._df)} rows, cache cleared")

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate that the input is a proper DataFrame with OHLC columns.

        Args:
            df: DataFrame to validate

        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If DataFrame is empty or missing required columns
        """
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input must be a pandas DataFrame")
            raise TypeError("Input must be a pandas DataFrame")

        if df.empty:
            self.logger.error("DataFrame is empty")
            raise ValueError("DataFrame cannot be empty")

        if len(df) < self.MIN_ROWS:
            self.logger.error(f"DataFrame must have at least {self.MIN_ROWS} rows")
            raise ValueError(f"DataFrame must have at least {self.MIN_ROWS} rows")

        # Check for OHLC columns (absolute or relative)
        has_absolute = self._has_ohlc_columns(df, relative=False)
        has_relative = self._has_ohlc_columns(df, relative=True)

        if not has_absolute and not has_relative:
            self.logger.error("DataFrame must contain OHLC columns")
            raise ValueError(
                "Missing required OHLC columns. Expected: "
                "['open', 'high', 'low', 'close'] or ['Open', 'High', 'Low', 'Close']"
            )

    def _has_ohlc_columns(self, df: pd.DataFrame, relative: bool = False) -> bool:
        """
        Check if DataFrame has required OHLC columns.

        Args:
            df: DataFrame to check
            relative: If True, check for relative columns (r-prefix)

        Returns:
            True if all required columns exist
        """
        try:
            self._get_ohlc_columns(df, relative=relative)
            return True
        except KeyError:
            return False

    def _get_ohlc_columns(
        self,
        df: Optional[pd.DataFrame] = None,
        relative: bool = False
    ) -> Tuple[str, str, str, str]:
        """
        Determine OHLC column names based on DataFrame columns.

        Args:
            df: DataFrame to check (uses self._df if None)
            relative: If True, use relative columns (r-prefix)

        Returns:
            Tuple of (open, high, low, close) column names

        Raises:
            KeyError: If required columns are not found
        """
        if df is None:
            df = self._df

        prefix = 'r' if relative else ''

        # Check for capitalized columns first
        if 'Open' in df.columns:
            ohlc = [f'{prefix}Open', f'{prefix}High', f'{prefix}Low', f'{prefix}Close']
        elif 'open' in df.columns:
            ohlc = [f'{prefix}open', f'{prefix}high', f'{prefix}low', f'{prefix}close']
        else:
            raise KeyError("No 'Open' or 'open' column found in DataFrame")

        # Validate all columns exist
        missing = [col for col in ohlc if col not in df.columns]
        if missing:
            raise KeyError(f"Missing OHLC columns: {missing}")

        return tuple(ohlc)

    def _validate_window(self, window: int, name: str = "window") -> None:
        """
        Validate that window parameter is valid.

        Args:
            window: Window size to validate
            name: Parameter name for error messages

        Raises:
            ValueError: If window is invalid
        """
        if not isinstance(window, int) or window < 1:
            self.logger.error(f"{name} must be a positive integer, got {window}")
            raise ValueError(f"{name} must be a positive integer, got {window}")

        if window > len(self._df):
            self.logger.error(
                f"{name} ({window}) exceeds DataFrame length ({len(self._df)})"
            )
            raise ValueError(
                f"{name} ({window}) cannot exceed DataFrame length ({len(self._df)})"
            )

    def _get_cached(self, key: str) -> Optional[pd.Series]:
        """
        Get a cached value if it exists.

        Args:
            key: Cache key

        Returns:
            Cached Series or None if not found
        """
        if key in self._cache:
            self.logger.debug(f"Cache hit: {key}")
            return self._cache[key]
        return None

    def _set_cached(self, key: str, value: pd.Series) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Series to cache
        """
        self._cache[key] = value
        self.logger.debug(f"Cached: {key}")

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
        self.logger.debug("Cache cleared")

    def get_column_prefix(self, relative: bool = False) -> Dict[str, str]:
        """
        Get column name prefixes based on relative flag.

        Args:
            relative: If True, return relative prefixes

        Returns:
            Dictionary with prefix keys
        """
        if relative:
            return {
                'high': 'rhi_',
                'low': 'rlo_',
                'breakout': 'rbo_',
                'turtle': 'rtt_',
                'sma': 'rsma_',
                'ema': 'rema_',
                'regime': 'rrg',
                'floor': 'rflr',
                'ceiling': 'rclg',
                'regime_change': 'rrg_ch'
            }
        return {
            'high': 'hi_',
            'low': 'lo_',
            'breakout': 'bo_',
            'turtle': 'tt_',
            'sma': 'sma_',
            'ema': 'ema_',
            'regime': 'rg',
            'floor': 'flr',
            'ceiling': 'clg',
            'regime_change': 'rg_ch'
        }

    @abstractmethod
    def compute(self, **kwargs) -> pd.DataFrame:
        """
        Compute regime signals. Must be implemented by subclasses.

        Returns:
            DataFrame with regime columns added
        """
        pass


def validate_window_order(*windows: int, names: Optional[List[str]] = None) -> None:
    """
    Validate that windows are in strictly increasing order.

    Args:
        *windows: Window values to validate
        names: Optional names for error messages

    Raises:
        ValueError: If windows are not in increasing order
    """
    if names is None:
        names = [f"window_{i}" for i in range(len(windows))]

    for i in range(len(windows) - 1):
        if windows[i] >= windows[i + 1]:
            raise ValueError(
                f"{names[i]} ({windows[i]}) must be less than "
                f"{names[i + 1]} ({windows[i + 1]})"
            )


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: ATR window (default: 14)

    Returns:
        ATR series
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=1).mean()

    return atr
