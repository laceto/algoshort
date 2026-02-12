"""
Stop-loss calculation module for trading strategies.

This module provides the StopLossCalculator class for calculating various
stop-loss levels using multiple methods (fixed percentage, ATR, breakout
channel, moving average, volatility-based, and pivot-based).

Classes:
    StopLossCalculator: Main class for stop-loss calculations.

Typical usage:
    calc = StopLossCalculator(ohlc_df)
    result = calc.get_stop_loss('my_signal', 'atr', multiplier=2.0)
"""
import logging
from inspect import signature
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# Module-level logger
logger = logging.getLogger(__name__)

# Minimum stop price to prevent negative stops
MIN_STOP_PRICE = 0.01


class StopLossCalculator:
    """
    A class to calculate stop-loss levels using several methods for both long and short positions.

    Automatically handles cache resets and column detection when data is updated.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC columns (e.g., 'open', 'high', 'low', 'close')
            and signal columns.

    Raises:
        KeyError: If required OHLC columns are not found.
        ValueError: If DataFrame is empty or contains non-numeric data.
    """

    def __init__(self, data: pd.DataFrame):
        self._data: Optional[pd.DataFrame] = None
        self._cache: Dict[str, pd.Series] = {}
        self.price_cols: Dict[str, str] = {}

        # Trigger the setter logic for initial data
        self.data = data
        logger.debug("StopLossCalculator initialized with %d rows", len(data))

    @property
    def data(self) -> pd.DataFrame:
        """Access the current DataFrame segment."""
        return self._data

    @data.setter
    def data(self, new_data: pd.DataFrame) -> None:
        """
        Updates the internal data, clears the cache, and re-detects columns.

        This enables 'self.calc.data = oos_data' to work correctly in loops.
        """
        if new_data is None or new_data.empty:
            logger.error("Attempted to set empty DataFrame")
            raise ValueError("DataFrame cannot be empty.")

        # 1. Update the data (operating on a copy for safety)
        self._data = new_data.copy()

        # 2. CRITICAL: Clear the cache so old metrics don't leak into new data segments
        self._cache = {}

        # 3. Re-detect OHLC column names
        self._detect_ohlc_columns()

        logger.debug("Data updated: %d rows, cache cleared", len(new_data))

    def _validate_signal_column(self, signal: str) -> None:
        """
        Validate that signal column exists and is numeric.

        Args:
            signal: Name of the signal column.

        Raises:
            KeyError: If signal column not found.
            ValueError: If signal column is not numeric.
        """
        if signal not in self.data.columns:
            logger.error(
                "Signal column '%s' not found. Available: %s",
                signal, list(self.data.columns)[:10]
            )
            raise KeyError(
                f"Signal column '{signal}' not found in DataFrame. "
                f"Available columns (first 10): {list(self.data.columns)[:10]}"
            )
        if not np.issubdtype(self.data[signal].dtype, np.number):
            logger.error("Signal column '%s' is not numeric: %s", signal, self.data[signal].dtype)
            raise ValueError(f"Signal column '{signal}' must be numeric")

    def _validate_percentage(self, percentage: float) -> None:
        """
        Validate percentage parameter is in valid range.

        Args:
            percentage: Stop-loss percentage (must be between 0 and 1).

        Raises:
            ValueError: If percentage is not in (0, 1).
        """
        if not 0 < percentage < 1:
            logger.error("Invalid percentage: %s (must be between 0 and 1)", percentage)
            raise ValueError(f"percentage must be between 0 and 1, got {percentage}")

    def _validate_multiplier(self, multiplier: float) -> None:
        """
        Validate multiplier parameter is positive.

        Args:
            multiplier: Stop distance multiplier (must be positive).

        Raises:
            ValueError: If multiplier is not positive.
        """
        if multiplier <= 0:
            logger.error("Invalid multiplier: %s (must be positive)", multiplier)
            raise ValueError(f"multiplier must be positive, got {multiplier}")

    def _validate_window(self, window: Any, param_name: str = "window") -> int:
        """
        Validate and convert window parameter to positive integer.

        Args:
            window: Window value to validate.
            param_name: Name of parameter for error messages.

        Returns:
            Validated window as integer.

        Raises:
            ValueError: If window is not a positive integer.
        """
        try:
            window_int = int(window)
            if window_int < 1:
                raise ValueError(f"{param_name} must be >= 1")
            return window_int
        except (TypeError, ValueError) as e:
            logger.error("Invalid %s value: %r", param_name, window)
            raise ValueError(
                f"Invalid {param_name} value: {window!r} - must be a positive integer"
            ) from e

    def _filter_kwargs(self, method_name: str, **kwargs) -> Dict[str, Any]:
        """
        Return only the kwargs that the target method actually accepts.

        Args:
            method_name: Name of the method to check parameters for.
            **kwargs: Keyword arguments to filter.

        Returns:
            Dictionary of accepted kwargs.

        Raises:
            ValueError: If method_name is not found.
        """
        method = getattr(self, method_name, None)
        if not callable(method):
            raise ValueError(f"Unknown method: {method_name}")

        sig = signature(method)
        accepted = set(sig.parameters.keys())
        # Remove 'self' if present
        accepted.discard('self')

        return {k: v for k, v in kwargs.items() if k in accepted}

    def _detect_ohlc_columns(self) -> None:
        """
        Internal helper to detect column naming conventions.

        Raises:
            KeyError: If required OHLC columns not found.
        """
        absolute_cols = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}
        relative_cols = {'open': 'r_open', 'high': 'r_high', 'low': 'r_low', 'close': 'r_close'}

        if all(col in self._data.columns for col in absolute_cols.values()):
            self.price_cols = absolute_cols
            logger.debug("Detected absolute OHLC columns")
        elif all(col in self._data.columns for col in relative_cols.values()):
            self.price_cols = relative_cols
            logger.debug("Detected relative OHLC columns")
        else:
            logger.error("Required OHLC columns not found. Available: %s", list(self._data.columns))
            raise KeyError("Required OHLC columns not found in the provided DataFrame.")

    def _atr(self, window: Any = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR) with caching.

        Args:
            window: Period for the rolling mean (will be converted to int).

        Returns:
            pd.Series: ATR values, indexed like the input price series.

        Raises:
            ValueError: If window cannot be converted to a positive integer.
        """
        # Create cache key using the original window value (before conversion)
        cache_key = f"ATR_{window}"
        if cache_key in self._cache:
            logger.debug("ATR cache hit: %s", cache_key)
            return self._cache[cache_key]

        window_int = self._validate_window(window, "ATR window")
        logger.debug("ATR cache miss, calculating: window=%d", window_int)

        # Get price series
        high = self._get_price_series('high')
        low = self._get_price_series('low')
        close = self._get_price_series('close')
        close_prev = close.shift(1)

        # Validate data quality: high should be >= low
        if (high < low).any():
            logger.warning(
                "Data quality issue: %d rows where high < low",
                (high < low).sum()
            )

        # True Range components
        tr_high_low = high - low
        tr_high_close = (high - close_prev).abs()
        tr_low_close = (low - close_prev).abs()

        # Combine using pandas max
        tr = pd.DataFrame({
            'hl': tr_high_low,
            'hc': tr_high_close,
            'lc': tr_low_close
        }).max(axis=1)

        # Rolling mean
        atr = tr.rolling(
            window=window_int,
            min_periods=1
        ).mean()

        # Cache and return
        self._cache[cache_key] = atr
        return atr

    def _get_price_series(
        self, col_type: str, col_override: Optional[str] = None
    ) -> pd.Series:
        """
        Get price series by column type or override.

        Args:
            col_type: Type of price column ('open', 'high', 'low', 'close').
            col_override: Optional column name override.

        Returns:
            Price series.

        Raises:
            KeyError: If column not found.
        """
        col_name = col_override if col_override else self.price_cols.get(col_type)
        if not col_name or col_name not in self.data.columns:
            raise KeyError(f"Price column for type '{col_type}' ('{col_name}') not found.")
        return self.data[col_name]

    def fixed_percentage_stop_loss(
        self,
        signal: str,
        price_col: Optional[str] = None,
        percentage: float = 0.05,
        forward_fill: bool = False
    ) -> pd.DataFrame:
        """
        Calculate stop-loss using fixed percentage from entry price.

        For long positions: stop = price * (1 - percentage)
        For short positions: stop = price * (1 + percentage)

        Args:
            signal: Name of the signal column.
            price_col: Optional override for price column (defaults to 'close').
            percentage: Stop-loss distance as decimal (e.g., 0.05 for 5%).
            forward_fill: If True, forward-fill NaN values.

        Returns:
            DataFrame with '{signal}_stop_loss' column added.

        Raises:
            KeyError: If signal column not found.
            ValueError: If percentage not in (0, 1).
        """
        self._validate_signal_column(signal)
        self._validate_percentage(percentage)

        result_df = self.data.copy()
        price = self._get_price_series('close', price_col)

        long_stop = price * (1 - percentage)
        short_stop = price * (1 + percentage)

        # Floor long stops at minimum price
        long_stop = np.maximum(long_stop, MIN_STOP_PRICE)

        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        result_df.loc[result_df[signal] > 0, stop_loss_col] = long_stop
        result_df.loc[result_df[signal] < 0, stop_loss_col] = short_stop

        if forward_fill:
            result_df[stop_loss_col] = result_df[stop_loss_col].ffill()

        logger.debug("Calculated fixed percentage stop-loss for '%s' (%.1f%%)", signal, percentage * 100)
        return result_df

    def atr_stop_loss(
        self,
        signal: str,
        price_col: Optional[str] = None,
        window: int = 14,
        multiplier: float = 2.0,
        forward_fill: bool = True
    ) -> pd.DataFrame:
        """
        Calculate stop-loss using ATR (Average True Range).

        For long positions: stop = price - (ATR * multiplier)
        For short positions: stop = price + (ATR * multiplier)

        Args:
            signal: Name of the signal column.
            price_col: Optional override for price column (defaults to 'close').
            window: ATR calculation period.
            multiplier: ATR multiplier for stop distance.
            forward_fill: If True, forward-fill NaN values (default True).

        Returns:
            DataFrame with '{signal}_stop_loss' column added.

        Raises:
            KeyError: If signal column not found.
            ValueError: If multiplier not positive.
        """
        self._validate_signal_column(signal)
        self._validate_multiplier(multiplier)

        result_df = self.data.copy()
        atr = self._atr(window=window)
        price = self._get_price_series('close', price_col)

        stop_distance = atr * multiplier

        # Floor long stops at minimum price
        long_stop = np.maximum(price - stop_distance, MIN_STOP_PRICE)

        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        result_df.loc[result_df[signal] > 0, stop_loss_col] = long_stop
        result_df.loc[result_df[signal] < 0, stop_loss_col] = price + stop_distance

        if forward_fill:
            result_df[stop_loss_col] = result_df[stop_loss_col].ffill()

        logger.debug("Calculated ATR stop-loss for '%s' (window=%d, multiplier=%.1f)", signal, window, multiplier)
        return result_df

    def breakout_channel_stop_loss(
        self,
        signal: str,
        high_col: Optional[str] = None,
        low_col: Optional[str] = None,
        window: int = 20,
        forward_fill: bool = False
    ) -> pd.DataFrame:
        """
        Calculate stop-loss using breakout channel (swing highs/lows).

        For long positions: stop = rolling minimum of lows
        For short positions: stop = rolling maximum of highs

        Args:
            signal: Name of the signal column.
            high_col: Optional override for high column.
            low_col: Optional override for low column.
            window: Rolling window period.
            forward_fill: If True, forward-fill NaN values.

        Returns:
            DataFrame with '{signal}_stop_loss' column added.

        Raises:
            KeyError: If signal column not found.
            ValueError: If window not positive.
        """
        self._validate_signal_column(signal)
        window = self._validate_window(window)

        result_df = self.data.copy()
        high = self._get_price_series('high', high_col)
        low = self._get_price_series('low', low_col)

        swing_highs = high.rolling(window=window).max()
        swing_lows = low.rolling(window=window).min()

        # Floor swing lows at minimum price
        swing_lows = np.maximum(swing_lows, MIN_STOP_PRICE)

        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        result_df.loc[result_df[signal] > 0, stop_loss_col] = swing_lows
        result_df.loc[result_df[signal] < 0, stop_loss_col] = swing_highs

        if forward_fill:
            result_df[stop_loss_col] = result_df[stop_loss_col].ffill()

        logger.debug("Calculated breakout channel stop-loss for '%s' (window=%d)", signal, window)
        return result_df

    def moving_average_stop_loss(
        self,
        signal: str,
        close_col: Optional[str] = None,
        window: int = 50,
        offset: float = 0.0,
        forward_fill: bool = False
    ) -> pd.DataFrame:
        """
        Calculate stop-loss using moving average with optional offset.

        For long positions: stop = MA - offset
        For short positions: stop = MA + offset

        Args:
            signal: Name of the signal column.
            close_col: Optional override for close column.
            window: Moving average period.
            offset: Offset from MA (default 0).
            forward_fill: If True, forward-fill NaN values.

        Returns:
            DataFrame with '{signal}_stop_loss' column added.

        Raises:
            KeyError: If signal column not found.
            ValueError: If window not positive.
        """
        self._validate_signal_column(signal)
        window = self._validate_window(window)

        result_df = self.data.copy()
        close = self._get_price_series('close', close_col)
        ma = close.rolling(window=window).mean()

        # Floor long stops at minimum price
        long_stop = np.maximum(ma - offset, MIN_STOP_PRICE)

        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        result_df.loc[result_df[signal] > 0, stop_loss_col] = long_stop
        result_df.loc[result_df[signal] < 0, stop_loss_col] = ma + offset

        if forward_fill:
            result_df[stop_loss_col] = result_df[stop_loss_col].ffill()

        logger.debug("Calculated MA stop-loss for '%s' (window=%d, offset=%.2f)", signal, window, offset)
        return result_df

    def volatility_std_stop_loss(
        self,
        signal: str,
        close_col: Optional[str] = None,
        window: int = 20,
        multiplier: float = 1.5,
        forward_fill: bool = False
    ) -> pd.DataFrame:
        """
        Calculate stop-loss using rolling standard deviation.

        For long positions: stop = close - (std * multiplier)
        For short positions: stop = close + (std * multiplier)

        Args:
            signal: Name of the signal column.
            close_col: Optional override for close column.
            window: Rolling standard deviation period.
            multiplier: Standard deviation multiplier.
            forward_fill: If True, forward-fill NaN values.

        Returns:
            DataFrame with '{signal}_stop_loss' column added.

        Raises:
            KeyError: If signal column not found.
            ValueError: If multiplier not positive.
        """
        self._validate_signal_column(signal)
        self._validate_multiplier(multiplier)
        window = self._validate_window(window)

        result_df = self.data.copy()
        close = self._get_price_series('close', close_col)
        std = close.rolling(window=window).std()

        # Floor long stops at minimum price
        long_stop = np.maximum(close - (std * multiplier), MIN_STOP_PRICE)

        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        result_df.loc[result_df[signal] > 0, stop_loss_col] = long_stop
        result_df.loc[result_df[signal] < 0, stop_loss_col] = close + (std * multiplier)

        if forward_fill:
            result_df[stop_loss_col] = result_df[stop_loss_col].ffill()

        logger.debug("Calculated volatility std stop-loss for '%s' (window=%d, multiplier=%.1f)", signal, window, multiplier)
        return result_df

    def support_resistance_stop_loss(
        self,
        signal: str,
        high_col: Optional[str] = None,
        low_col: Optional[str] = None,
        window: int = 20,
        forward_fill: bool = False
    ) -> pd.DataFrame:
        """
        Calculate stop-loss using support/resistance levels.

        Alias for breakout_channel_stop_loss.

        Args:
            signal: Name of the signal column.
            high_col: Optional override for high column.
            low_col: Optional override for low column.
            window: Rolling window period.
            forward_fill: If True, forward-fill NaN values.

        Returns:
            DataFrame with '{signal}_stop_loss' column added.
        """
        return self.breakout_channel_stop_loss(signal, high_col, low_col, window, forward_fill)

    def classified_pivot_stop_loss(
        self,
        signal: str,
        price_col: Optional[str] = None,
        high_col: Optional[str] = None,
        low_col: Optional[str] = None,
        atr_window: int = 14,
        atr_multiplier: float = 1.5,
        swing_window: int = 20,
        distance_threshold: float = 0.01,
        retracement_level: float = 0.618,
        forward_fill: bool = False
    ) -> pd.DataFrame:
        """
        Calculate stop-loss using classified pivot points with ATR and retracement.

        Combines ATR-based stops with Fibonacci retracement levels.

        Args:
            signal: Name of the signal column.
            price_col: Optional override for price column.
            high_col: Optional override for high column.
            low_col: Optional override for low column.
            atr_window: ATR calculation period.
            atr_multiplier: ATR multiplier for base stop.
            swing_window: Window for swing high/low calculation.
            distance_threshold: Minimum stop distance as fraction of price.
            retracement_level: Fibonacci retracement level (default 0.618).
            forward_fill: If True, forward-fill NaN values.

        Returns:
            DataFrame with '{signal}_stop_loss' column added.

        Raises:
            KeyError: If signal column not found.
            ValueError: If parameters are invalid.
        """
        self._validate_signal_column(signal)
        self._validate_multiplier(atr_multiplier)
        swing_window = self._validate_window(swing_window, "swing_window")

        result_df = self.data.copy()
        close = self._get_price_series('close', price_col)
        high = self._get_price_series('high', high_col)
        low = self._get_price_series('low', low_col)
        atr = self._atr(window=atr_window)

        swing_low = low.rolling(window=swing_window).min()
        swing_high = high.rolling(window=swing_window).max()

        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan

        is_long = result_df[signal] > 0
        is_short = result_df[signal] < 0

        # Long Stops logic
        base_stop_long = close - atr * atr_multiplier
        retrace_stop_long = swing_low + (swing_high - swing_low) * retracement_level
        long_stops = np.minimum(base_stop_long, retrace_stop_long)
        # Floor at minimum price
        long_stops = np.maximum(long_stops, MIN_STOP_PRICE)
        result_df.loc[is_long, stop_loss_col] = long_stops[is_long]

        # Short Stops logic
        base_stop_short = close + atr * atr_multiplier
        retrace_stop_short = swing_high - (swing_high - swing_low) * retracement_level
        result_df.loc[is_short, stop_loss_col] = np.maximum(base_stop_short, retrace_stop_short)[is_short]

        # Distance threshold enforcement with zero-guard
        mask_nonzero = close > 0

        is_long_too_close = is_long & mask_nonzero & (
            np.abs(close - result_df[stop_loss_col]) / close < distance_threshold
        )
        result_df.loc[is_long_too_close, stop_loss_col] = np.maximum(
            close * (1 - distance_threshold), MIN_STOP_PRICE
        )

        is_short_too_close = is_short & mask_nonzero & (
            np.abs(close - result_df[stop_loss_col]) / close < distance_threshold
        )
        result_df.loc[is_short_too_close, stop_loss_col] = close * (1 + distance_threshold)

        if forward_fill:
            result_df[stop_loss_col] = result_df[stop_loss_col].ffill()

        logger.debug(
            "Calculated classified pivot stop-loss for '%s' (ATR window=%d, swing window=%d)",
            signal, atr_window, swing_window
        )
        return result_df

    def get_stop_loss(self, signal: str, method: str, **kwargs) -> pd.DataFrame:
        """
        Generic interface to calculate stop-loss using any method.

        Args:
            signal: Name of the signal column.
            method: Stop-loss method name. Available methods:
                - 'fixed_percentage': Fixed percentage from price
                - 'atr': ATR-based stop
                - 'breakout_channel': Swing high/low based
                - 'support_resistance': Alias for breakout_channel
                - 'moving_average': Moving average based
                - 'volatility_std': Standard deviation based
                - 'classified_pivot': Pivot with ATR and retracement
            **kwargs: Method-specific parameters.

        Returns:
            DataFrame with '{signal}_stop_loss' column added.

        Raises:
            ValueError: If method is unknown.
        """
        method_map = {
            'fixed_percentage': self.fixed_percentage_stop_loss,
            'atr': self.atr_stop_loss,
            'breakout_channel': self.breakout_channel_stop_loss,
            'support_resistance': self.support_resistance_stop_loss,
            'moving_average': self.moving_average_stop_loss,
            'volatility_std': self.volatility_std_stop_loss,
            'classified_pivot': self.classified_pivot_stop_loss,
        }

        if method not in method_map:
            logger.error("Unknown stop-loss method: %s. Available: %s", method, list(method_map.keys()))
            raise ValueError(f"Unknown stop-loss method: {method}. Available: {list(method_map)}")

        # Filter kwargs to only those accepted by the method
        method_func = method_map[method]
        safe_kwargs = self._filter_kwargs(method_func.__name__, **kwargs)

        logger.debug("Calling stop-loss method '%s' with kwargs: %s", method, safe_kwargs)
        return method_func(signal, **safe_kwargs)
