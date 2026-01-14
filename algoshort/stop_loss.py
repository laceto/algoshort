import pandas as pd
import numpy as np

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
            self._data = None
            self._cache: dict[str, pd.Series] = {}
            self.price_cols = {}
            
            # Trigger the setter logic for initial data
            self.data = data

    @property
    def data(self) -> pd.DataFrame:
        """Access the current DataFrame segment."""
        return self._data

    @data.setter
    def data(self, new_data: pd.DataFrame):
        """
        Updates the internal data, clears the cache, and re-detects columns.
        This enables 'self.calc.data = oos_data' to work correctly in loops.
        """
        if new_data is None or new_data.empty:
            raise ValueError("DataFrame cannot be empty.")
        
        # 1. Update the data (operating on a copy for safety)
        self._data = new_data.copy()
        
        # 2. CRITICAL: Clear the cache so old metrics don't leak into new data segments
        self._cache = {}
        
        # 3. Re-detect OHLC column names
        self._detect_ohlc_columns()

    def _detect_ohlc_columns(self):
        """Internal helper to detect column naming conventions."""
        absolute_cols = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}
        relative_cols = {'open': 'r_open', 'high': 'r_high', 'low': 'r_low', 'close': 'r_close'}
        
        if all(col in self._data.columns for col in absolute_cols.values()):
            self.price_cols = absolute_cols
        elif all(col in self._data.columns for col in relative_cols.values()):
            self.price_cols = relative_cols
        else:
            raise KeyError("Required OHLC columns not found in the provided DataFrame.")

    def _atr(self, window: int = 14) -> pd.Series:
        cache_key = f"ATR_{window}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        high = self.data[self.price_cols['high']]
        low = self.data[self.price_cols['low']]
        close = self.data[self.price_cols['close']]
        
        tr = np.maximum(high - low, np.abs(high - close.shift()))
        tr = np.maximum(tr, np.abs(low - close.shift()))
        atr = tr.rolling(window=window).mean()
        
        self._cache[cache_key] = atr
        return atr

    def _get_price_series(self, col_type: str, col_override: str = None) -> pd.Series:
        col_name = col_override if col_override else self.price_cols.get(col_type)
        if not col_name or col_name not in self.data.columns:
             raise KeyError(f"Price column for type '{col_type}' ('{col_name}') not found.")
        return self.data[col_name]

    def fixed_percentage_stop_loss(self, signal: str, price_col: str = None, percentage: float = 0.05) -> pd.DataFrame:
        result_df = self.data.copy()
        price = self._get_price_series('close', price_col)
        
        long_stop = price * (1 - percentage)
        short_stop = price * (1 + percentage)

        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        result_df.loc[result_df[signal] > 0, stop_loss_col] = long_stop
        result_df.loc[result_df[signal] < 0, stop_loss_col] = short_stop
        
        return result_df

    def atr_stop_loss(self, signal: str, price_col: str = None, window: int = 14, multiplier: float = 2.0) -> pd.DataFrame:
        result_df = self.data.copy()
        atr = self._atr(window=window)
        price = self._get_price_series('close', price_col)

        stop_distance = atr * multiplier
        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        result_df.loc[result_df[signal] > 0, stop_loss_col] = price - stop_distance
        result_df.loc[result_df[signal] < 0, stop_loss_col] = price + stop_distance
        result_df[stop_loss_col] = result_df[stop_loss_col].ffill()
        
        return result_df

    def breakout_channel_stop_loss(self, signal: str, high_col: str = None, low_col: str = None, window: int = 20) -> pd.DataFrame:
        result_df = self.data.copy()
        high = self._get_price_series('high', high_col)
        low = self._get_price_series('low', low_col)
        
        swing_highs = high.rolling(window=window).max()
        swing_lows = low.rolling(window=window).min()
        
        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        result_df.loc[result_df[signal] > 0, stop_loss_col] = swing_lows
        result_df.loc[result_df[signal] < 0, stop_loss_col] = swing_highs
        
        return result_df

    def moving_average_stop_loss(self, signal: str, close_col: str = None, window: int = 50, offset: float = 0.0) -> pd.DataFrame:
        result_df = self.data.copy()
        close = self._get_price_series('close', close_col)
        ma = close.rolling(window=window).mean()
        
        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        result_df.loc[result_df[signal] > 0, stop_loss_col] = ma - offset
        result_df.loc[result_df[signal] < 0, stop_loss_col] = ma + offset
        
        return result_df

    def volatility_std_stop_loss(self, signal: str, close_col: str = None, window: int = 20, multiplier: float = 1.5) -> pd.DataFrame:
        result_df = self.data.copy()
        close = self._get_price_series('close', close_col)
        std = close.rolling(window=window).std()
        
        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        result_df.loc[result_df[signal] > 0, stop_loss_col] = close - (std * multiplier)
        result_df.loc[result_df[signal] < 0, stop_loss_col] = close + (std * multiplier)
        
        return result_df

    def support_resistance_stop_loss(self, signal: str, high_col: str = None, low_col: str = None, window: int = 20) -> pd.DataFrame:
        return self.breakout_channel_stop_loss(signal, high_col, low_col, window)

    def classified_pivot_stop_loss(self, signal: str, price_col: str = None, high_col: str = None, low_col: str = None, 
                                    atr_window: int = 14, distance_threshold: float = 0.01, retest_threshold: float = 0.02, 
                                    retracement_level: float = 0.618, magnitude_level: int = 2) -> pd.DataFrame:
        result_df = self.data.copy()
        close = self._get_price_series('close', price_col)
        high = self._get_price_series('high', high_col)
        low = self._get_price_series('low', low_col)
        atr = self._atr(window=atr_window)
        
        swing_window = 20
        swing_low = low.rolling(window=swing_window).min()
        swing_high = high.rolling(window=swing_window).max()
        
        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        
        is_long = result_df[signal] > 0
        is_short = result_df[signal] < 0
        
        # Long Stops logic
        base_stop_long = close - atr * 1.5
        retrace_stop_long = swing_low + (swing_high - swing_low) * retracement_level
        result_df.loc[is_long, stop_loss_col] = np.minimum(base_stop_long, retrace_stop_long)

        # Short Stops logic
        base_stop_short = close + atr * 1.5
        retrace_stop_short = swing_high - (swing_high - swing_low) * retracement_level
        result_df.loc[is_short, stop_loss_col] = np.maximum(base_stop_short, retrace_stop_short)

        # Distance threshold enforcement
        is_long_too_close = is_long & (np.abs(close - result_df[stop_loss_col]) / close < distance_threshold)
        result_df.loc[is_long_too_close, stop_loss_col] = close - close * distance_threshold

        is_short_too_close = is_short & (np.abs(close - result_df[stop_loss_col]) / close < distance_threshold)
        result_df.loc[is_short_too_close, stop_loss_col] = close + close * distance_threshold
        
        return result_df

    def get_stop_loss(self, signal: str, method: str, **kwargs) -> pd.DataFrame:
        method_map = {
            'fixed_percentage': self.fixed_percentage_stop_loss,
            'atr': self.atr_stop_loss,
            'breakout_channel': self.breakout_channel_stop_loss,
            'moving_average': self.moving_average_stop_loss,
            'volatility_std': self.volatility_std_stop_loss,
            'support_resistance': self.support_resistance_stop_loss,
            'classified_pivot': self.classified_pivot_stop_loss,
        }
        if method in method_map:
            return method_map[method](signal, **kwargs)
        raise ValueError(f"Unknown stop-loss method: {method}")
    
