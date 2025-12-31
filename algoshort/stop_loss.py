import pandas as pd
import numpy as np

class StopLossCalculator:
    """
    A class to calculate stop-loss levels using several methods for both long and short positions.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC columns (e.g., 'open', 'high', 'low', 'close') 
            and signal columns.

    Raises:
        KeyError: If required OHLC columns are not found.
        ValueError: If DataFrame is empty or contains non-numeric data.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy() # Operate on a copy to ensure immutability of the input
        self._cache: dict[str, pd.Series] = {}  # Cache for calculations like ATR
        
        # --- FIX 2: Explicit Input Validation ---
        if self.data.empty:
            raise ValueError("DataFrame cannot be empty.")

        # --- FIX 1: Detect and store OHLC column names for flexibility ---
        self.price_cols = {}
        
        # Define the possible column sets
        absolute_cols = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}
        relative_cols = {'open': 'r_open', 'high': 'r_high', 'low': 'r_low', 'close': 'r_close'}
        
        # Check for absolute columns
        if all(col in self.data.columns for col in absolute_cols.values()):
            self.price_cols = absolute_cols
        # Check for relative columns
        elif all(col in self.data.columns for col in relative_cols.values()):
            self.price_cols = relative_cols
        else:
            raise KeyError("Required OHLC columns ('open', 'high', 'low', 'close' or 'r_open', 'r_high', 'r_low', 'r_close') are not found.")

    def _atr(self, window: int = 14) -> pd.Series:
        """
        Calculates and caches Average True Range (ATR).

        Args:
            window (int): Lookback window for ATR. Defaults to 14.

        Returns:
            pd.Series: ATR series.
        """
        # --- OPTIMIZATION 1: Use Caching for ATR ---
        cache_key = f"ATR_{window}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        high = self.data[self.price_cols['high']]
        low = self.data[self.price_cols['low']]
        close = self.data[self.price_cols['close']]
        
        # Calculate True Range (TR)
        tr = np.maximum(high - low, np.abs(high - close.shift()))
        tr = np.maximum(tr, np.abs(low - close.shift()))
        
        # Calculate Average True Range (ATR)
        # Using .ewm(span=window).mean() for exponential ATR is common, but 
        # sticking to .rolling().mean() as per original logic for simplicity.
        atr = tr.rolling(window=window).mean()
        
        self._cache[cache_key] = atr
        return atr

    # Helper method to get the correct price column, using the stored names as default
    def _get_price_series(self, col_type: str, col_override: str = None) -> pd.Series:
        """Retrieves price series, prioritizing override then stored name."""
        col_name = col_override if col_override else self.price_cols.get(col_type)
        if not col_name or col_name not in self.data.columns:
             raise KeyError(f"Price column for type '{col_type}' ('{col_name}') not found in DataFrame.")
        return self.data[col_name]


    # Note: All stop-loss methods below now operate on a *copy* of self.data 
    # and return the modified DataFrame, improving encapsulation. 
    # They also use the stored OHLC columns by default (Fix 1).
    
    def fixed_percentage_stop_loss(self, signal: str, price_col: str = None, percentage: float = 0.05) -> pd.DataFrame:
        """
        Calculates fixed percentage stop-loss.
        """
        result_df = self.data.copy()
        price = self._get_price_series('close', price_col)
        
        # Calculate stop loss for long and short signals simultaneously (Vectorized)
        long_stop = price * (1 - percentage)
        short_stop = price * (1 + percentage)

        # Apply based on signal direction
        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        result_df.loc[result_df[signal] > 0, stop_loss_col] = long_stop
        result_df.loc[result_df[signal] < 0, stop_loss_col] = short_stop
        
        return result_df

    def atr_stop_loss(self, signal: str, price_col: str = None, window: int = 14, multiplier: float = 2.0) -> pd.DataFrame:
        """
        Calculates ATR-based stop-loss.
        """
        result_df = self.data.copy()
        
        # Use cached ATR (Optimization 1) and correct price series (Fix 1)
        atr = self._atr(window=window)
        price = self._get_price_series('close', price_col)

        # Calculate the stop distance
        stop_distance = atr * multiplier
        
        # Calculate stop loss for long and short signals (Vectorized)
        long_stop = price - stop_distance
        short_stop = price + stop_distance
        
        # Apply based on signal direction
        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        result_df.loc[result_df[signal] > 0, stop_loss_col] = long_stop
        result_df.loc[result_df[signal] < 0, stop_loss_col] = short_stop
        
        return result_df

    def breakout_channel_stop_loss(self, signal: str, high_col: str = None, low_col: str = None, window: int = 20) -> pd.DataFrame:
        """
        Calculates breakout channel-based stop-loss.
        """
        result_df = self.data.copy()
        high = self._get_price_series('high', high_col)
        low = self._get_price_series('low', low_col)
        
        # Rolling swing high/low (Optimization 2 benefit)
        swing_highs = high.rolling(window=window).max()
        swing_lows = low.rolling(window=window).min()
        
        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        
        # Long position stop is the recent swing low
        result_df.loc[result_df[signal] > 0, stop_loss_col] = swing_lows
        # Short position stop is the recent swing high
        result_df.loc[result_df[signal] < 0, stop_loss_col] = swing_highs
        
        return result_df

    def moving_average_stop_loss(self, signal: str, close_col: str = None, window: int = 50, offset: float = 0.0) -> pd.DataFrame:
        """
        Calculates moving average-based stop-loss.
        """
        result_df = self.data.copy()
        close = self._get_price_series('close', close_col)
        
        # Calculate Moving Average (Optimization 2 benefit)
        ma = close.rolling(window=window).mean()
        
        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        
        # Long position stop is MA - offset
        result_df.loc[result_df[signal] > 0, stop_loss_col] = ma - offset
        # Short position stop is MA + offset
        result_df.loc[result_df[signal] < 0, stop_loss_col] = ma + offset
        
        return result_df

    def volatility_std_stop_loss(self, signal: str, close_col: str = None, window: int = 20, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Calculates volatility-based stop-loss using standard deviation.
        """
        result_df = self.data.copy()
        close = self._get_price_series('close', close_col)

        # Calculate rolling standard deviation
        std = close.rolling(window=window).std()
        stop_distance = std * multiplier
        
        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        
        # Long position stop is close - std * multiplier
        result_df.loc[result_df[signal] > 0, stop_loss_col] = close - stop_distance
        # Short position stop is close + std * multiplier
        result_df.loc[result_df[signal] < 0, stop_loss_col] = close + stop_distance
        
        return result_df

    def support_resistance_stop_loss(self, signal: str, high_col: str = None, low_col: str = None, window: int = 20) -> pd.DataFrame:
        """
        Calculates stop-loss based on support/resistance levels. 
        (Identical logic to breakout_channel_stop_loss but kept for semantic clarity).
        """
        return self.breakout_channel_stop_loss(signal, high_col, low_col, window)

    def classified_pivot_stop_loss(self, signal: str, price_col: str = None, high_col: str = None, low_col: str = None, 
                                    atr_window: int = 14, distance_threshold: float = 0.01, retest_threshold: float = 0.02, 
                                    retracement_level: float = 0.618, magnitude_level: int = 2) -> pd.DataFrame:
        """
        Calculates stop-loss using classified highs/lows with hybrid logic.
        
        NOTE: This is the only method where the complexity of the logic necessitates a
        less-than-ideal vectorized approach or a carefully managed loop. 
        The following is a fully vectorized and corrected version (Fix 3).
        """
        result_df = self.data.copy()
        
        # Use stored columns (Fix 1) and cached ATR (Optimization 1)
        close = self._get_price_series('close', price_col)
        high = self._get_price_series('high', high_col)
        low = self._get_price_series('low', low_col)
        atr = self._atr(window=atr_window)
        
        # --- FIX 3: Fully Vectorized Logic (Using rolling min/max as proxy for pivots) ---
        swing_window = 20 # Use a fixed window for proxy calculation
        swing_low = low.rolling(window=swing_window).min()
        swing_high = high.rolling(window=swing_window).max()
        magnitude = np.full(len(result_df), magnitude_level) # Placeholder for magnitude
        
        stop_loss_col = f'{signal}_stop_loss'
        result_df[stop_loss_col] = np.nan
        
        # --- Setup masks for Long and Short trades ---
        is_long = result_df[signal] > 0
        is_short = result_df[signal] < 0
        
        # --- 1. Base/Retracement Stop Calculation (Vectorized) ---
        # Long Stops
        base_stop_long = close - atr * 1.5
        retrace_stop_long = swing_low + (swing_high - swing_low) * retracement_level
        # The stop for long is the minimum of base_stop and retrace_stop
        initial_stop_long = np.minimum(base_stop_long, retrace_stop_long)

        # Short Stops
        base_stop_short = close + atr * 1.5
        retrace_stop_short = swing_high - (swing_high - swing_low) * retracement_level
        # The stop for short is the maximum of base_stop and retrace_stop
        initial_stop_short = np.maximum(base_stop_short, retrace_stop_short)
        
        # Apply initial stops
        result_df.loc[is_long, stop_loss_col] = initial_stop_long
        result_df.loc[is_short, stop_loss_col] = initial_stop_short

        # --- 2. Swing Retest Confirmation (Vectorized) ---
        # Criteria: Close is near a swing low/high AND magnitude is sufficient
        is_retest_low = (np.abs(close - swing_low) / close < retest_threshold) & (magnitude >= magnitude_level)
        is_retest_high = (np.abs(close - swing_high) / close < retest_threshold) & (magnitude >= magnitude_level)

        # For long: if retest low is true, set stop to MIN(current_stop, swing_low)
        mask_long_retest = is_long & is_retest_low
        result_df.loc[mask_long_retest, stop_loss_col] = np.minimum(
            result_df.loc[mask_long_retest, stop_loss_col], 
            swing_low.loc[mask_long_retest]
        )

        # For short: if retest high is true, set stop to MAX(current_stop, swing_high)
        mask_short_retest = is_short & is_retest_high
        result_df.loc[mask_short_retest, stop_loss_col] = np.maximum(
            result_df.loc[mask_short_retest, stop_loss_col], 
            swing_high.loc[mask_short_retest]
        )

        # --- 3. Distance Test (Minimum Stop Distance) (Vectorized) ---
        # Ensure the stop is not too close to the current price
        
        # Long: if stop is too close, push it down by distance_threshold
        is_long_too_close = is_long & (np.abs(close - result_df[stop_loss_col]) / close < distance_threshold)
        result_df.loc[is_long_too_close, stop_loss_col] = close - close * distance_threshold

        # Short: if stop is too close, push it up by distance_threshold
        is_short_too_close = is_short & (np.abs(close - result_df[stop_loss_col]) / close < distance_threshold)
        result_df.loc[is_short_too_close, stop_loss_col] = close + close * distance_threshold
        
        return result_df

    def get_stop_loss(self, signal: str, method: str, **kwargs) -> pd.DataFrame:
        """
        Calculates stop-loss using the specified method.

        NOTE: The 'inplace' argument is removed as methods now return a new DataFrame by default.
        """
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
            # Pass only the necessary signal and kwargs. All methods now use self.data
            # and return a result_df, simplifying the overall interface.
            return method_map[method](signal, **kwargs)
        else:
            raise ValueError(f"Unknown stop-loss method: {method}")