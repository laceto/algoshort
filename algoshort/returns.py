import pandas as pd
from typing import Dict, Tuple
import numpy as np


class ReturnsCalculator:
    """
    A class to calculate returns and equity curves for trading strategies based on OHLC data and regime signals.

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
        self._cache: Dict[str, pd.Series] = {}  # Cache for rolling calculations

    @staticmethod
    def _lower_upper_OHLC(data: pd.DataFrame, relative: bool = False) -> Tuple[str, str, str, str]:
        """
        Determines OHLC column names based on DataFrame columns and relative flag.

        Args:
            data (pd.DataFrame): DataFrame to check for OHLC columns.
            relative (bool): If True, use relative OHLC columns (e.g., 'r_open'); else absolute (e.g., 'open').

        Returns:
            Tuple[str, str, str, str]: Column names for open, high, low, close.

        Raises:
            KeyError: If required OHLC columns are not found.
        """
        rel = 'r' if relative else ''
        if 'Open' in data.columns:
            ohlc = [f'{rel}Open', f'{rel}High', f'{rel}Low', f'{rel}Close']
        elif 'open' in data.columns:
            ohlc = [f'{rel}open', f'{rel}high', f'{rel}low', f'{rel}close']
        else:
            raise KeyError("No 'Open' or 'open' column found in DataFrame.")

        for col in ohlc:
            if col not in data.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        
        return tuple(ohlc)

    def get_returns(self, df: pd.DataFrame, signal: str, relative: bool = False, inplace: bool = False) -> pd.DataFrame:
        """
        Calculates returns based on a regime signal, including daily changes, cumulative returns, and stop-loss levels.

        Args:
            df (pd.DataFrame): DataFrame containing OHLC data and the regime signal column.
            signal (str): Name of the signal column (e.g., 'bo_5', 'tt_52', 'sma_358').
            relative (bool): If True, use relative OHLC columns ('r_close', etc.); else absolute ('close', etc.). Defaults to False.
            inplace (bool): If True, modify input DataFrame; else return a new one. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with additional columns.

        Raises:
            KeyError: If required columns are missing.
            ValueError: If DataFrame is empty or data types are non-numeric.
        """
        try:
            # Validate inputs
            if df.empty:
                raise ValueError("Input DataFrame is empty.")
            
            # Get OHLC column names
            _o, _h, _l, _c = self._lower_upper_OHLC(data=df, relative=relative)
            
            # Validate signal column
            if signal not in df.columns:
                raise KeyError(f"Signal column '{signal}' not found in DataFrame.")
            
            # Validate numeric data
            if not np.issubdtype(df[_c].dtype, np.number) or \
            not np.issubdtype(df[_h].dtype, np.number) or \
            not np.issubdtype(df[_l].dtype, np.number) or \
            not np.issubdtype(df[signal].dtype, np.number):
                raise ValueError("OHLC and signal columns must contain numeric data.")

            # Work on copy if not inplace
            result_df = df if inplace else df.copy()
            
            # Fill NaN in signal column with 0
            signal_filled = result_df[signal].fillna(0)
            
            # Pre-compute reusable intermediate values
            close_prices = result_df[_c]
            price_diff = close_prices.diff()
            lagged_signal = signal_filled.shift()
            
            # Calculate daily price changes
            chg1D = price_diff * lagged_signal
            
            # Calculate returns
            pct_returns = close_prices.pct_change() * lagged_signal
            log_returns = np.log(close_prices / close_prices.shift()) * lagged_signal
            
            # Build new columns DataFrame
            new_columns_df = pd.DataFrame({
                f'{signal}_chg1D': chg1D,
                f'{signal}_chg1D_fx': chg1D,
                f'{signal}_PL_cum': chg1D.cumsum(),
                f'{signal}_PL_cum_fx': chg1D.cumsum(),
                f'{signal}_returns': pct_returns,
                f'{signal}_log_returns': log_returns,
                f'{signal}_cumul': np.exp(log_returns.cumsum()) - 1
            }, index=result_df.index)
            
            # SINGLE BATCH ASSIGNMENT - NO FRAGMENTATION
            # Using concat with axis=1 to add all columns at once
            result_df = pd.concat([result_df, new_columns_df], axis=1)
            
            # Update signal column (fillna)
            result_df[signal] = signal_filled
            
            return result_df

        except Exception as e:
            raise ValueError(f"Error computing returns: {str(e)}")