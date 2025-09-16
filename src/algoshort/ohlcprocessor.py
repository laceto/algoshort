import pandas as pd
import yfinance as yf
import logging

# Configure logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OHLCProcessor:
    """
    A class to process OHLC data.
    
    Data acquisition methods (`download_data` and `get_ohlc_data`) have been removed,
    and this class now focuses purely on processing the data passed to it.
    """
    def __init__(self):
        """Initializes the OHLCProcessor."""
        pass

    def _calculate_relative(self, df: pd.DataFrame, _o: str, _h: str, _l: str, _c: str, bm_df: pd.DataFrame, bm_col: str, dgt: int, rebase: bool = True) -> pd.DataFrame:
        '''
        Calculates relative prices of an OHLC dataset against a benchmark.
        
        This is an internal helper method used by `calculate_relative_prices`.

        Args:
            df (pd.DataFrame): The primary DataFrame with OHLC data.
            _o (str): Name of the 'open' column in df.
            _h (str): Name of the 'high' column in df.
            _l (str): Name of the 'low' column in df.
            _c (str): Name of the 'close' column in df.
            bm_df (pd.DataFrame): The benchmark DataFrame.
            bm_col (str): The column name to use from the benchmark DataFrame for calculation.
            dgt (int): The number of decimal places for rounding.
            rebase (bool): If True, rebases the benchmark to 1.0 at the start.

        Returns:
            pd.DataFrame: A new DataFrame with the 'r' prefixed relative columns.
        '''
        try:
            logging.debug("Starting relative price calculation...")
            
            # Create a copy to avoid modifying the original benchmark DataFrame
            bm_df = bm_df.copy()
            
            # Rename the benchmark column for consistent merging
            bm_df.rename(columns={bm_col: 'bm'}, inplace=True)
            
            # Ensure 'date' is a column for merging if it's the index
            if 'date' not in bm_df.columns:
                bm_df = bm_df.reset_index()
                bm_df.rename(columns={bm_df.columns[0]: 'date'}, inplace=True)

            logging.info(f"Merging primary data (shape: {df.shape}) with benchmark data (shape: {bm_df.shape}).")
            # Merge the primary and benchmark dataframes
            merged_df = pd.merge(df, bm_df[['date', 'bm']], how='left', on='date')
            logging.info(f"Merge completed. New DataFrame shape: {merged_df.shape}")
            
            # Calculate the benchmark adjustment factor (fwd fill missing values)
            merged_df['bmfx'] = round(merged_df['bm'], dgt).ffill()
            
            # Apply rebase if requested
            if rebase:
                if not merged_df['bmfx'].empty and merged_df['bmfx'].iloc[0] != 0:
                    logging.info(f"Rebasing benchmark to 1.0 using the first value: {merged_df['bmfx'].iloc[0]}")
                    merged_df['bmfx'] = merged_df['bmfx'].div(merged_df['bmfx'].iloc[0])
                else:
                    logging.warning("Benchmark data is empty or first value is zero, cannot rebase.")
            
            # Calculate relative prices for OHLC columns
            logging.info("Calculating relative OHLC prices...")
            merged_df['r' + str(_o)] = round(merged_df[_o].div(merged_df['bmfx']), dgt)
            merged_df['r' + str(_h)] = round(merged_df[_h].div(merged_df['bmfx']), dgt)
            merged_df['r' + str(_l)] = round(merged_df[_l].div(merged_df['bmfx']), dgt)
            merged_df['r' + str(_c)] = round(merged_df[_c].div(merged_df['bmfx']), dgt)
            logging.info("Relative price calculation complete.")
            
            # Drop the temporary benchmark columns
            merged_df = merged_df.drop(['bm', 'bmfx'], axis=1)
            
            return merged_df
        except Exception as e:
            logging.error(f"An error occurred during relative price calculation: {e}")
            raise
    
    def calculate_relative_prices(self, 
                                stock_data: pd.DataFrame, 
                                benchmark_data: pd.DataFrame,
                                benchmark_column: str = 'close',
                                digits: int = 4,
                                rebase: bool = True) -> pd.DataFrame:
        """
        Calculate relative prices using the internal `_calculate_relative()` function.
        
        Args:
            stock_data (pd.DataFrame): Primary DataFrame for analysis.
            benchmark_data (pd.DataFrame): Benchmark DataFrame for comparison.
            benchmark_column (str): Column to use from benchmark data.
            digits (int): Decimal places for rounding.
            rebase (bool): Whether to rebase benchmark to 1.0.
            
        Returns:
            pd.DataFrame: DataFrame with relative prices.
            
        Raises:
            ValueError: If calculation fails.
        """
        
        try:
            # Ensure proper column naming, especially for date
            if 'date' not in stock_data.columns:
                 stock_data = stock_data.reset_index()
                 stock_data.rename(columns={'Date': 'date'}, inplace=True)
            
            result = self._calculate_relative(
                df=stock_data,
                _o='open', _h='high', _l='low', _c='close',
                bm_df=benchmark_data,
                bm_col=benchmark_column,
                dgt=digits,
                rebase=rebase
            )
            
            return result
        
        except Exception as e:
            logging.error(f"Failed to calculate relative prices: {e}")
            raise ValueError(f"Calculation failed.")