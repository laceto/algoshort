import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Union, List, Dict, Optional, Tuple
import logging
import warnings
from pathlib import Path
from algoshort.utils import relative
import os

class YFinanceDataHandler:
    """
    A comprehensive class for downloading, managing, and preparing financial data using yfinance.
    
    This class provides easy-to-use methods for downloading stock data, managing multiple symbols,
    preparing data for analysis, and integrating with relative price calculations.
    
    Attributes:
        symbols (List[str]): List of symbols being tracked
        data (Dict[str, pd.DataFrame]): Dictionary storing data for each symbol
        period_map (Dict[str, str]): Mapping of common period names to yfinance periods
        interval_map (Dict[str, str]): Mapping of common interval names to yfinance intervals
    
    Example:
        >>> handler = YFinanceDataHandler()
        >>> handler.download_data('AAPL', period='1y')
        >>> data = handler.get_data('AAPL')
        >>> handler.calculate_relative_prices('AAPL', benchmark_symbol='^GSPC')
    """
    
    def __init__(self, cache_dir: Optional[str] = None, enable_logging: bool = True):
        """
        Initialize the YFinanceDataHandler.
        
        Args:
            cache_dir (str, optional): Directory to cache downloaded data
            enable_logging (bool): Enable logging for operations
        """
        self.symbols = []
        self.data = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Setup logging
        if enable_logging:
            logging.basicConfig(level=logging.INFO, 
                              format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Common period mappings for user convenience
        self.period_map = {
            '1d': '1d', '5d': '5d', '1mo': '1mo', '3mo': '3mo', '6mo': '6mo',
            '1y': '1y', '2y': '2y', '5y': '5y', '10y': '10y', 'ytd': 'ytd', 'max': 'max',
            # User-friendly aliases
            'day': '1d', 'week': '5d', 'month': '1mo', 'quarter': '3mo',
            'half_year': '6mo', 'year': '1y', 'two_years': '2y', 
            'five_years': '5y', 'ten_years': '10y', 'all_time': 'max'
        }
        
        # Interval mappings
        self.interval_map = {
            '1m': '1m', '2m': '2m', '5m': '5m', '15m': '15m', '30m': '30m',
            '60m': '60m', '90m': '90m', '1h': '1h', '1d': '1d', '5d': '5d',
            '1wk': '1wk', '1mo': '1mo', '3mo': '3mo',
            # User-friendly aliases
            'minute': '1m', 'hourly': '1h', 'daily': '1d', 'weekly': '1wk', 'monthly': '1mo'
        }

    def download_data(self, 
                     symbols: Union[str, List[str]], 
                     period: str = '1y',
                     interval: str = '1d',
                     start: Optional[str] = None,
                     end: Optional[str] = None,
                     auto_adjust: bool = True,
                     prepost: bool = False,
                     threads: bool = True,
                     group_by_ticker: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Download financial data for one or more symbols.
        
        Args:
            symbols (str or List[str]): Stock symbol(s) to download
            period (str): Period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            start (str, optional): Start date (YYYY-MM-DD)
            end (str, optional): End date (YYYY-MM-DD)
            auto_adjust (bool): Automatically adjust for splits and dividends
            prepost (bool): Include pre/post market data
            threads (bool): Use threading for multiple downloads
            group_by_ticker (bool): Group columns by ticker
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with symbol as key and DataFrame as value
            
        Raises:
            ValueError: If invalid period/interval or no data downloaded
            Exception: For network or data retrieval issues
        """
        
        try:
            # Handle single symbol input
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # Validate and map periods/intervals
            period = self.period_map.get(period, period)
            interval = self.interval_map.get(interval, interval)
            
            # Validate period and interval combination
            self._validate_period_interval(period, interval)
            
            logging.info(f"Downloading data for {symbols} - Period: {period}, Interval: {interval}")
            
            # Download data
            if len(symbols) == 1:
                ticker = yf.Ticker(symbols[0])
                if start and end:
                    data = ticker.history(start=start, end=end, interval=interval, 
                                        auto_adjust=auto_adjust, prepost=prepost)
                else:
                    data = ticker.history(period=period, interval=interval,
                                        auto_adjust=auto_adjust, prepost=prepost)
                
                if data.empty:
                    raise ValueError(f"No data found for symbol {symbols[0]}")
                
                # Clean and prepare data
                cleaned_data = self._clean_data(data, symbols[0])
                self.data[symbols[0]] = cleaned_data
                
                if symbols[0] not in self.symbols:
                    self.symbols.append(symbols[0])
                    
            else:
                # Multiple symbols
                if start and end:
                    data = yf.download(symbols, start=start, end=end, interval=interval,
                                     auto_adjust=auto_adjust, prepost=prepost, 
                                     threads=threads, group_by='ticker' if group_by_ticker else None)
                else:
                    data = yf.download(symbols, period=period, interval=interval,
                                     auto_adjust=auto_adjust, prepost=prepost,
                                     threads=threads, group_by='ticker' if group_by_ticker else None)
                
                if data.empty:
                    raise ValueError(f"No data found for symbols {symbols}")
                
                # Process multi-symbol data
                for symbol in symbols:
                    if len(symbols) > 1 and group_by_ticker:
                        symbol_data = data[symbol] if symbol in data.columns.levels[0] else pd.DataFrame()
                    else:
                        symbol_data = data
                    
                    if not symbol_data.empty:
                        cleaned_data = self._clean_data(symbol_data, symbol)
                        self.data[symbol] = cleaned_data
                        
                        if symbol not in self.symbols:
                            self.symbols.append(symbol)
                    else:
                        logging.warning(f"No data available for {symbol}")
            
            # Cache data if cache directory is set
            if self.cache_dir:
                self._cache_data(symbols, period, interval)
            
            logging.info(f"Successfully downloaded data for {len([s for s in symbols if s in self.data])} symbols")
            return {symbol: self.data[symbol] for symbol in symbols if symbol in self.data}
            
        except Exception as e:
            logging.error(f"Error downloading data: {str(e)}")
            raise

    def get_data(self, symbol: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieve data for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            columns (List[str], optional): Specific columns to return
            
        Returns:
            pd.DataFrame: Stock data for the symbol
            
        Raises:
            KeyError: If symbol not found in stored data
        """
        
        try:
            if symbol not in self.data:
                raise KeyError(f"Symbol {symbol} not found. Available symbols: {self.symbols}")
            
            data = self.data[symbol].copy()
            
            if columns:
                missing_cols = [col for col in columns if col not in data.columns]
                if missing_cols:
                    logging.warning(f"Columns {missing_cols} not found in {symbol} data")
                available_cols = [col for col in columns if col in data.columns]
                data = data[available_cols]
            
            return data
            
        except Exception as e:
            logging.error(f"Error retrieving data for {symbol}: {str(e)}")
            raise

    def get_ohlc_data(self, symbol: str) -> pd.DataFrame:
        """
        Get OHLC data in format suitable for the relative() function.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: DataFrame with date, open, high, low, close columns
        """
        
        try:
            data = self.get_data(symbol, ['open', 'high', 'low', 'close'])
            data = data.reset_index()
            data.columns = data.columns.str.lower()
            
            # Ensure date column is named 'date'
            if 'date' not in data.columns and data.index.name in ['Date', 'Datetime']:
                data = data.reset_index()
                data.rename(columns={data.columns[0]: 'date'}, inplace=True)
            
            return data
            
        except Exception as e:
            logging.error(f"Error preparing OHLC data for {symbol}: {str(e)}")
            raise

    def calculate_relative_prices(self, 
                                symbol: str, 
                                benchmark_symbol: str = '^GSPC',
                                benchmark_column: str = 'close',
                                digits: int = 4,
                                rebase: bool = True) -> pd.DataFrame:
        """
        Calculate relative prices using the relative() function.
        
        Args:
            symbol (str): Primary symbol for analysis
            benchmark_symbol (str): Benchmark symbol (default: S&P 500)
            benchmark_column (str): Column to use from benchmark data
            digits (int): Decimal places for rounding
            rebase (bool): Whether to rebase benchmark to 1.0
            
        Returns:
            pd.DataFrame: DataFrame with relative prices
            
        Raises:
            ValueError: If symbols not available or calculation fails
        """
        
        try:
            # Ensure both symbols are available
            if symbol not in self.data:
                logging.info(f"Downloading data for {symbol}")
                self.download_data(symbol)
            
            if benchmark_symbol not in self.data:
                logging.info(f"Downloading benchmark data for {benchmark_symbol}")
                self.download_data(benchmark_symbol)
            
            # Prepare data for relative calculation
            main_data = self.get_ohlc_data(symbol)
            benchmark_data = self.get_ohlc_data('FTSEMIB.MI').reset_index()[['date', 'close']]
            # benchmark_data.columns = benchmark_data.columns.str.lower()
            
            # Ensure proper column naming
            if 'date' not in benchmark_data.columns:
                benchmark_data = benchmark_data.reset_index()
                benchmark_data.rename(columns={benchmark_data.columns[0]: 'date'}, inplace=True)
            
            # Import and use the relative function (assuming it's available)
            # from your_module import relative  # Adjust import as needed
            
            result = relative(
                df=main_data,
                _o='open', _h='high', _l='low', _c='close',
                bm_df=benchmark_data,
                bm_col=benchmark_column,
                dgt=digits,
                rebase=rebase
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Error calculating relative prices: {str(e)}")
            raise

    def get_multiple_symbols_data(self, symbols: List[str], column: str = 'close') -> pd.DataFrame:
        """
        Get data for multiple symbols in a single DataFrame.
        
        Args:
            symbols (List[str]): List of symbols
            column (str): Column to extract for each symbol
            
        Returns:
            pd.DataFrame: DataFrame with dates and symbol columns
        """
        
        try:
            dfs = []
            
            for symbol in symbols:
                if symbol not in self.data:
                    logging.info(f"Downloading missing data for {symbol}")
                    self.download_data(symbol)
                
                data = self.get_data(symbol, [column]).reset_index()
                data.columns = ['date', symbol]
                dfs.append(data)
            
            # Merge all dataframes
            result = dfs[0]
            for df in dfs[1:]:
                result = pd.merge(result, df, on='date', how='outer')
            
            result = result.sort_values('date').reset_index(drop=True)
            return result
            
        except Exception as e:
            logging.error(f"Error combining multiple symbols: {str(e)}")
            raise

    def get_info(self, symbol: str) -> Dict:
        """
        Get company information for a symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Company information
        """
        
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
            
        except Exception as e:
            logging.error(f"Error getting info for {symbol}: {str(e)}")
            return {}

    def list_available_data(self) -> Dict[str, Dict]:
        """
        List all available data with summary statistics.
        
        Returns:
            Dict: Summary of available data
        """
        
        summary = {}
        
        for symbol in self.symbols:
            data = self.data[symbol]
            summary[symbol] = {
                'rows': len(data),
                'columns': list(data.columns),
                'date_range': f"{data.index.min()} to {data.index.max()}",
                'missing_values': data.isnull().sum().sum()
            }
        
        return summary


    def save_data(self, filepath: str, symbols: Optional[List[str]] = None, format: str = 'csv'):
        """
        Save financial data to a file or multiple files based on the requested format.
        
        Args:
            filepath (str): Path to save file (base path for multiple symbols)
            symbols (List[str], optional): Specific symbols to save (defaults to all self.symbols)
            format (str): File format ('csv', 'excel', 'parquet')
        """
        try:
            symbols_to_save = symbols or self.symbols
            if not symbols_to_save:
                raise ValueError("No symbols to save. Provide symbols or ensure self.symbols is populated.")

            # Validate data availability
            missing_symbols = [s for s in symbols_to_save if s not in self.data]
            if missing_symbols:
                logging.warning(f"No data found for: {missing_symbols}")
            symbols_to_save = [s for s in symbols_to_save if s in self.data]

            # Save single symbol
            if len(symbols_to_save) == 1:
                symbol = symbols_to_save[0]
                df = self.data[symbol]

                if format.lower() == 'csv':
                    df.to_csv(filepath, index=False)
                elif format.lower() == 'excel':
                    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name=symbol, index=False)
                elif format.lower() == 'parquet':
                    df.to_parquet(filepath, index=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            else:  # Save multiple symbols
                base, ext = os.path.splitext(filepath)

                if format.lower() == 'csv':
                    for symbol in symbols_to_save:
                        symbol_path = f"{base}_{symbol}.csv"
                        self.data[symbol].to_csv(symbol_path, index=False)

                elif format.lower() == 'excel':
                    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                        for symbol in symbols_to_save:
                            self.data[symbol].to_excel(writer, sheet_name=symbol, index=False)

                elif format.lower() == 'parquet':
                    for symbol in symbols_to_save:
                        symbol_path = f"{base}_{symbol}.parquet"
                        self.data[symbol].to_parquet(symbol_path, index=False)

                else:
                    raise ValueError(f"Unsupported format: {format}")

            logging.info(f"Data saved successfully to {filepath}")

        except Exception as e:
            logging.error(f"Error saving data: {str(e)}")
            raise


    def _clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and prepare downloaded data.
        
        Args:
            data (pd.DataFrame): Raw data from yfinance
            symbol (str): Symbol name for logging
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        
        # Make column names lowercase and consistent
        data.columns = data.columns.str.lower()
        
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        # Forward fill minor gaps (up to 5 days for daily data)
        if not data.empty:
            data = data.ffill()
        
        # Log data quality issues
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            logging.warning(f"{symbol}: {missing_values} missing values found")
        
        return data

    def _validate_period_interval(self, period: str, interval: str):
        """
        Validate period and interval combinations.
        
        Args:
            period (str): Time period
            interval (str): Data interval
            
        Raises:
            ValueError: If invalid combination
        """
        
        # Define valid combinations
        intraday_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
        intraday_periods = ['1d', '5d', '1mo', '3mo']
        
        if interval in intraday_intervals and period not in intraday_periods:
            raise ValueError(f"Intraday interval {interval} only supports periods: {intraday_periods}")

    def _cache_data(self, symbols: List[str], period: str, interval: str):
        """
        Cache downloaded data to disk.
        
        Args:
            symbols (List[str]): Symbols to cache
            period (str): Period used for download
            interval (str): Interval used for download
        """
        
        try:
            if not self.cache_dir:
                return
            
            for symbol in symbols:
                if symbol in self.data:
                    cache_file = self.cache_dir / f"{symbol}_{period}_{interval}.parquet"
                    self.data[symbol].to_parquet(cache_file)
                    
        except Exception as e:
            logging.warning(f"Failed to cache data: {str(e)}")

    def __repr__(self) -> str:
        return f"YFinanceDataHandler(symbols={len(self.symbols)}, cached_data={list(self.data.keys())})"

    def __len__(self) -> int:
        return len(self.symbols)