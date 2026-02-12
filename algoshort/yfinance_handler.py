import yfinance as yf
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Union, List, Dict, Optional, Tuple, Literal
import logging
from pathlib import Path

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
        logger: Class-specific logger instance
    
    Example:
        >>> handler = YFinanceDataHandler()
        >>> handler.download_data('AAPL', period='1y')
        >>> data = handler.get_data('AAPL')
    """
    
    def __init__(self, cache_dir: Optional[str] = None, enable_logging: bool = True, 
                 chunk_size: int = 50, log_level: int = logging.INFO):
        """
        Initialize the YFinanceDataHandler.
        
        Args:
            cache_dir (str, optional): Directory to cache downloaded data
            enable_logging (bool): Enable logging for operations
            chunk_size (int): Maximum symbols per download chunk (default: 50)
            log_level (int): Logging level (default: logging.INFO)
        """
        self.symbols = []
        self.data = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.chunk_size = max(1, chunk_size)  # Ensure at least 1
        
        # Setup class-specific logger (avoid duplicate handlers)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if enable_logging and not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(log_level)
            self.logger.propagate = False
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
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
                     use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Download financial data for one or more symbols with chunking and caching strategy.
        
        Args:
            symbols (str or List[str]): Stock symbol(s) to download
            period (str): Period to download
            interval (str): Data interval
            start (str, optional): Start date (YYYY-MM-DD)
            end (str, optional): End date (YYYY-MM-DD)
            auto_adjust (bool): Automatically adjust for splits and dividends
            prepost (bool): Include pre/post market data
            threads (bool): Use threading for multiple downloads
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with symbol as key and DataFrame as value
            
        Raises:
            ValueError: If invalid period/interval or no data downloaded
            TypeError: If inputs are not of expected types
        """
        
        try:
            # Input validation and preprocessing
            symbols_list = self._preprocess_symbols(symbols)
            period, interval = self._validate_and_map_params(period, interval)
            
            # Separate symbols based on cache availability
            symbols_to_download, symbols_from_cache = self._separate_cached_symbols(
                symbols_list, period, interval, use_cache
            )
            
            self.logger.info(f"Cache hits: {len(symbols_from_cache)}, Downloads needed: {len(symbols_to_download)}")
            
            # Download only symbols not in cache
            if symbols_to_download:
                self._process_downloads(
                    symbols_to_download, period, interval, start, end, 
                    auto_adjust, prepost, threads
                )
            
            # Return all successfully loaded data
            successful_symbols = [s for s in symbols_list if s in self.data]
            self.logger.info(f"Successfully processed {len(successful_symbols)}/{len(symbols_list)} symbols")
            
            return {symbol: self.data[symbol] for symbol in successful_symbols}
            
        except Exception as e:
            self.logger.error(f"Error in download_data: {str(e)}")
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
            KeyError: If symbol data not found in handler
        """
        
        try:
            if symbol not in self.data:
                available = list(self.data.keys())
                raise KeyError(
                    f"Data for symbol '{symbol}' not found in handler. "
                    f"Available symbols: {available}. "
                    f"Use download_data('{symbol}') first."
                )
            
            data = self.data[symbol].copy()
            
            if columns:
                missing_cols = [col for col in columns if col not in data.columns]
                if missing_cols:
                    self.logger.warning(f"Columns {missing_cols} not found in {symbol} data")
                available_cols = [col for col in columns if col in data.columns]
                if available_cols:
                    data = data[available_cols]
                else:
                    # Return empty DataFrame with requested columns
                    return pd.DataFrame(columns=columns)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving data for {symbol}: {str(e)}")
            raise

    def get_ohlc_data(self, symbol: str) -> pd.DataFrame:
        """
        Get OHLC data in format suitable for analysis functions.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: DataFrame with date, open, high, low, close columns
            
        Raises:
            KeyError: If symbol data not found
            ValueError: If required OHLC columns are missing
        """
        
        try:
            required_cols = ['open', 'high', 'low', 'close']
            data = self.get_data(symbol, required_cols)
            
            if data.empty:
                raise ValueError(f"No OHLC data available for {symbol}")
            
            # Reset index to get date column
            data = data.reset_index()
            
            # Ensure proper column naming
            data.columns = data.columns.str.lower()
            
            # Rename index column to 'date' if needed
            if data.columns[0].lower() in ['date', 'datetime'] or 'date' not in data.columns:
                data = data.rename(columns={data.columns[0]: 'date'})

            data = data.set_index('date')  # Fixed: assign result
            data.columns.name = None

            return data
            
        except Exception as e:
            self.logger.error(f"Error preparing OHLC data for {symbol}: {str(e)}")
            raise

    def get_combined_data(self, symbols: List[str], columns: Union[str, List[str], None] = None) -> pd.DataFrame:
            """
            Get data for multiple symbols in row-bound format (long format).
            
            This method will NOT download missing data; it only works with symbols
            that have already been loaded into memory.

            Args:
                symbols (List[str]): List of symbols
                columns (Union[str, List[str], None]): Specific column(s) to extract.
                                                    If None, returns the entire DataFrame.

            Returns:
                pd.DataFrame: DataFrame with a 'symbol' column and the requested data.
                            Returns an empty DataFrame if no data is available.
            """
            combined_rows = []

            if isinstance(columns, str):
                columns = [columns]

            for symbol in symbols:
                # Check if the symbol is in memory before proceeding
                if symbol not in self.data:
                    self.logger.warning(f"Skipping '{symbol}': data not found in memory.")
                    continue

                try:
                    data = self.get_data(symbol, columns)
                    if not data.empty:
                        symbol_data = data.copy().reset_index()
                        symbol_data.columns = symbol_data.columns.str.lower()  # Standardize to lowercase
                        symbol_data['symbol'] = symbol

                        # Ensure 'date' column is present before reordering
                        if 'date' in symbol_data.columns:
                            cols = ['date', 'symbol'] + [col for col in symbol_data.columns if col not in ['date', 'symbol']]
                            symbol_data = symbol_data[cols]
                        else:
                            self.logger.warning(f"Skipping {symbol} due to missing 'date' column.")
                            continue

                        combined_rows.append(symbol_data)
                    else:
                        self.logger.warning(f"No data available for {symbol}")

                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    continue

            if combined_rows:
                result = pd.concat(combined_rows, ignore_index=True)
                return result.sort_values(['date', 'symbol']).reset_index(drop=True)
            else:
                # Return an empty DataFrame with a consistent schema
                return pd.DataFrame(columns=['date', 'symbol'])

    def get_multiple_symbols_data(self, symbols: List[str], column: str = 'close') -> pd.DataFrame:
        """
        Get data for multiple symbols in wide format (each symbol as a column).
        
        Args:
            symbols (List[str]): List of symbols
            column (str): Column to extract for each symbol
            
        Returns:
            pd.DataFrame: DataFrame with dates and symbol columns (wide format)
            
        Raises:
            ValueError: If column is not valid or no symbols processed successfully
        """
        
        valid_columns = ['open', 'high', 'low', 'close', 'volume']
        if column not in valid_columns:
            raise ValueError(f"Column '{column}' not valid. Must be one of: {valid_columns}")
        
        try:
            dfs = []
            successful_symbols = []
            
            for symbol in symbols:
                try:
                    data = self.get_data(symbol, [column])
                    if not data.empty:
                        data_reset = data.reset_index()
                        # Use consistent column naming
                        date_col = data_reset.columns[0]
                        data_reset.columns = ['date', symbol]
                        dfs.append(data_reset)
                        successful_symbols.append(symbol)
                    else:
                        self.logger.warning(f"No data available for {symbol}")
                        
                except Exception as e:
                    self.logger.warning(f"Error processing {symbol}: {e}")
                    continue
            
            if not dfs:
                raise ValueError("No data could be retrieved for any of the specified symbols")
            
            # Merge all dataframes
            result = dfs[0]
            for df in dfs[1:]:
                result = pd.merge(result, df, on='date', how='outer')
            
            result = result.sort_values('date').reset_index(drop=True)
            self.logger.info(f"Successfully combined data for symbols: {successful_symbols}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error combining multiple symbols: {str(e)}")
            raise

    def get_info(self, symbol: str) -> Dict:
        """
        Get company information for a symbol.

        Args:
            symbol (str): Stock symbol

        Returns:
            Dict: Company information from yfinance

        Raises:
            Exception: If unable to retrieve information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info:
                self.logger.warning(f"No info available for {symbol}")
                return {}
            self.logger.info(f"Retrieved company info for {symbol}")
            return info

        except Exception as e:
            self.logger.error(f"Error getting info for {symbol}: {str(e)}")
            raise  # Re-raise instead of silent failure



    def list_available_data(self) -> Dict[str, Dict]:
        """
        List all available data with summary statistics.
        
        Returns:
            Dict: Summary of available data for each symbol
        """
        
        summary = {}
        
        for symbol in self.symbols:
            if symbol in self.data:
                data = self.data[symbol]
                
                # Ensure index is datetime for min/max operations
                try:
                    if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
                        date_min = pd.to_datetime(data.index.min()).strftime('%Y-%m-%d')
                        date_max = pd.to_datetime(data.index.max()).strftime('%Y-%m-%d')
                        date_range = f"{date_min} to {date_max}"
                    else:
                        date_range = "Invalid date range"
                except Exception:
                    date_range = "Unable to determine date range"
                
                summary[symbol] = {
                    'rows': len(data),
                    'columns': list(data.columns),
                    'date_range': date_range,
                    'missing_values': data.isnull().sum().sum()
                }
        
        return summary

    def list_cached_data(self) -> Dict[str, Dict]:
        """
        Lists all data files present in the cache directory.

        Returns:
            Dict: A dictionary with filenames as keys and file details as values.
        """
        if not self.cache_dir or not self.cache_dir.exists():
            self.logger.warning("Cache directory does not exist.")
            return {}

        cached_files = {}
        for file_path in self.cache_dir.iterdir():
            if file_path.suffix == '.parquet':
                try:
                    # Assuming the format is 'SYMBOL_PERIOD_INTERVAL.parquet'
                    parts = file_path.stem.split('_')
                    if len(parts) >= 3:
                        symbol = parts[0]
                        period = parts[1]
                        interval = parts[2]
                    else:
                        symbol = file_path.stem
                        period = 'unknown'
                        interval = 'unknown'

                    file_size_kb = file_path.stat().st_size / 1024

                    cached_files[file_path.name] = {
                        'symbol': symbol,
                        'period': period,
                        'interval': interval,
                        'size_kb': round(file_size_kb, 2),
                        'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                    }
                except Exception as e:
                    self.logger.error(f"Failed to process cached file {file_path.name}: {e}")
                    
        return cached_files

    def list_cached_symbols(self) -> List[str]:
            """
            Scans the cache directory and returns a list of all unique cached symbols.

            Returns:
                list[str]: A list of unique symbols found in the cache.
            """
            if not self.cache_dir or not self.cache_dir.exists():
                self.logger.warning("Cache directory does not exist or is not configured.")
                return []

            symbols_in_cache = set()
            for file_path in self.cache_dir.iterdir():
                if file_path.suffix == '.parquet':
                    try:
                        # Assuming the filename format is "SYMBOL_PERIOD_INTERVAL.parquet"
                        symbol = file_path.stem.split('_')[0]
                        symbols_in_cache.add(symbol)
                    except IndexError:
                        self.logger.warning(f"Skipping malformed filename in cache: {file_path.name}")
            return sorted(list(symbols_in_cache))
    
    def save_data(self, 
                  filepath: str, 
                  symbols: Optional[List[str]] = None, 
                  format: str = 'csv',
                  multi_symbol_strategy: Literal['separate_files', 'single_file', 'excel_sheets'] = 'separate_files',
                  combine_column: str = 'close') -> None:
        """
        Save data to file with flexible options for multiple symbols.
        
        Args:
            filepath (str): Base path for saving files
            symbols (List[str], optional): Specific symbols to save (default: all)
            format (str): File format ('csv', 'excel', 'parquet')
            multi_symbol_strategy (str): How to handle multiple symbols:
                - 'separate_files': Create separate file for each symbol
                - 'single_file': Combine all symbols in one file (row-bound format)
                - 'excel_sheets': Save as Excel with separate sheets per symbol
            combine_column (str): Column to use when combining symbols in single file
            
        Raises:
            ValueError: If invalid format or strategy specified
            FileNotFoundError: If unable to create output directory
        """
        
        valid_formats = ['csv', 'excel', 'parquet']
        valid_strategies = ['separate_files', 'single_file', 'excel_sheets']
        
        if format not in valid_formats:
            raise ValueError(f"Format must be one of: {valid_formats}")
        
        if multi_symbol_strategy not in valid_strategies:
            raise ValueError(f"Multi-symbol strategy must be one of: {valid_strategies}")
        
        try:
            symbols_to_save = symbols or self.symbols
            symbols_to_save = [s for s in symbols_to_save if s in self.data]
            
            if not symbols_to_save:
                raise ValueError("No symbols with data available to save")
            
            # Create output directory if needed
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if len(symbols_to_save) == 1:
                # Single symbol - straightforward save
                symbol = symbols_to_save[0]
                data = self.data[symbol]
                self._save_single_file(data, filepath, format)
                self.logger.info(f"Saved data for {symbol} to {filepath}")
                
            else:
                # Multiple symbols - use specified strategy
                if multi_symbol_strategy == 'separate_files':
                    self._save_separate_files(symbols_to_save, filepath, format)
                    
                elif multi_symbol_strategy == 'single_file':
                    combined_data = self.get_combined_data(symbols_to_save, combine_column)
                    self._save_single_file(combined_data, filepath, format)
                    self.logger.info(f"Saved combined data for {len(symbols_to_save)} symbols to {filepath}")
                    
                elif multi_symbol_strategy == 'excel_sheets':
                    if format != 'excel':
                        self.logger.warning("Forcing format to 'excel' for excel_sheets strategy")
                        format = 'excel'
                    self._save_excel_sheets(symbols_to_save, filepath)
                    
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise

    def clear_cache(self, symbols: Optional[List[str]] = None) -> int:
        """
        Clear cached files for specified symbols or all symbols.
        
        Args:
            symbols (List[str], optional): Symbols to clear from cache (default: all)
            
        Returns:
            int: Number of files removed
            
        Raises:
            OSError: If unable to remove cache files
        """
        
        if not self.cache_dir or not self.cache_dir.exists():
            self.logger.warning("No cache directory found")
            return 0
        
        try:
            removed_count = 0
            
            if symbols is None:
                # Remove all cache files
                for cache_file in self.cache_dir.glob("*.parquet"):
                    cache_file.unlink()
                    removed_count += 1
                    self.logger.debug(f"Removed cache file: {cache_file.name}")
            else:
                # Remove specific symbols
                for symbol in symbols:
                    pattern = f"{symbol}_*.parquet"
                    for cache_file in self.cache_dir.glob(pattern):
                        cache_file.unlink()
                        removed_count += 1
                        self.logger.debug(f"Removed cache file: {cache_file.name}")
            
            self.logger.info(f"Cleared {removed_count} cache files")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            raise

    # Private helper methods
    
    def _preprocess_symbols(self, symbols: Union[str, List[str]]) -> List[str]:
        """
        Preprocess and validate symbol inputs.

        Args:
            symbols: Single symbol string or list of symbol strings.

        Returns:
            List of validated, uppercase symbol strings.

        Raises:
            TypeError: If symbols is not a string or list of strings.
            ValueError: If symbol format is invalid or empty.
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        elif not isinstance(symbols, list):
            raise TypeError("Symbols must be a string or list of strings")

        validated = []
        for s in symbols:
            if not isinstance(s, str):
                raise TypeError(f"Symbol must be string, got {type(s).__name__}")
            cleaned = s.strip().upper()
            if not cleaned:
                raise ValueError("Empty symbol not allowed")
            # Only allow valid ticker characters (letters, numbers, ., -, ^, =)
            if not re.match(r'^[A-Z0-9.\-\^=]+$', cleaned):
                raise ValueError(f"Invalid symbol format: '{s}'")
            validated.append(cleaned)

        if not validated:
            raise ValueError("No valid symbols provided")
        return validated

    def _validate_and_map_params(self, period: str, interval: str) -> Tuple[str, str]:
        """Validate and map period/interval parameters."""
        period = self.period_map.get(period, period)
        interval = self.interval_map.get(interval, interval)

        self._validate_period_interval(period, interval)
        return period, interval

    def _get_safe_cache_path(self, symbol: str, period: str, interval: str) -> Path:
        """
        Generate safe cache file path, preventing path traversal attacks.

        Args:
            symbol: Stock symbol (will be sanitized).
            period: Data period.
            interval: Data interval.

        Returns:
            Safe Path object within cache_dir.

        Raises:
            ValueError: If sanitized path would escape cache_dir.
        """
        # Sanitize all components to prevent path traversal
        safe_symbol = re.sub(r'[^A-Z0-9_\-]', '_', symbol.upper())
        safe_period = re.sub(r'[^a-z0-9]', '_', period.lower())
        safe_interval = re.sub(r'[^a-z0-9]', '_', interval.lower())

        cache_file = self.cache_dir / f"{safe_symbol}_{safe_period}_{safe_interval}.parquet"

        # Verify resolved path stays within cache_dir
        try:
            resolved = cache_file.resolve()
            cache_resolved = self.cache_dir.resolve()
            if not str(resolved).startswith(str(cache_resolved)):
                raise ValueError(f"Invalid symbol causes path traversal: {symbol}")
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid cache path for symbol {symbol}: {e}")

        return cache_file

    def _separate_cached_symbols(self, symbols: List[str], period: str, interval: str, 
                                use_cache: bool) -> Tuple[List[str], List[str]]:
        """Separate symbols into those needing download vs those available in cache."""
        symbols_to_download = []
        symbols_from_cache = []
        
        if use_cache and self.cache_dir:
            for symbol in symbols:
                cached_data = self._load_from_cache(symbol, period, interval)
                if cached_data is not None:
                    cleaned_data = self._clean_data(cached_data, symbol)
                    self.data[symbol] = cleaned_data
                    if symbol not in self.symbols:
                        self.symbols.append(symbol)
                    symbols_from_cache.append(symbol)
                else:
                    symbols_to_download.append(symbol)
        else:
            symbols_to_download = symbols
        
        return symbols_to_download, symbols_from_cache

    def _process_downloads(self, symbols: List[str], period: str, interval: str,
                          start: Optional[str], end: Optional[str], auto_adjust: bool,
                          prepost: bool, threads: bool) -> None:
        """Process downloads using chunking strategy."""
        chunks = self._create_chunks(symbols)
        total_chunks = len(chunks)
        
        self.logger.info(f"Downloading {len(symbols)} symbols in {total_chunks} chunks of {self.chunk_size}")
        
        for chunk_idx, chunk in enumerate(chunks, 1):
            self.logger.debug(f"Processing chunk {chunk_idx}/{total_chunks}: {chunk}")
            
            try:
                if len(chunk) == 1:
                    self._download_single_symbol(
                        chunk[0], period, interval, start, end, auto_adjust, prepost
                    )
                else:
                    self._download_multiple_symbols(
                        chunk, period, interval, start, end, auto_adjust, prepost, threads
                    )
                    
            except Exception as e:
                self.logger.error(f"Error downloading chunk {chunk_idx}: {e}")
                continue

    def _create_chunks(self, symbols: List[str]) -> List[List[str]]:
        """Split symbols into chunks for efficient downloading."""
        return [symbols[i:i + self.chunk_size] for i in range(0, len(symbols), self.chunk_size)]

    def _load_from_cache(self, symbol: str, period: str, interval: str,
                         max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available and not stale.

        Args:
            symbol: Stock symbol.
            period: Data period.
            interval: Data interval.
            max_age_hours: Maximum cache age in hours (default 24).

        Returns:
            Cached DataFrame or None if not available/stale.
        """
        if not self.cache_dir:
            return None

        try:
            cache_file = self._get_safe_cache_path(symbol, period, interval)
        except ValueError as e:
            self.logger.warning(f"Invalid cache path for {symbol}: {e}")
            return None

        if cache_file.exists():
            try:
                # Check cache age
                file_age_seconds = datetime.now().timestamp() - cache_file.stat().st_mtime
                if file_age_seconds > max_age_hours * 3600:
                    self.logger.debug(f"Cache for {symbol} is stale ({file_age_seconds/3600:.1f}h old)")
                    return None

                cached_data = pd.read_parquet(cache_file)
                self.logger.debug(f"Loaded {symbol} from cache")
                return cached_data
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {symbol}: {e}")

        return None

    def _download_single_symbol(self, symbol: str, period: str, interval: str,
                               start: Optional[str], end: Optional[str], 
                               auto_adjust: bool, prepost: bool) -> None:
        """Download data for a single symbol."""
        ticker = yf.Ticker(symbol)
        
        if start and end:
            data = ticker.history(start=start, end=end, interval=interval, 
                                auto_adjust=auto_adjust, prepost=prepost)
        else:
            data = ticker.history(period=period, interval=interval,
                                auto_adjust=auto_adjust, prepost=prepost)
        
        if data.empty:
            self.logger.warning(f"No data found for symbol {symbol}")
            return
        
        # Clean, store, and cache data
        cleaned_data = self._clean_data(data, symbol)
        self.data[symbol] = cleaned_data
        
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        
        if self.cache_dir:
            self._cache_single_symbol(symbol, cleaned_data, period, interval)

    def _download_multiple_symbols(self, symbols: List[str], period: str, interval: str,
                                  start: Optional[str], end: Optional[str], 
                                  auto_adjust: bool, prepost: bool, threads: bool) -> None:
        """Download data for multiple symbols."""
        if start and end:
            data = yf.download(symbols, start=start, end=end, interval=interval,
                             auto_adjust=auto_adjust, prepost=prepost, 
                             threads=threads, group_by='ticker')
        else:
            data = yf.download(symbols, period=period, interval=interval,
                             auto_adjust=auto_adjust, prepost=prepost,
                             threads=threads, group_by='ticker')
        
        if data.empty:
            self.logger.warning(f"No data found for symbols {symbols}")
            return
        
        # Process each symbol
        for symbol in symbols:
            try:
                symbol_data = self._extract_symbol_data(data, symbol, len(symbols) > 1)
                
                if not symbol_data.empty:
                    cleaned_data = self._clean_data(symbol_data, symbol)
                    self.data[symbol] = cleaned_data
                    
                    if symbol not in self.symbols:
                        self.symbols.append(symbol)
                    
                    if self.cache_dir:
                        self._cache_single_symbol(symbol, cleaned_data, period, interval)
                else:
                    self.logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")

    def _extract_symbol_data(self, data: pd.DataFrame, symbol: str, multi_symbol: bool) -> pd.DataFrame:
        """Extract data for a specific symbol from multi-symbol download."""
        if multi_symbol:
            if hasattr(data.columns, 'levels') and symbol in data.columns.levels[0]:
                return data[symbol]
            else:
                # Handle case where only one symbol returned data
                return data
        else:
            return data

    def _clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and prepare downloaded data."""
        # Make column names lowercase and consistent
        data.columns = data.columns.str.lower()
        
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        # Forward fill minor gaps (up to 5 days for daily data)
        if not data.empty:
            data = data.ffill(limit=5)
        
        # Log data quality issues
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            self.logger.warning(f"{symbol}: {missing_values} missing values found")
        
        return data

    def _cache_single_symbol(self, symbol: str, data: pd.DataFrame, period: str, interval: str) -> None:
        """
        Cache data for a single symbol with atomic write.

        Args:
            symbol: Stock symbol.
            data: DataFrame to cache.
            period: Data period.
            interval: Data interval.
        """
        try:
            cache_file = self._get_safe_cache_path(symbol, period, interval)
            # Write to temp file first, then atomic rename
            temp_file = cache_file.with_suffix('.tmp')
            data.to_parquet(temp_file)
            temp_file.replace(cache_file)  # Atomic rename
            self.logger.debug(f"Cached data for {symbol}")
        except Exception as e:
            self.logger.warning(f"Failed to cache {symbol}: {e}")

    def _validate_period_interval(self, period: str, interval: str) -> None:
        """Validate period and interval combinations."""
        intraday_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
        intraday_periods = ['1d', '5d', '1mo', '3mo']
        
        if interval in intraday_intervals and period not in intraday_periods:
            raise ValueError(f"Intraday interval {interval} only supports periods: {intraday_periods}")

    def _save_single_file(self, data: pd.DataFrame, filepath: str, format: str) -> None:
        """Save data to a single file."""
        if format == 'csv':
            data.to_csv(filepath)
        elif format == 'excel':
            data.to_excel(filepath, engine='openpyxl')
        elif format == 'parquet':
            data.to_parquet(filepath)

    def _save_separate_files(self, symbols: List[str], filepath: str, format: str) -> None:
        """Save each symbol to a separate file."""
        base_path = Path(filepath)
        base_name = base_path.stem
        extension = f".{format}" if format != 'excel' else '.xlsx'
        
        for symbol in symbols:
            if symbol in self.data:
                symbol_path = base_path.parent / f"{base_name}_{symbol}{extension}"
                self._save_single_file(self.data[symbol], str(symbol_path), format)
                self.logger.debug(f"Saved {symbol} to {symbol_path}")
        
        self.logger.info(f"Saved {len(symbols)} symbols to separate files")

    def _save_excel_sheets(self, symbols: List[str], filepath: str) -> None:
        """Save symbols as separate sheets in an Excel file."""
        if not filepath.endswith('.xlsx'):
            filepath = filepath.replace(Path(filepath).suffix, '.xlsx')
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for symbol in symbols:
                if symbol in self.data:
                    # Excel sheet names have character limits and restrictions
                    sheet_name = symbol.replace(':', '_').replace('/', '_')[:31]
                    self.data[symbol].to_excel(writer, sheet_name=sheet_name)
                    self.logger.debug(f"Added {symbol} as sheet '{sheet_name}'")
        
        self.logger.info(f"Saved {len(symbols)} symbols as Excel sheets to {filepath}")

    def __repr__(self) -> str:
        return f"YFinanceDataHandler(symbols={len(self.symbols)}, cached_data={list(self.data.keys())})"

    def __len__(self) -> int:
        return len(self.symbols)