# YFinanceDataHandler API Documentation

## Overview

`YFinanceDataHandler` is a production-grade data management class for downloading, caching, and preparing financial market data using the yfinance library. It abstracts away complexity around multi-symbol downloads, intelligent caching, data quality management, and format conversions—enabling data scientists and quantitative analysts to focus on analysis rather than data plumbing.

**Primary use case**: Financial data science workflows requiring efficient, reliable access to historical and recent market data across multiple symbols with minimal latency.

**Core benefits**: 
- Automatic chunking for bulk downloads (avoids rate limits)
- Intelligent caching with parquet files (10-50x faster repeated access)
- Built-in data cleaning and quality checks
- Flexible output formats for different analysis needs (wide vs. long format, OHLC format)

---

## Class: YFinanceDataHandler

### Signature

```python
class YFinanceDataHandler:
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        enable_logging: bool = True,
        chunk_size: int = 50,
        log_level: int = logging.INFO
    )
```

### Constructor Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `cache_dir` | `Optional[str]` | No | `None` | Directory path for caching downloaded data as parquet files. If `None`, no caching occurs. Directory is created automatically if it doesn't exist. |
| `enable_logging` | `bool` | No | `True` | Enable structured logging for download operations, cache hits/misses, and errors. Logs to stdout with timestamps. |
| `chunk_size` | `int` | No | `50` | Maximum symbols per download batch. Higher values increase speed but may hit rate limits. Minimum enforced value is 1. Recommended: 20-50 for most use cases. |
| `log_level` | `int` | No | `logging.INFO` | Python logging level constant (e.g., `logging.DEBUG`, `logging.WARNING`). Controls verbosity of log output. |

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `symbols` | `List[str]` | List of all symbols that have been downloaded or attempted. Updated automatically. |
| `data` | `Dict[str, pd.DataFrame]` | In-memory storage of downloaded data. Keys are uppercase ticker symbols, values are DataFrames with datetime index. |
| `period_map` | `Dict[str, str]` | Maps user-friendly period names (e.g., `'year'`) to yfinance period codes (e.g., `'1y'`). See Period Mappings section below. |
| `interval_map` | `Dict[str, str]` | Maps user-friendly interval names (e.g., `'daily'`) to yfinance interval codes (e.g., `'1d'`). See Interval Mappings section below. |
| `logger` | `logging.Logger` | Class-specific logger instance for debugging and monitoring. |
| `cache_dir` | `Optional[Path]` | Path object for cache directory if caching is enabled. |

### Period & Interval Mappings

**Period aliases** (use either format):
- Short: `'1d'`, `'5d'`, `'1mo'`, `'3mo'`, `'6mo'`, `'1y'`, `'2y'`, `'5y'`, `'10y'`, `'ytd'`, `'max'`
- Friendly: `'day'`, `'week'`, `'month'`, `'quarter'`, `'half_year'`, `'year'`, `'two_years'`, `'five_years'`, `'ten_years'`, `'all_time'`

**Interval aliases** (use either format):
- Short: `'1m'`, `'2m'`, `'5m'`, `'15m'`, `'30m'`, `'1h'`, `'1d'`, `'1wk'`, `'1mo'`
- Friendly: `'minute'`, `'hourly'`, `'daily'`, `'weekly'`, `'monthly'`

**⚠️ Intraday interval restrictions**: Intervals `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h` only support periods `1d`, `5d`, `1mo`, `3mo`. Longer periods will raise `ValueError`.

---

## Method: download_data()

### Overview

Downloads financial market data for one or more symbols with intelligent caching and chunking. This is the primary data ingestion method. Automatically handles bulk downloads by splitting large symbol lists into chunks, checks cache before downloading, and stores cleaned data in memory.

**When to use**: Initial data loading, adding new symbols, or forcing fresh downloads when cache is stale.

### Signature

```python
def download_data(
    self,
    symbols: Union[str, List[str]],
    period: str = '1y',
    interval: str = '1d',
    start: Optional[str] = None,
    end: Optional[str] = None,
    auto_adjust: bool = True,
    prepost: bool = False,
    threads: bool = True,
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `symbols` | `Union[str, List[str]]` | Yes | — | Single ticker symbol (`'AAPL'`) or list of symbols (`['AAPL', 'MSFT', 'GOOGL']`). Automatically converted to uppercase. |
| `period` | `str` | No | `'1y'` | Time period to download. Use period codes (e.g., `'1y'`, `'5y'`) or friendly names (e.g., `'year'`). Ignored if `start` and `end` are provided. |
| `interval` | `str` | No | `'1d'` | Data granularity. Common: `'1d'` (daily), `'1h'` (hourly), `'1m'` (minute). See Interval Mappings above. ⚠️ Intraday intervals have period restrictions. |
| `start` | `Optional[str]` | No | `None` | Start date in `'YYYY-MM-DD'` format. If provided with `end`, overrides `period` parameter. |
| `end` | `Optional[str]` | No | `None` | End date in `'YYYY-MM-DD'` format. Must be used with `start`. |
| `auto_adjust` | `bool` | No | `True` | Automatically adjust prices for stock splits and dividends. Recommended `True` for most analysis to ensure data continuity. |
| `prepost` | `bool` | No | `False` | Include pre-market and post-market trading data. Only relevant for intraday intervals. |
| `threads` | `bool` | No | `True` | Enable multi-threading for parallel downloads when downloading multiple symbols. Improves performance significantly for bulk downloads. |
| `use_cache` | `bool` | No | `True` | Check cache before downloading. Set to `False` to force fresh download and update cache. |

### Returns

- **Type**: `Dict[str, pd.DataFrame]`
- **Description**: Dictionary mapping each successfully downloaded symbol (uppercase string) to its corresponding DataFrame. Keys are ticker symbols, values are DataFrames with datetime index.
- **DataFrame structure**: Index is datetime. Columns (lowercase): `'open'`, `'high'`, `'low'`, `'close'`, `'volume'`, potentially `'dividends'` and `'stock splits'` depending on `auto_adjust` setting.
- **Possible values**: 
  - Dictionary with 1+ entries if any downloads succeeded
  - Empty dictionary if all downloads failed (check logs for details)
- **Throws**: 
  - `ValueError` – Invalid period/interval combination, no data downloaded for any symbol
  - `TypeError` – `symbols` parameter is not string or list of strings

### Usage Examples

#### Basic Usage

```python
from yfinance_handler import YFinanceDataHandler

# Initialize handler with caching
handler = YFinanceDataHandler(cache_dir='./market_data_cache')

# Download 1 year of daily data for Apple
data = handler.download_data('AAPL', period='1y')
# => {'AAPL': DataFrame with ~252 rows (trading days)}

# Access the DataFrame
aapl_df = data['AAPL']
print(aapl_df.head())
# =>             open    high     low   close      volume
#    2024-02-06  185.2  186.45  184.13  185.92  58472100
```

#### Common Usage

```python
# Download multiple tech stocks with 5 years of weekly data
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
data = handler.download_data(
    symbols=symbols,
    period='5y',
    interval='1wk',
    use_cache=True
)

# Check which symbols downloaded successfully
print(f"Successfully downloaded: {list(data.keys())}")
# => Successfully downloaded: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Download specific date range (overrides period)
spy_2023 = handler.download_data(
    symbols='SPY',
    start='2023-01-01',
    end='2023-12-31',
    interval='1d'
)
```

#### Advanced Usage

```python
# Download 100+ symbols efficiently with custom chunk size
import logging

handler = YFinanceDataHandler(
    cache_dir='./data_cache',
    chunk_size=25,  # Download 25 symbols at a time
    log_level=logging.DEBUG  # Verbose logging for monitoring
)

# S&P 500 tickers (example subset)
sp500_symbols = ['AAPL', 'MSFT', 'GOOGL', ...] # 100+ symbols

# Force fresh download, ignoring cache
data = handler.download_data(
    symbols=sp500_symbols,
    period='2y',
    interval='1d',
    use_cache=False,  # Force fresh download
    threads=True,     # Parallel downloads
    auto_adjust=True  # Adjust for corporate actions
)

# Handle partial failures gracefully
successful_count = len(data)
failed_symbols = set(sp500_symbols) - set(data.keys())
print(f"Downloaded {successful_count}/{len(sp500_symbols)} symbols")
if failed_symbols:
    print(f"Failed: {failed_symbols}")
    # Retry failed symbols individually for debugging
    for symbol in failed_symbols:
        try:
            handler.download_data(symbol, period='2y')
        except Exception as e:
            print(f"{symbol} error: {e}")
```

### Common Pitfalls & Edge Cases

- **Intraday data restrictions**: ⚠️ Attempting to download `period='5y'` with `interval='1m'` will raise `ValueError`. Intraday intervals (`1m`, `5m`, `1h`, etc.) only support short periods (`1d`, `5d`, `1mo`, `3mo`).

- **Symbol case sensitivity**: Symbols are automatically converted to uppercase. `'aapl'` becomes `'AAPL'` internally.

- **Cache staleness**: Using `use_cache=True` (default) returns cached data even if it's weeks old. For time-sensitive analysis, periodically set `use_cache=False` or use `clear_cache()` first.

- **Memory usage**: Downloading 500+ symbols with `period='max'` can consume several GB of RAM. Monitor memory if working with large symbol lists or long periods. Consider processing in smaller batches.

- **Partial failures are silent**: If downloading 10 symbols and 2 fail, the method returns data for the 8 successful ones without raising an error. Always check `len(data)` vs. `len(symbols)` to detect partial failures. Check logs for details on failures.

- **Rate limiting**: Yahoo Finance may rate-limit aggressive requests. If downloads fail intermittently, reduce `chunk_size` (e.g., to 10-20) or add delays between calls.

- **Data quality**: Not all symbols return clean data. Use `list_available_data()` to inspect completeness. Missing values are forward-filled up to 5 periods automatically.

### Best Practices & Anti-Patterns

| ✅ Do this | ❌ Don't do this | Why |
|-----------|------------------|-----|
| Use caching for repeated analysis | Download fresh data every script run | Caching reduces API load and improves performance 10-50x |
| Check returned dictionary length vs. input symbols | Assume all symbols downloaded successfully | Partial failures are common with invalid/delisted tickers |
| Use `period='max'` only when truly needed | Always download maximum history | Wastes bandwidth and memory; most analysis needs 1-5 years |
| Set `chunk_size=20-30` for bulk downloads | Use default `chunk_size=50` for 200+ symbols | Lower chunk size reduces rate limit errors |
| Enable logging in production | Disable logging to "improve performance" | Logging overhead is negligible; diagnostics are invaluable |
| Store cache outside project directory | Cache in `/tmp` or project root | Caches in `/tmp` are lost on reboot; version control excludes cache dirs |

### Related / See Also

- `get_data()` – Retrieve previously downloaded data for a symbol without re-downloading
- `clear_cache()` – Remove stale cached data before downloading fresh data
- `list_available_data()` – Inspect what data is currently in memory
- `get_combined_data()` – Merge multiple symbols into a single DataFrame for cross-sectional analysis

---

## Method: get_data()

### Overview

Retrieves previously downloaded data for a specific symbol from the in-memory cache. This is the primary method for accessing data after calling `download_data()`. Optionally filters to specific columns for memory efficiency.

**When to use**: Accessing data that's already been downloaded, extracting specific columns for analysis.

### Signature

```python
def get_data(
    self,
    symbol: str,
    columns: Optional[List[str]] = None
) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `symbol` | `str` | Yes | — | Ticker symbol to retrieve. Must have been previously downloaded via `download_data()`. Case-insensitive (converted to uppercase internally). |
| `columns` | `Optional[List[str]]` | No | `None` | Specific column names to return (e.g., `['close', 'volume']`). If `None`, returns all columns. Missing columns trigger a warning but don't raise errors. |

### Returns

- **Type**: `pd.DataFrame`
- **Description**: Copy of the stored DataFrame for the specified symbol with datetime index
- **Possible values**: 
  - DataFrame with requested columns if symbol exists and columns are valid
  - DataFrame with subset of requested columns if some columns are missing (warns in log)
  - Empty DataFrame with requested column names if symbol exists but all requested columns are missing
- **Throws**: 
  - `KeyError` – Symbol not found in handler. Error message lists available symbols and suggests calling `download_data()` first.

### Usage Examples

#### Basic Usage

```python
# Retrieve all data for a symbol
aapl_data = handler.get_data('AAPL')
print(aapl_data.columns)
# => Index(['open', 'high', 'low', 'close', 'volume'], dtype='object')

# Get specific columns only
close_prices = handler.get_data('AAPL', columns=['close'])
# => DataFrame with single 'close' column
```

#### Common Usage

```python
# Extract closing prices and volume for multiple symbols
symbols = ['AAPL', 'MSFT', 'GOOGL']
price_volume_data = {}

for symbol in symbols:
    price_volume_data[symbol] = handler.get_data(
        symbol, 
        columns=['close', 'volume']
    )

# Calculate returns using close prices
aapl_close = handler.get_data('AAPL', columns=['close'])
aapl_returns = aapl_close.pct_change().dropna()
```

#### Advanced Usage

```python
# Robust data retrieval with error handling
def safe_get_data(handler, symbol, columns=None):
    """
    Safely retrieve data with fallback to download if missing.
    """
    try:
        return handler.get_data(symbol, columns=columns)
    except KeyError:
        print(f"{symbol} not in memory, downloading...")
        handler.download_data(symbol, period='1y')
        return handler.get_data(symbol, columns=columns)

# Use in analysis pipeline
data = safe_get_data(handler, 'TSLA', columns=['close', 'volume'])

# Handle missing columns gracefully
requested_cols = ['close', 'volume', 'adjusted_close']  # 'adjusted_close' doesn't exist
data = handler.get_data('AAPL', columns=requested_cols)
# => Returns DataFrame with 'close' and 'volume' only, logs warning about 'adjusted_close'
```

### Common Pitfalls & Edge Cases

- **Symbol not downloaded**: Calling `get_data('NVDA')` without first calling `download_data('NVDA')` raises `KeyError`. Always download data before accessing it, or use a wrapper function with fallback logic.

- **Column name case sensitivity**: Column names are lowercase (`'close'`, not `'Close'`). Requesting `columns=['Close']` will log a warning and return an empty DataFrame or subset of available columns.

- **Mutation risk**: ⚠️ Although `get_data()` returns a copy (not a view), modifying the returned DataFrame and reassigning it back to `handler.data[symbol]` will persist changes. This can cause data integrity issues in long-running scripts.

- **Missing columns don't raise errors**: Requesting non-existent columns logs a warning but doesn't fail. Always check returned DataFrame columns if you need specific fields guaranteed.

- **Memory efficiency**: Requesting `columns=['close']` instead of retrieving all columns reduces memory usage significantly for large datasets. For 100 symbols with 10 years of data, this can save several hundred MB.

### Best Practices & Anti-Patterns

| ✅ Do this | ❌ Don't do this | Why |
|-----------|------------------|-----|
| Request specific columns when you only need a few | Always retrieve all columns | Reduces memory footprint by 80% when you only need close prices |
| Check if symbol exists before calling `get_data()` | Wrap every call in try/except | Use `if symbol in handler.data` for cleaner code |
| Use lowercase column names | Mix case (e.g., `'Close'` vs `'close'`) | Column names are normalized to lowercase after download |
| Work with copies when experimenting | Modify returned DataFrame in-place assuming isolation | Although copies are returned, reassigning to `handler.data` persists changes |

### Related / See Also

- `download_data()` – Download data before retrieving it
- `get_ohlc_data()` – Get data in OHLC format specifically for analysis functions
- `get_combined_data()` – Get data for multiple symbols in a single DataFrame
- `list_available_data()` – Check which symbols are currently in memory

---

## Method: get_ohlc_data()

### Overview

Retrieves data formatted specifically for OHLC (Open-High-Low-Close) analysis functions. Returns a DataFrame with standardized column names (`date`, `open`, `high`, `low`, `close`) and datetime index, ready for technical indicator calculations or charting libraries.

**When to use**: Feeding data into technical analysis libraries (e.g., TA-Lib, pandas_ta), charting tools, or custom OHLC-based algorithms.

### Signature

```python
def get_ohlc_data(self, symbol: str) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `symbol` | `str` | Yes | — | Ticker symbol to retrieve. Must have been previously downloaded. Case-insensitive. |

### Returns

- **Type**: `pd.DataFrame`
- **Description**: DataFrame with `date` index and lowercase OHLC columns: `open`, `high`, `low`, `close`
- **Possible values**: DataFrame with 4 columns and datetime index if data exists
- **Throws**: 
  - `KeyError` – Symbol not found in handler
  - `ValueError` – Symbol exists but OHLC columns are missing (e.g., symbol has dividends data only)

### Usage Examples

#### Basic Usage

```python
# Get OHLC data for technical analysis
ohlc = handler.get_ohlc_data('AAPL')
print(ohlc.head())
# =>             open    high     low   close
#    date                                     
#    2024-02-06  185.2  186.45  184.13  185.92
#    2024-02-07  186.1  187.20  185.50  186.75
```

#### Common Usage

```python
# Use with technical analysis library
import pandas_ta as ta

ohlc = handler.get_ohlc_data('SPY')

# Calculate RSI
ohlc['rsi'] = ta.rsi(ohlc['close'], length=14)

# Calculate Bollinger Bands
bbands = ta.bbands(ohlc['close'], length=20)
ohlc = pd.concat([ohlc, bbands], axis=1)

# Calculate moving averages
ohlc['sma_50'] = ta.sma(ohlc['close'], length=50)
ohlc['sma_200'] = ta.sma(ohlc['close'], length=200)
```

#### Advanced Usage

```python
# Build multi-symbol technical indicator pipeline
import pandas_ta as ta

def calculate_signals(handler, symbols):
    """
    Calculate buy/sell signals for multiple symbols.
    """
    signals = {}
    
    for symbol in symbols:
        try:
            ohlc = handler.get_ohlc_data(symbol)
            
            # Calculate indicators
            ohlc['rsi'] = ta.rsi(ohlc['close'], length=14)
            ohlc['macd'] = ta.macd(ohlc['close'])['MACD_12_26_9']
            ohlc['signal'] = ta.macd(ohlc['close'])['MACDs_12_26_9']
            
            # Generate trading signals
            ohlc['buy_signal'] = (
                (ohlc['rsi'] < 30) & 
                (ohlc['macd'] > ohlc['signal'])
            )
            ohlc['sell_signal'] = (
                (ohlc['rsi'] > 70) & 
                (ohlc['macd'] < ohlc['signal'])
            )
            
            signals[symbol] = ohlc
            
        except (KeyError, ValueError) as e:
            print(f"Skipping {symbol}: {e}")
            continue
    
    return signals

# Use in backtesting
symbols = ['AAPL', 'MSFT', 'GOOGL']
signal_data = calculate_signals(handler, symbols)
```

### Common Pitfalls & Edge Cases

- **Missing OHLC columns**: If a symbol's data doesn't include all four OHLC columns (rare, but possible for some special securities), `ValueError` is raised. This is intentional to prevent silent failures in technical analysis.

- **Volume column not included**: `get_ohlc_data()` returns only OHLC columns, not volume. If you need volume, use `get_data(symbol, columns=['open', 'high', 'low', 'close', 'volume'])` instead.

- **Date column vs. index**: The returned DataFrame has dates as the index AND includes a `date` column. Some libraries expect index-only dates. Use `df.drop('date', axis=1)` or `df.set_index('date', drop=True)` if needed.

- **Column name case**: Returned columns are lowercase. If your analysis library expects uppercase (`'Close'`, `'High'`), use `df.columns = df.columns.str.upper()` to convert.

### Best Practices & Anti-Patterns

| ✅ Do this | ❌ Don't do this | Why |
|-----------|------------------|-----|
| Use `get_ohlc_data()` for technical analysis | Use `get_data()` and manually select columns | Ensures consistent column naming and structure |
| Verify columns exist before complex calculations | Assume OHLC data is always complete | Some symbols may have partial data or corporate actions that disrupt OHLC structure |
| Reset or drop the `date` column if your library doesn't need it | Keep duplicate date information (index + column) | Some libraries get confused by duplicate date representations |

### Related / See Also

- `get_data()` – Get raw data with all available columns including volume
- `get_combined_data()` – Get OHLC data for multiple symbols in long format
- pandas_ta library for technical indicators
- TA-Lib for additional technical analysis functions

---

## Method: get_combined_data()

### Overview

Retrieves data for multiple symbols in a single DataFrame with row-bound (long) format, where each row is tagged with its symbol. This format is ideal for cross-sectional analysis, panel regressions, or visualization libraries that expect long-format data (e.g., Plotly Express, Seaborn).

**Critical constraint**: ⚠️ This method does NOT download missing data. It only works with symbols already in memory via `download_data()`.

### Signature

```python
def get_combined_data(
    self,
    symbols: List[str],
    columns: Union[str, List[str], None] = None
) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `symbols` | `List[str]` | Yes | — | List of ticker symbols to combine. Symbols not in memory are skipped with a warning logged. |
| `columns` | `Union[str, List[str], None]` | No | `None` | Column name(s) to extract from each symbol's data. Can be a single string (`'close'`) or list (`['close', 'volume']`). If `None`, includes all available columns. |

### Returns

- **Type**: `pd.DataFrame`
- **Description**: Row-bound DataFrame with `Date`, `symbol` columns plus requested data columns, sorted by date then symbol
- **Possible values**: 
  - DataFrame with combined data from all available symbols
  - Empty DataFrame with schema `['Date', 'symbol']` if no symbols are in memory or all specified symbols are missing
- **Throws**: No exceptions raised. Missing symbols are skipped with warnings logged.

### Usage Examples

#### Basic Usage

```python
# Combine close prices for multiple symbols
combined = handler.get_combined_data(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    columns='close'
)

print(combined.head())
# =>         Date  symbol   close
#    0 2024-02-06   AAPL  185.92
#    1 2024-02-06  GOOGL  142.35
#    2 2024-02-06   MSFT  408.55
```

#### Common Usage

```python
# Prepare data for panel regression or Plotly visualization
import plotly.express as px

# Get close prices for tech stocks
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
price_data = handler.get_combined_data(symbols, columns='close')

# Create interactive line chart
fig = px.line(
    price_data,
    x='Date',
    y='close',
    color='symbol',
    title='Tech Stock Prices'
)
fig.show()

# Calculate correlation matrix across symbols
import pandas as pd

pivot_data = price_data.pivot(index='Date', columns='symbol', values='close')
correlation_matrix = pivot_data.corr()
print(correlation_matrix)
```

#### Advanced Usage

```python
# Multi-column analysis with missing data handling
symbols = ['AAPL', 'MSFT', 'GOOGL', 'INVALID_SYMBOL']  # One invalid symbol

combined = handler.get_combined_data(
    symbols=symbols,
    columns=['close', 'volume']
)

# Check which symbols were successfully included
included_symbols = combined['symbol'].unique()
missing_symbols = set(symbols) - set(included_symbols)

print(f"Included: {list(included_symbols)}")
print(f"Missing: {missing_symbols}")
# => Included: ['AAPL', 'MSFT', 'GOOGL']
# => Missing: {'INVALID_SYMBOL'}

# Calculate cross-sectional metrics by date
daily_stats = combined.groupby('Date').agg({
    'close': ['mean', 'std', 'min', 'max'],
    'volume': 'sum'
})

# Find dates where all symbols moved in same direction
pivot_close = combined.pivot(index='Date', columns='symbol', values='close')
daily_returns = pivot_close.pct_change()
all_positive_days = (daily_returns > 0).all(axis=1)
print(f"Days where all symbols were positive: {all_positive_days.sum()}")
```

### Common Pitfalls & Edge Cases

- **Silently skips missing symbols**: ⚠️ If you request 10 symbols but only 7 are in memory, the method returns data for the 7 without raising an error. Always check `combined['symbol'].nunique()` against your expected count.

- **Does NOT auto-download**: Unlike some other methods that might trigger downloads, `get_combined_data()` explicitly refuses to download missing symbols. Pre-download all symbols first.

- **Missing 'Date' column**: If any symbol's data doesn't have a proper datetime index, that symbol is skipped with a warning. This can happen with corrupted cache files.

- **Memory usage with many symbols**: Combining 100+ symbols with all columns can create a very large DataFrame (millions of rows). Use the `columns` parameter to limit data size.

- **Inconsistent date ranges**: If symbols have different date ranges (e.g., IPO dates vary), combined data will have gaps for newer symbols. Handle missing values appropriately in your analysis.

### Best Practices & Anti-Patterns

| ✅ Do this | ❌ Don't do this | Why |
|-----------|------------------|-----|
| Pre-download all symbols before calling | Expect method to download missing data | Method design explicitly avoids implicit downloads for predictability |
| Verify symbol count in output vs. input | Assume all requested symbols are included | Missing symbols fail silently to avoid breaking analysis pipelines |
| Use `columns` parameter to limit data size | Retrieve all columns when you only need close prices | Can reduce DataFrame size by 80% for large symbol lists |
| Use `.pivot()` to convert to wide format if needed | Process long-format data when wide is more appropriate | Many statistical methods (correlation, covariance) expect wide format |

### Related / See Also

- `get_multiple_symbols_data()` – Get data in wide format (each symbol as a column) instead of long format
- `download_data()` – Pre-download symbols before combining
- `get_data()` – Get individual symbol data
- Plotly Express documentation for long-format visualization

---

## Method: get_multiple_symbols_data()

### Overview

Retrieves data for multiple symbols in wide format (one column per symbol), ideal for correlation analysis, portfolio optimization, or time-series modeling where symbols need to be aligned by date.

**When to use**: Cross-sectional statistical analysis, correlation matrices, portfolio returns calculations, any analysis requiring symbol-by-symbol comparison.

### Signature

```python
def get_multiple_symbols_data(
    self,
    symbols: List[str],
    column: str = 'close'
) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `symbols` | `List[str]` | Yes | — | List of ticker symbols to retrieve. Symbols not in memory are skipped with warnings. |
| `column` | `str` | No | `'close'` | Which data column to extract for each symbol. Must be one of: `'open'`, `'high'`, `'low'`, `'close'`, `'volume'`. |

### Returns

- **Type**: `pd.DataFrame`
- **Description**: DataFrame with `date` column plus one column per symbol, all aligned by date with outer join (fills missing dates across symbols with NaN)
- **Possible values**: DataFrame with date column and symbol columns, sorted by date
- **Throws**: 
  - `ValueError` – If `column` parameter is not one of the valid columns, or if no symbols have data available

### Usage Examples

#### Basic Usage

```python
# Get close prices in wide format
prices = handler.get_multiple_symbols_data(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    column='close'
)

print(prices.head())
# =>         date     AAPL    MSFT    GOOGL
#    0 2024-02-06  185.92  408.55   142.35
#    1 2024-02-07  186.75  410.20   143.10
```

#### Common Usage

```python
# Calculate correlation matrix
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
prices = handler.get_multiple_symbols_data(symbols, column='close')

# Drop date column and calculate correlation
price_matrix = prices.drop('date', axis=1)
correlation = price_matrix.corr()

print("Correlation Matrix:")
print(correlation)

# Calculate daily returns
returns = price_matrix.pct_change().dropna()

# Portfolio variance-covariance matrix
cov_matrix = returns.cov()
print("\nCovariance Matrix (daily returns):")
print(cov_matrix)

# Simple equal-weight portfolio returns
portfolio_returns = returns.mean(axis=1)
```

#### Advanced Usage

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Modern Portfolio Theory - Mean-Variance Optimization
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
prices = handler.get_multiple_symbols_data(symbols, column='close')

# Calculate returns and statistics
price_matrix = prices.drop('date', axis=1)
returns = price_matrix.pct_change().dropna()

mean_returns = returns.mean() * 252  # Annualized
cov_matrix = returns.cov() * 252     # Annualized

def portfolio_stats(weights, mean_returns, cov_matrix):
    """Calculate portfolio return and volatility."""
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_vol

def minimize_volatility(mean_returns, cov_matrix):
    """Find minimum variance portfolio."""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    result = minimize(
        lambda w, mr, cm: portfolio_stats(w, mr, cm)[1],
        initial_guess,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result

# Find optimal portfolio
optimal = minimize_volatility(mean_returns, cov_matrix)
optimal_weights = dict(zip(symbols, optimal.x))

print("Optimal Portfolio Weights:")
for symbol, weight in optimal_weights.items():
    print(f"{symbol}: {weight:.2%}")

portfolio_return, portfolio_vol = portfolio_stats(
    optimal.x, mean_returns, cov_matrix
)
print(f"\nExpected Annual Return: {portfolio_return:.2%}")
print(f"Expected Annual Volatility: {portfolio_vol:.2%}")
print(f"Sharpe Ratio: {portfolio_return / portfolio_vol:.2f}")
```

### Common Pitfalls & Edge Cases

- **Invalid column name**: Requesting `column='adjusted_close'` or any column not in `['open', 'high', 'low', 'close', 'volume']` raises `ValueError` immediately. Unlike other methods, this one enforces valid columns strictly.

- **Symbols with different date ranges**: ⚠️ Method uses outer join, so if one symbol has data from 2020-2025 and another from 2023-2025, the result will have NaN for the earlier symbol pre-2023. Always handle NaN values before statistical calculations.

- **Missing symbols are silently skipped**: If 5 symbols are requested but only 3 are in memory, the result will have 3 symbol columns (+ date) without error. Check `len(df.columns) - 1` against expected symbol count.

- **Memory efficiency**: Wide format is less memory-efficient than long format for many symbols. A DataFrame with 500 symbols and 10 years of daily data can consume 10+ GB of RAM.

### Best Practices & Anti-Patterns

| ✅ Do this | ❌ Don't do this | Why |
|-----------|------------------|-----|
| Drop NaN rows before correlation/covariance calculations | Include NaN rows in statistical calculations | NaN values will propagate through matrix operations, producing invalid results |
| Verify symbol count in columns | Assume all requested symbols are present | Missing symbols fail silently; validates data completeness |
| Use `column='close'` for price analysis | Use `'open'` or `'high'` | Close prices are standard for returns and correlation analysis |
| Convert to long format if you have 100+ symbols | Keep wide format for very large symbol lists | Long format is more memory-efficient and works better with groupby operations |

### Related / See Also

- `get_combined_data()` – Get data in long format (row-bound) instead of wide format
- `get_data()` – Get individual symbol data with all columns
- pandas `.corr()` and `.cov()` methods for correlation and covariance matrices
- scipy.optimize for portfolio optimization

---

## Method: get_info()

### Overview

Retrieves company metadata and fundamental information for a symbol directly from yfinance (not from cached price data). Returns details like market cap, sector, description, executives, and financial metrics.

**When to use**: Fundamental analysis, screening based on company attributes, displaying company information in dashboards.

### Signature

```python
def get_info(self, symbol: str) -> Dict
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `symbol` | `str` | Yes | — | Ticker symbol to retrieve information for. Makes a fresh API call; does not use cached data. |

### Returns

- **Type**: `Dict`
- **Description**: Dictionary containing company information fields from yfinance
- **Possible values**: 
  - Dictionary with 50-100+ fields if symbol is valid (field names vary by security type)
  - Empty dictionary `{}` if symbol is invalid or info cannot be retrieved
- **Common fields**: `'longName'`, `'sector'`, `'industry'`, `'marketCap'`, `'dividendYield'`, `'beta'`, `'trailingPE'`, `'forwardPE'`, `'52WeekHigh'`, `'52WeekLow'`, `'averageVolume'`, `'longBusinessSummary'`
- **Throws**: No exceptions raised. Returns empty dict on errors.

### Usage Examples

#### Basic Usage

```python
# Get company information
info = handler.get_info('AAPL')

print(f"Company: {info.get('longName')}")
print(f"Sector: {info.get('sector')}")
print(f"Market Cap: ${info.get('marketCap'):,}")
# => Company: Apple Inc.
# => Sector: Technology
# => Market Cap: $2,850,000,000,000
```

#### Common Usage

```python
# Extract key metrics for multiple symbols
symbols = ['AAPL', 'MSFT', 'GOOGL']
company_metrics = []

for symbol in symbols:
    info = handler.get_info(symbol)
    
    if info:  # Check if info was successfully retrieved
        company_metrics.append({
            'symbol': symbol,
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', None),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', None)
        })

# Convert to DataFrame for analysis
import pandas as pd
metrics_df = pd.DataFrame(company_metrics)
print(metrics_df)
```

#### Advanced Usage

```python
# Build a sector-based screener
def screen_by_sector(handler, sector, min_market_cap=1e9):
    """
    Screen stocks by sector and market cap.
    
    Args:
        handler: YFinanceDataHandler instance
        sector: Target sector (e.g., 'Technology')
        min_market_cap: Minimum market cap in dollars
        
    Returns:
        List of symbols matching criteria
    """
    # Assume you have a universe of symbols
    universe = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC', 'XOM', 'CVX', ...]
    
    matching_symbols = []
    
    for symbol in universe:
        info = handler.get_info(symbol)
        
        if (info.get('sector') == sector and 
            info.get('marketCap', 0) >= min_market_cap):
            matching_symbols.append({
                'symbol': symbol,
                'name': info.get('longName'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE')
            })
    
    return pd.DataFrame(matching_symbols).sort_values('market_cap', ascending=False)

# Find large-cap technology stocks
tech_stocks = screen_by_sector(handler, 'Technology', min_market_cap=100e9)
print(tech_stocks)
```

### Common Pitfalls & Edge Cases

- **Empty dictionary on failures**: ⚠️ Method returns `{}` instead of raising errors when symbol is invalid or API call fails. Always check `if info:` or `if info.get('marketCap'):` before accessing fields.

- **Field availability varies**: Not all symbols have all fields. ETFs, mutual funds, and foreign stocks may have different field sets than US equities. Use `.get()` with defaults, not direct key access.

- **Fresh API call every time**: Unlike price data, info is not cached. Calling `get_info()` repeatedly for the same symbol makes multiple API requests. Cache results yourself if making frequent calls.

- **Rate limiting**: Calling `get_info()` for 100+ symbols in quick succession may trigger rate limits. Add delays between calls if screening large universes.

- **Data staleness**: Company info may be stale by hours or days depending on yfinance's data source. Don't use for real-time fundamental data.

### Best Practices & Anti-Patterns

| ✅ Do this | ❌ Don't do this | Why |
|-----------|------------------|-----|
| Use `.get(field, default)` for field access | Access fields directly with `info['marketCap']` | Prevents KeyError if field is missing for some security types |
| Cache info results if using repeatedly | Call `get_info()` in tight loops | Each call makes an API request; cache for performance |
| Verify `info` is not empty before extracting | Assume info retrieval always succeeds | Invalid symbols or API failures return empty dict |
| Batch requests with delays for large universes | Request info for 500+ symbols rapidly | Rate limiting will cause failures; add 0.1-0.2s delays |

### Related / See Also

- yfinance documentation for complete list of info fields
- `download_data()` – For price and volume data
- pandas DataFrame for organizing company metrics across symbols

---

## Method: list_available_data()

### Overview

Returns a summary of all data currently stored in memory, including row counts, column names, date ranges, and data quality metrics. Useful for debugging, data quality checks, and understanding what's been loaded.

### Signature

```python
def list_available_data(self) -> Dict[str, Dict]
```

### Parameters

None

### Returns

- **Type**: `Dict[str, Dict]`
- **Description**: Nested dictionary with symbol as key, metadata dict as value
- **Metadata fields**:
  - `'rows'` (int): Number of data rows
  - `'columns'` (List[str]): Column names
  - `'date_range'` (str): Date range as `'YYYY-MM-DD to YYYY-MM-DD'`
  - `'missing_values'` (int): Total count of NaN values across all columns
- **Possible values**: Dictionary with metadata for each symbol in memory

### Usage Examples

#### Basic Usage

```python
summary = handler.list_available_data()

for symbol, info in summary.items():
    print(f"{symbol}: {info['rows']} rows, {info['date_range']}")
# => AAPL: 1258 rows, 2020-02-06 to 2025-02-06
# => MSFT: 1258 rows, 2020-02-06 to 2025-02-06
```

#### Common Usage

```python
# Data quality audit across all symbols
summary = handler.list_available_data()

for symbol, info in summary.items():
    missing_pct = (info['missing_values'] / info['rows']) * 100 if info['rows'] > 0 else 0
    
    print(f"{symbol}:")
    print(f"  Rows: {info['rows']}")
    print(f"  Date Range: {info['date_range']}")
    print(f"  Missing Values: {info['missing_values']} ({missing_pct:.1f}%)")
    print(f"  Columns: {', '.join(info['columns'])}")
    print()
```

#### Advanced Usage

```python
# Find symbols with data quality issues
def find_problematic_symbols(handler, max_missing_pct=5.0, min_rows=252):
    """
    Identify symbols with missing data or insufficient history.
    
    Args:
        handler: YFinanceDataHandler instance
        max_missing_pct: Maximum acceptable percentage of missing values
        min_rows: Minimum required number of rows
        
    Returns:
        Dict with 'low_quality' and 'insufficient_data' symbol lists
    """
    summary = handler.list_available_data()
    
    low_quality = []
    insufficient_data = []
    
    for symbol, info in summary.items():
        # Check data quality
        missing_pct = (info['missing_values'] / info['rows']) * 100 if info['rows'] > 0 else 100
        if missing_pct > max_missing_pct:
            low_quality.append({
                'symbol': symbol,
                'missing_pct': missing_pct,
                'missing_values': info['missing_values']
            })
        
        # Check data sufficiency
        if info['rows'] < min_rows:
            insufficient_data.append({
                'symbol': symbol,
                'rows': info['rows'],
                'date_range': info['date_range']
            })
    
    return {
        'low_quality': low_quality,
        'insufficient_data': insufficient_data
    }

# Run audit
issues = find_problematic_symbols(handler)

if issues['low_quality']:
    print("Symbols with >5% missing data:")
    for item in issues['low_quality']:
        print(f"  {item['symbol']}: {item['missing_pct']:.1f}% missing")

if issues['insufficient_data']:
    print("\nSymbols with <252 rows (less than ~1 year of daily data):")
    for item in issues['insufficient_data']:
        print(f"  {item['symbol']}: {item['rows']} rows")
```

### Common Pitfalls & Edge Cases

- **Returns empty dict if no data loaded**: If `download_data()` hasn't been called yet, returns `{}`. Always check `if summary:` before iterating.

- **Missing values count can be high**: Forward-filling (automatic in `download_data()`) reduces but doesn't eliminate missing values. High counts may indicate delisted securities or data quality issues.

- **Date range format can fail**: In rare cases with malformed data, `date_range` might show `"Unable to determine date range"` or `"Invalid date range"`. Handle this in production code.

### Best Practices & Anti-Patterns

| ✅ Do this | ❌ Don't do this | Why |
|-----------|------------------|-----|
| Use this for data quality audits before analysis | Assume all downloaded data is complete and clean | Identifies problematic symbols early in pipeline |
| Log summary after bulk downloads | Skip data quality checks | Provides visibility into download success rates |
| Filter symbols by row count before analysis | Process all symbols regardless of data sufficiency | Prevents errors from insufficient historical data |

### Related / See Also

- `list_cached_data()` – List data files in cache directory
- `list_cached_symbols()` – List just the symbol names in cache

---

## Method: list_cached_data()

### Overview

Scans the cache directory and returns metadata about all cached parquet files, including symbols, periods, intervals, file sizes, and last modified timestamps.

**When to use**: Cache management, debugging cache issues, auditing disk usage.

### Signature

```python
def list_cached_data(self) -> Dict[str, Dict]
```

### Parameters

None

### Returns

- **Type**: `Dict[str, Dict]`
- **Description**: Dictionary with filenames as keys, file metadata as values
- **Metadata fields**:
  - `'symbol'` (str): Ticker symbol extracted from filename
  - `'period'` (str): Period string (e.g., `'1y'`, `'5y'`)
  - `'interval'` (str): Interval string (e.g., `'1d'`, `'1h'`)
  - `'size_kb'` (float): File size in kilobytes
  - `'last_modified'` (datetime): Last modification timestamp
- **Possible values**: 
  - Dictionary with cache file metadata if cache directory exists and contains files
  - Empty dict `{}` if cache directory doesn't exist or is empty

### Usage Examples

#### Basic Usage

```python
cached = handler.list_cached_data()

for filename, info in cached.items():
    print(f"{filename}: {info['symbol']}, {info['size_kb']} KB")
# => AAPL_1y_1d.parquet: AAPL, 156.3 KB
# => MSFT_5y_1wk.parquet: MSFT, 78.9 KB
```

#### Common Usage

```python
# Check cache disk usage
cached = handler.list_cached_data()

total_size_mb = sum(info['size_kb'] for info in cached.values()) / 1024
print(f"Total cache size: {total_size_mb:.2f} MB")
print(f"Total files: {len(cached)}")

# Find stale cache files (older than 7 days)
from datetime import datetime, timedelta

stale_threshold = datetime.now() - timedelta(days=7)
stale_files = []

for filename, info in cached.items():
    if info['last_modified'] < stale_threshold:
        stale_files.append(filename)

print(f"\nStale cache files (>7 days old): {len(stale_files)}")
for filename in stale_files:
    print(f"  {filename}")
```

#### Advanced Usage

```python
# Automated cache cleanup based on age and size
def cleanup_cache(handler, max_age_days=30, max_size_mb=1000):
    """
    Remove old cache files or largest files if cache exceeds size limit.
    
    Args:
        handler: YFinanceDataHandler instance
        max_age_days: Remove files older than this
        max_size_mb: Maximum total cache size in MB
        
    Returns:
        Dict with cleanup statistics
    """
    from datetime import datetime, timedelta
    
    cached = handler.list_cached_data()
    
    if not cached:
        return {'removed_files': 0, 'freed_mb': 0}
    
    # Remove files older than max_age_days
    age_threshold = datetime.now() - timedelta(days=max_age_days)
    old_symbols = []
    
    for filename, info in cached.items():
        if info['last_modified'] < age_threshold:
            old_symbols.append(info['symbol'])
    
    removed_count = 0
    freed_kb = 0
    
    if old_symbols:
        removed_count = handler.clear_cache(symbols=list(set(old_symbols)))
        
        # Recalculate cache
        cached = handler.list_cached_data()
    
    # Check total size
    total_kb = sum(info['size_kb'] for info in cached.values())
    total_mb = total_kb / 1024
    
    # If still over limit, remove largest files
    if total_mb > max_size_mb:
        sorted_files = sorted(
            cached.items(),
            key=lambda x: x[1]['size_kb'],
            reverse=True
        )
        
        freed_mb = 0
        symbols_to_remove = []
        
        for filename, info in sorted_files:
            if total_mb - freed_mb <= max_size_mb:
                break
            
            symbols_to_remove.append(info['symbol'])
            freed_mb += info['size_kb'] / 1024
        
        if symbols_to_remove:
            removed_count += handler.clear_cache(symbols=list(set(symbols_to_remove)))
    
    return {
        'removed_files': removed_count,
        'freed_mb': round(freed_kb / 1024, 2)
    }

# Run cleanup
stats = cleanup_cache(handler, max_age_days=14, max_size_mb=500)
print(f"Cleanup complete: {stats['removed_files']} files removed, {stats['freed_mb']} MB freed")
```

### Common Pitfalls & Edge Cases

- **Returns empty dict if no cache configured**: If `cache_dir=None` in constructor, this method returns `{}` and logs a warning. Not an error.

- **Filename parsing can fail**: If cache files are manually renamed or corrupted, `symbol/period/interval` extraction may fall back to `'unknown'`. Files are still listed but with limited metadata.

- **File size is KB not MB**: `size_kb` is in kilobytes. Divide by 1024 for megabytes.

- **Timestamps are local time**: `last_modified` uses system local timezone, not UTC. Be careful with timezone-sensitive comparisons.

### Best Practices & Anti-Patterns

| ✅ Do this | ❌ Don't do this | Why |
|-----------|------------------|-----|
| Check cache size before large downloads | Let cache grow indefinitely | Prevents disk space issues on long-running systems |
| Implement automated cleanup for production systems | Manual cache management | Production systems need automatic maintenance |
| Log cache metrics for monitoring | Ignore cache state | Cache performance impacts overall system performance |

### Related / See Also

- `list_cached_symbols()` – Get just the list of cached symbols
- `clear_cache()` – Remove cached files
- `list_available_data()` – List data in memory (not cache)

---

## Method: list_cached_symbols()

### Overview

Returns a deduplicated list of all symbols that have cached data files, regardless of period or interval. Useful for checking which symbols are available offline.

### Signature

```python
def list_cached_symbols(self) -> List[str]
```

### Parameters

None

### Returns

- **Type**: `List[str]`
- **Description**: Sorted list of unique ticker symbols found in cache directory
- **Possible values**: 
  - List of symbols if cache exists and contains files
  - Empty list `[]` if no cache directory or no cached files

### Usage Examples

#### Basic Usage

```python
cached_symbols = handler.list_cached_symbols()
print(f"Cached symbols: {cached_symbols}")
# => Cached symbols: ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
```

#### Common Usage

```python
# Determine which symbols need downloading
desired_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
cached_symbols = handler.list_cached_symbols()

symbols_to_download = [s for s in desired_symbols if s not in cached_symbols]

if symbols_to_download:
    print(f"Downloading missing symbols: {symbols_to_download}")
    handler.download_data(symbols_to_download, period='1y')
else:
    print("All symbols already cached")

# Load from cache
handler.download_data(desired_symbols, use_cache=True)
```

#### Advanced Usage

```python
# Synchronize cache with analysis universe
def sync_cache_with_universe(handler, universe_symbols, period='1y', interval='1d'):
    """
    Ensure all universe symbols are cached, remove orphaned cache files.
    
    Args:
        handler: YFinanceDataHandler instance
        universe_symbols: List of symbols that should be cached
        period: Period to download for missing symbols
        interval: Interval to download
        
    Returns:
        Dict with sync statistics
    """
    cached_symbols = handler.list_cached_symbols()
    
    # Find missing symbols
    missing = [s for s in universe_symbols if s not in cached_symbols]
    
    # Find orphaned symbols (in cache but not in universe)
    orphaned = [s for s in cached_symbols if s not in universe_symbols]
    
    # Download missing
    if missing:
        print(f"Downloading {len(missing)} missing symbols...")
        handler.download_data(missing, period=period, interval=interval)
    
    # Clean orphaned
    if orphaned:
        print(f"Removing {len(orphaned)} orphaned symbols from cache...")
        handler.clear_cache(symbols=orphaned)
    
    return {
        'downloaded': len(missing),
        'removed': len(orphaned),
        'total_cached': len(handler.list_cached_symbols())
    }

# Sync cache
universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Your analysis universe
stats = sync_cache_with_universe(handler, universe)
print(stats)
```

### Common Pitfalls & Edge Cases

- **Returns empty list if no cache**: Not an error; indicates no cache directory or empty cache.

- **Deduplication across periods/intervals**: A symbol with `1y_1d` and `5y_1wk` cache files appears only once in the list.

- **Case sensitivity**: Symbols are uppercase. If cache files were manually created with lowercase, they may not appear.

### Best Practices & Anti-Patterns

| ✅ Do this | ❌ Don't do this | Why |
|-----------|------------------|-----|
| Use to implement smart download logic | Always download without checking cache | Reduces API calls and latency dramatically |
| Sync cache with analysis universe regularly | Let cache diverge from current needs | Keeps cache lean and relevant |

### Related / See Also

- `list_cached_data()` – Get detailed cache file information
- `clear_cache()` – Remove cached files
- `download_data()` – Download data with caching

---

## Method: save_data()

### Overview

Exports downloaded data to files in multiple formats (CSV, Excel, Parquet) with flexible strategies for handling multiple symbols (separate files, single combined file, or Excel sheets).

**When to use**: Archiving data, sharing data with colleagues, integrating with external tools, backup before analysis.

### Signature

```python
def save_data(
    self,
    filepath: str,
    symbols: Optional[List[str]] = None,
    format: str = 'csv',
    multi_symbol_strategy: Literal['separate_files', 'single_file', 'excel_sheets'] = 'separate_files',
    combine_column: str = 'close'
) -> None
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `filepath` | `str` | Yes | — | Base path for output file(s). Directory is created automatically if it doesn't exist. For `'separate_files'` strategy, this is the base name with symbol suffix added. |
| `symbols` | `Optional[List[str]]` | No | `None` | Specific symbols to save. If `None`, saves all symbols currently in memory. |
| `format` | `str` | No | `'csv'` | Output format: `'csv'`, `'excel'`, or `'parquet'`. Invalid formats raise `ValueError`. |
| `multi_symbol_strategy` | `Literal` | No | `'separate_files'` | How to handle multiple symbols: `'separate_files'` (one file per symbol), `'single_file'` (combined into one file in long format), or `'excel_sheets'` (one Excel file with multiple sheets). |
| `combine_column` | `str` | No | `'close'` | Column to use when `multi_symbol_strategy='single_file'`. Typically `'close'` or `'volume'`. Ignored for other strategies. |

### Returns

- **Type**: `None`
- **Side effects**: Creates file(s) on disk, logs save operations
- **Throws**: 
  - `ValueError` – Invalid format or strategy, or no symbols with data available
  - `FileNotFoundError` – Unable to create output directory (rare; usually auto-created)

### Usage Examples

#### Basic Usage

```python
# Save single symbol to CSV
handler.save_data(
    filepath='./output/aapl_data.csv',
    symbols=['AAPL'],
    format='csv'
)
# Creates: ./output/aapl_data.csv
```

#### Common Usage

```python
# Save multiple symbols as separate CSV files
symbols = ['AAPL', 'MSFT', 'GOOGL']
handler.save_data(
    filepath='./output/stock_data.csv',
    symbols=symbols,
    format='csv',
    multi_symbol_strategy='separate_files'
)
# Creates:
#   ./output/stock_data_AAPL.csv
#   ./output/stock_data_MSFT.csv
#   ./output/stock_data_GOOGL.csv

# Save multiple symbols to single Excel file with sheets
handler.save_data(
    filepath='./output/tech_stocks.xlsx',
    symbols=symbols,
    format='excel',
    multi_symbol_strategy='excel_sheets'
)
# Creates: ./output/tech_stocks.xlsx with 3 sheets (AAPL, MSFT, GOOGL)

# Save combined close prices to single CSV
handler.save_data(
    filepath='./output/combined_prices.csv',
    symbols=symbols,
    format='csv',
    multi_symbol_strategy='single_file',
    combine_column='close'
)
# Creates: ./output/combined_prices.csv in long format (Date, symbol, close)
```

#### Advanced Usage

```python
# Comprehensive export pipeline
def export_analysis_data(handler, symbols, output_dir='./exports'):
    """
    Export data in multiple formats for different use cases.
    
    Args:
        handler: YFinanceDataHandler instance
        symbols: List of symbols to export
        output_dir: Base directory for exports
    """
    from pathlib import Path
    import os
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Parquet for fast re-loading (separate files)
    print("Exporting to Parquet (separate files)...")
    handler.save_data(
        filepath=f'{output_dir}/parquet/data.parquet',
        symbols=symbols,
        format='parquet',
        multi_symbol_strategy='separate_files'
    )
    
    # 2. Excel with sheets for manual inspection
    print("Exporting to Excel (multi-sheet)...")
    handler.save_data(
        filepath=f'{output_dir}/analysis_data.xlsx',
        symbols=symbols,
        format='excel',
        multi_symbol_strategy='excel_sheets'
    )
    
    # 3. CSV combined for R/Python external analysis
    print("Exporting to CSV (combined close prices)...")
    handler.save_data(
        filepath=f'{output_dir}/close_prices.csv',
        symbols=symbols,
        format='csv',
        multi_symbol_strategy='single_file',
        combine_column='close'
    )
    
    # 4. CSV combined for volume analysis
    print("Exporting to CSV (combined volume)...")
    handler.save_data(
        filepath=f'{output_dir}/volume_data.csv',
        symbols=symbols,
        format='csv',
        multi_symbol_strategy='single_file',
        combine_column='volume'
    )
    
    print(f"Export complete. Files in: {output_dir}")
    
    # Print file sizes
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            filepath = os.path.join(root, file)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {file}: {size_mb:.2f} MB")

# Use in workflow
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
export_analysis_data(handler, symbols, output_dir='./data_exports')
```

### Common Pitfalls & Edge Cases

- **Excel sheet name length limit**: ⚠️ Excel sheet names are limited to 31 characters. Symbols with colons or slashes (e.g., `'BRK.A'`, `'BTC-USD'`) are sanitized by replacing with underscores and truncating. This can cause name collisions for similar symbols.

- **Separate files with wrong extension**: When using `separate_files`, if `filepath='output.csv'` and `format='excel'`, files will still get `.xlsx` extension (not `.csv`). Always match filepath extension to format.

- **Single file strategy with all columns**: `single_file` strategy only includes the specified `combine_column`, not all columns. To save all columns in single file, manually use `get_combined_data()` then save the result.

- **Directory creation**: Parent directories are created automatically. No need to pre-create output directories.

- **Overwriting files**: Existing files are overwritten without warning. Back up important files before saving.

### Best Practices & Anti-Patterns

| ✅ Do this | ❌ Don't do this | Why |
|-----------|------------------|-----|
| Use Parquet for large datasets | Use CSV for multi-GB datasets | Parquet is 5-10x smaller and faster to read/write |
| Match filepath extension to format | Use `.csv` extension with `format='excel'` | Avoids confusion about actual file type |
| Use `excel_sheets` for reports | Use `separate_files` when creating reports for colleagues | Single Excel file is easier to share and navigate |
| Specify symbols explicitly in production | Use `symbols=None` (save all) in automated pipelines | Makes pipeline behavior predictable and documented |

### Related / See Also

- `get_combined_data()` – Get combined data before saving
- `get_multiple_symbols_data()` – Get wide-format data before saving
- pandas `.to_csv()`, `.to_excel()`, `.to_parquet()` methods for custom export logic

---

## Method: clear_cache()

### Overview

Removes cached parquet files for specified symbols or all symbols. Useful for forcing fresh downloads or managing disk space.

### Signature

```python
def clear_cache(
    self,
    symbols: Optional[List[str]] = None
) -> int
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `symbols` | `Optional[List[str]]` | No | `None` | Specific symbols to remove from cache. If `None`, clears entire cache directory. |

### Returns

- **Type**: `int`
- **Description**: Number of cache files successfully removed
- **Possible values**: Integer >= 0 representing count of deleted files
- **Throws**: 
  - `OSError` – Unable to remove cache files (rare; usually permissions issue)

### Usage Examples

#### Basic Usage

```python
# Clear cache for specific symbols
removed = handler.clear_cache(symbols=['AAPL', 'MSFT'])
print(f"Removed {removed} cache files")
# => Removed 2 cache files

# Clear entire cache
removed = handler.clear_cache()
print(f"Removed {removed} cache files")
# => Removed 47 cache files
```

#### Common Usage

```python
# Force fresh download by clearing cache first
handler.clear_cache(symbols=['AAPL'])
handler.download_data('AAPL', period='1y', use_cache=True)
# Even with use_cache=True, downloads fresh data since cache was cleared

# Clear stale cache before critical analysis
print("Clearing cache before production run...")
removed = handler.clear_cache()
print(f"Removed {removed} old cache files")

# Download fresh data for all symbols
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
handler.download_data(symbols, period='2y', use_cache=False)
```

#### Advanced Usage

```python
from datetime import datetime, timedelta

# Clear cache files older than 7 days
def clear_stale_cache(handler, max_age_days=7):
    """
    Clear cache files older than specified days.
    
    Args:
        handler: YFinanceDataHandler instance
        max_age_days: Maximum age in days before cache is considered stale
        
    Returns:
        Number of files removed
    """
    cached_data = handler.list_cached_data()
    age_threshold = datetime.now() - timedelta(days=max_age_days)
    
    stale_symbols = set()
    
    for filename, info in cached_data.items():
        if info['last_modified'] < age_threshold:
            stale_symbols.add(info['symbol'])
    
    if stale_symbols:
        print(f"Clearing {len(stale_symbols)} stale symbols...")
        return handler.clear_cache(symbols=list(stale_symbols))
    else:
        print("No stale cache files found")
        return 0

# Use in automated workflow
removed = clear_stale_cache(handler, max_age_days=7)
print(f"Removed {removed} stale cache files")
```

### Common Pitfalls & Edge Cases

- **Clears all periods/intervals for a symbol**: Calling `clear_cache(symbols=['AAPL'])` removes ALL cached files for AAPL, regardless of period or interval. If you have `AAPL_1y_1d.parquet` and `AAPL_5y_1wk.parquet`, both are deleted.

- **Doesn't affect in-memory data**: Clearing cache doesn't remove data from `handler.data`. To clear memory, you'd need to manually `del handler.data[symbol]` or create a new handler instance.

- **Returns 0 if no cache directory**: If `cache_dir=None` or cache directory doesn't exist, logs a warning and returns `0`. Not an error.

- **Case-sensitive symbol matching**: Symbol names are case-sensitive. Clearing `['aapl']` won't clear files for `'AAPL'`.

### Best Practices & Anti-Patterns

| ✅ Do this | ❌ Don't do this | Why |
|-----------|------------------|-----|
| Clear cache before critical production runs | Trust old cached data for time-sensitive analysis | Ensures fresh, accurate data for important decisions |
| Implement age-based cache cleanup | Clear entire cache randomly | Preserves frequently-used symbols while removing stale data |
| Clear specific symbols when you need fresh data | Clear entire cache just to update one symbol | Preserves other cached data, reduces download time |

### Related / See Also

- `list_cached_data()` – See what's in cache before clearing
- `list_cached_symbols()` – Get list of cached symbols
- `download_data()` with `use_cache=False` – Alternative to clearing then downloading

---

## Usage Patterns & Best Practices

### Typical Workflow

```python
from yfinance_handler import YFinanceDataHandler
import logging

# 1. Initialize with caching
handler = YFinanceDataHandler(
    cache_dir='./market_data_cache',
    enable_logging=True,
    chunk_size=30,
    log_level=logging.INFO
)

# 2. Download data (uses cache if available)
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
handler.download_data(
    symbols=symbols,
    period='2y',
    interval='1d',
    use_cache=True
)

# 3. Verify data quality
summary = handler.list_available_data()
for symbol, info in summary.items():
    print(f"{symbol}: {info['rows']} rows, {info['missing_values']} missing")

# 4. Access data for analysis
# Option A: Individual symbols
aapl_data = handler.get_data('AAPL', columns=['close', 'volume'])

# Option B: Combined long format
combined = handler.get_combined_data(symbols, columns='close')

# Option C: Wide format for correlation
prices = handler.get_multiple_symbols_data(symbols, column='close')

# 5. Save results
handler.save_data(
    filepath='./output/analysis_data.xlsx',
    symbols=symbols,
    format='excel',
    multi_symbol_strategy='excel_sheets'
)
```

### Performance Optimization

**Caching strategy**:
- Always enable caching for production workflows
- Set `cache_dir` outside project directory (e.g., `~/.market_data_cache`)
- Use `use_cache=True` (default) for repeated analysis
- Implement age-based cache cleanup (clear cache > 7-14 days old)

**Bulk downloads**:
- Use `chunk_size=20-30` for 100+ symbols to avoid rate limits
- Enable `threads=True` for parallel downloads
- Download during off-peak hours if processing 500+ symbols

**Memory management**:
- Request specific columns instead of all columns when possible
- Clear `handler.data` for symbols you no longer need: `del handler.data[symbol]`
- For very large symbol lists (1000+), process in batches and save results incrementally

### Error Handling Patterns

```python
# Robust symbol processing
def process_symbols_safely(handler, symbols, period='1y'):
    """
    Download and process symbols with comprehensive error handling.
    """
    successful = []
    failed = []
    
    for symbol in symbols:
        try:
            # Attempt download
            data = handler.download_data(symbol, period=period)
            
            # Verify data quality
            if symbol in data and not data[symbol].empty:
                summary = handler.list_available_data()
                
                if summary[symbol]['missing_values'] / summary[symbol]['rows'] < 0.05:
                    successful.append(symbol)
                else:
                    print(f"Warning: {symbol} has high missing data percentage")
                    successful.append(symbol)  # Include but flag
            else:
                failed.append((symbol, "No data returned"))
                
        except ValueError as e:
            failed.append((symbol, f"ValueError: {e}"))
        except Exception as e:
            failed.append((symbol, f"Unexpected error: {e}"))
    
    return {
        'successful': successful,
        'failed': failed,
        'success_rate': len(successful) / len(symbols) if symbols else 0
    }

# Use in pipeline
results = process_symbols_safely(handler, ['AAPL', 'MSFT', 'INVALID', 'GOOGL'])
print(f"Success rate: {results['success_rate']:.1%}")
print(f"Failed symbols: {results['failed']}")
```

---

## Common Error Messages & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError: "Data for symbol 'AAPL' not found"` | Symbol not downloaded yet | Call `handler.download_data('AAPL')` first |
| `ValueError: Intraday interval 1m only supports periods: [...]` | Invalid period/interval combination | Use `period='1d'` or `'5d'` with minute-level intervals |
| `ValueError: No data could be retrieved for any of the specified symbols` | All downloads failed | Check internet connection, verify symbols are valid, check yfinance service status |
| `TypeError: Symbols must be a string or list of strings` | Invalid symbol type | Pass string (`'AAPL'`) or list of strings (`['AAPL', 'MSFT']`) |
| Empty DataFrame returned from `get_data()` with `columns` parameter | Requested columns don't exist | Check `data.columns` for available columns; use lowercase names |

---

## Appendix: Data Format Reference

### Standard DataFrame Structure

After `download_data()`, DataFrames have this structure:

```
Index: DatetimeIndex (trading days only, no weekends/holidays)

Columns (lowercase):
- open: float64
- high: float64
- low: float64
- close: float64
- volume: int64
- dividends: float64 (if auto_adjust=False)
- stock splits: float64 (if auto_adjust=False)
```

### OHLC Format

From `get_ohlc_data()`:

```
Index: date (datetime)

Columns:
- date: datetime64
- open: float64
- high: float64
- low: float64
- close: float64
```

### Long Format

From `get_combined_data()`:

```
Columns:
- Date: datetime64
- symbol: object (string)
- [requested column(s)]: float64 or int64
```

### Wide Format

From `get_multiple_symbols_data()`:

```
Columns:
- date: datetime64
- [symbol1]: float64
- [symbol2]: float64
- ...
```

---

## See Also

- **yfinance documentation**: https://pypi.org/project/yfinance/
- **pandas documentation**: https://pandas.pydata.org/docs/
- **TA-Lib (technical analysis)**: https://mrjbq7.github.io/ta-lib/
- **pandas_ta (technical analysis)**: https://github.com/twopirllc/pandas-ta

---

*Documentation generated for YFinanceDataHandler class - Production-grade financial data management for Python*