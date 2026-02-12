"""
OHLC data processing module for relative price calculations.

This module provides the OHLCProcessor class for calculating asset prices
relative to a benchmark index or security. Supports custom column naming
and various normalization options.

Classes:
    OHLCColumns: Configuration dataclass for OHLC column naming.
    OHLCProcessor: Main processor for relative price calculations.

Typical usage:
    processor = OHLCProcessor()
    relative_data = processor.calculate_relative_prices(stock_df, benchmark_df)
"""
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

# Configure module-level logging - let application configure handlers
logger = logging.getLogger(__name__)


@dataclass
class OHLCColumns:
    """Configuration for OHLC column naming conventions."""
    open: str = 'open'
    high: str = 'high'
    low: str = 'low'
    close: str = 'close'
    date: str = 'date'


class OHLCProcessor:
    """
    Processes OHLC (Open, High, Low, Close) financial data.
    
    Primary responsibility: Calculate relative prices of an asset
    against a benchmark index/security.
    """
    
    def __init__(
        self, 
        column_config: Optional[OHLCColumns] = None
    ):
        """
        Initialize OHLCProcessor.

        Args:
            column_config: Custom column naming configuration.
        """
        self.columns: OHLCColumns = column_config or OHLCColumns()
        self.logger: logging.Logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def calculate_relative_prices(
        self, 
        stock_data: pd.DataFrame, 
        benchmark_data: pd.DataFrame,
        benchmark_column: str = 'close',
        digits: int = 4,
        rebase: bool = True
    ) -> pd.DataFrame:
        """
        Calculate asset prices relative to a benchmark.
        
        Divides each OHLC value by the corresponding benchmark value,
        optionally rebasing the benchmark to 1.0 at the start.
        
        Args:
            stock_data: DataFrame with OHLC columns (open, high, low, close) and date.
            benchmark_data: Benchmark DataFrame with date and price column.
            benchmark_column: Column name to use from benchmark (default: 'close').
            digits: Decimal places for rounding (0-10, default: 4).
            rebase: If True, rebase benchmark to 1.0 at first data point (default: True).
            
        Returns:
            DataFrame with original columns plus relative columns: ropen, rhigh, rlow, rclose.
            
        Raises:
            TypeError: If inputs are not DataFrames.
            ValueError: If required columns missing, DataFrames empty, or parameters invalid.
        """
        
        # Fail fast: Type validation
        if not isinstance(stock_data, pd.DataFrame):
            raise TypeError(
                f"stock_data must be pandas.DataFrame, got {type(stock_data).__name__}"
            )
        if not isinstance(benchmark_data, pd.DataFrame):
            raise TypeError(
                f"benchmark_data must be pandas.DataFrame, got {type(benchmark_data).__name__}"
            )
        
        # Fail fast: Empty DataFrame check
        if stock_data.empty:
            raise ValueError("stock_data cannot be empty")
        if benchmark_data.empty:
            raise ValueError("benchmark_data cannot be empty")
        
        # Fail fast: Parameter range validation
        if not isinstance(digits, int) or not 0 <= digits <= 10:
            raise ValueError(f"digits must be integer between 0-10, got {digits}")
        
        try:
            # Normalize data structures (validates columns internally)
            normalized_stock = self._normalize_dataframe(
                stock_data, 
                require_ohlc=True
            )
            normalized_benchmark = self._normalize_dataframe(
                benchmark_data,
                require_ohlc=False
            )
            
            # Validate benchmark column exists after normalization
            if benchmark_column not in normalized_benchmark.columns:
                available = ', '.join(normalized_benchmark.columns)
                raise ValueError(
                    f"Benchmark column '{benchmark_column}' not found. "
                    f"Available columns: {available}"
                )
            
            # DEBUG only - developers can enable to trace execution
            self.logger.debug(
                "Calculating relative prices: stock_rows=%d, benchmark_rows=%d, benchmark_col='%s', rebase=%s",
                len(normalized_stock),
                len(normalized_benchmark),
                benchmark_column,
                rebase
            )
            
            result = self._calculate_relative(
                df=normalized_stock,
                _o=self.columns.open,
                _h=self.columns.high,
                _l=self.columns.low,
                _c=self.columns.close,
                bm_df=normalized_benchmark,
                bm_col=benchmark_column,
                dgt=digits,
                rebase=rebase
            )
            
            # No INFO log for success - silence is golden
            return result
        
        except Exception as e:
            # ERROR: Only log when something actually goes wrong
            self.logger.error(
                "Relative price calculation failed: %s (stock_shape=%s, benchmark_shape=%s, benchmark_col='%s')",
                str(e),
                stock_data.shape,
                benchmark_data.shape,
                benchmark_column,
                exc_info=True  # Include stack trace
            )
            
            # Re-raise with context
            raise ValueError(
                f"Relative price calculation failed: {str(e)} "
                f"(stock_shape: {stock_data.shape}, "
                f"benchmark_column: '{benchmark_column}')"
            ) from e
    
    def _normalize_dataframe(
        self, 
        df: pd.DataFrame, 
        require_ohlc: bool = False
    ) -> pd.DataFrame:
        """
        Normalize DataFrame structure for consistent processing.
        
        Ensures 'date' column exists and OHLC columns are lowercase.
        Returns a copy without modifying the input.
        
        Args:
            df: Input DataFrame to normalize.
            require_ohlc: If True, validates presence of OHLC columns.
            
        Returns:
            New DataFrame with normalized structure.
            
        Raises:
            ValueError: If required columns are missing or structure is invalid.
        """
        # Work on copy to avoid mutating input
        normalized = df.copy()
        
        # Ensure date column exists
        if 'date' not in normalized.columns:
            if normalized.index.name in ('Date', 'date', None):
                normalized = normalized.reset_index()
                first_col = normalized.columns[0]
                if first_col in ('index', 'Date', 'level_0'):
                    normalized.rename(columns={first_col: 'date'}, inplace=True)
            else:
                raise ValueError(
                    f"Cannot find date column. "
                    f"Index name '{normalized.index.name}' not recognized. "
                    f"Expected 'Date', 'date', or unnamed index."
                )
        
        # Normalize column names to lowercase
        col_mapping = {}
        for col in normalized.columns:
            lower_col = col.lower()
            if lower_col in ('open', 'high', 'low', 'close', 'date'):
                col_mapping[col] = lower_col
        
        if col_mapping:
            normalized.rename(columns=col_mapping, inplace=True)
        
        # Validate OHLC columns if required
        if require_ohlc:
            required_cols = {
                self.columns.open, 
                self.columns.high, 
                self.columns.low, 
                self.columns.close
            }
            actual_cols = set(normalized.columns)
            missing = required_cols - actual_cols
            
            if missing:
                available = ', '.join(sorted(actual_cols))
                raise ValueError(
                    f"Missing required OHLC columns: {missing}. "
                    f"Available columns: {available}"
                )
        
        return normalized
    
    def _calculate_relative(
        self, 
        df: pd.DataFrame, 
        _o: str, 
        _h: str, 
        _l: str, 
        _c: str, 
        bm_df: pd.DataFrame, 
        bm_col: str, 
        dgt: int, 
        rebase: bool = True
    ) -> pd.DataFrame:
        """
        Internal helper to calculate relative OHLC prices against benchmark.
        
        Args:
            df: Normalized primary DataFrame with OHLC data.
            _o: Name of 'open' column in df.
            _h: Name of 'high' column in df.
            _l: Name of 'low' column in df.
            _c: Name of 'close' column in df.
            bm_df: Normalized benchmark DataFrame.
            bm_col: Column name to use from benchmark.
            dgt: Decimal places for rounding.
            rebase: If True, rebase benchmark to 1.0 at start.

        Returns:
            DataFrame with original columns plus relative price columns (r-prefixed).
            
        Raises:
            ValueError: If benchmark rebasing fails or division by zero occurs.
        """
        # Work on copies to preserve inputs
        working_df = df.copy()
        benchmark = bm_df.copy()
        
        # Rename benchmark column for merging
        benchmark = benchmark.rename(columns={bm_col: 'bm'})
        
        # Merge on date
        merged_df = pd.merge(
            working_df, 
            benchmark[['date', 'bm']], 
            how='left', 
            on='date'
        )
        
        # Check for missing benchmark values with threshold
        missing_bm_count = merged_df['bm'].isna().sum()
        max_missing_pct = 10.0  # Configurable threshold

        if missing_bm_count > 0:
            missing_pct = (missing_bm_count / len(merged_df)) * 100

            if missing_pct > max_missing_pct:
                raise ValueError(
                    f"Too much missing benchmark data: {missing_pct:.1f}% "
                    f"(threshold: {max_missing_pct}%). "
                    f"Missing {missing_bm_count} of {len(merged_df)} rows."
                )

            # WARNING: Data quality issue that could affect results
            self.logger.warning(
                "Missing benchmark data for %d rows (%.1f%%). Values will be forward-filled. "
                "This may indicate misaligned date ranges.",
                missing_bm_count,
                missing_pct
            )

        # Create benchmark adjustment factor (forward-fill missing)
        # NOTE: Don't round here - preserve precision for rebase calculation
        merged_df['bmfx'] = merged_df['bm'].ffill()
        
        # Fail fast: Check for NaN after forward-fill
        if merged_df['bmfx'].isna().any():
            first_valid_idx = merged_df['bmfx'].first_valid_index()
            raise ValueError(
                f"Benchmark data contains NaN values that cannot be forward-filled. "
                f"First valid benchmark value at index: {first_valid_idx}"
            )
        
        # Apply rebase if requested
        if rebase:
            first_bm_value = merged_df['bmfx'].iloc[0]
            
            # Fail fast: Division by zero check
            if first_bm_value == 0:
                raise ValueError(
                    "Cannot rebase: first benchmark value is zero. "
                    "Rebasing requires non-zero initial value."
                )
            
            merged_df['bmfx'] = merged_df['bmfx'] / first_bm_value

        # Check for zero values BEFORE division to prevent inf
        zero_mask = merged_df['bmfx'] == 0
        if zero_mask.any():
            zero_count = zero_mask.sum()
            zero_dates = merged_df.loc[zero_mask, 'date'].tolist()
            raise ValueError(
                f"Benchmark contains zero values at {zero_count} dates. "
                f"First occurrence: {zero_dates[0]}. "
                f"Cannot calculate relative prices with zero divisor."
            )

        # Check for negative values (data quality issue)
        if (merged_df['bmfx'] < 0).any():
            neg_count = (merged_df['bmfx'] < 0).sum()
            raise ValueError(
                f"Benchmark contains {neg_count} negative values. "
                f"This indicates data quality issues."
            )

        # Calculate relative prices for OHLC columns (round only final results)
        merged_df[f'r{_o}'] = (merged_df[_o] / merged_df['bmfx']).round(dgt)
        merged_df[f'r{_h}'] = (merged_df[_h] / merged_df['bmfx']).round(dgt)
        merged_df[f'r{_l}'] = (merged_df[_l] / merged_df['bmfx']).round(dgt)
        merged_df[f'r{_c}'] = (merged_df[_c] / merged_df['bmfx']).round(dgt)

        # Check for inf/nan in results - RAISE exception, don't just warn
        relative_cols = [f'r{_o}', f'r{_h}', f'r{_l}', f'r{_c}']
        for col in relative_cols:
            # Use np.isinf to catch both positive and negative infinity
            invalid_mask = merged_df[col].isnull() | np.isinf(merged_df[col])
            invalid_count = invalid_mask.sum()

            if invalid_count > 0:
                self.logger.error(
                    "Column '%s' contains %d invalid values (NaN/inf). "
                    "This indicates zero or near-zero benchmark values.",
                    col,
                    invalid_count
                )
                raise ValueError(
                    f"Calculation produced {invalid_count} invalid values in '{col}'. "
                    f"This indicates near-zero benchmark values that passed validation."
                )
        
        # Drop temporary benchmark columns
        merged_df = merged_df.drop(columns=['bm', 'bmfx'])
        
        return merged_df