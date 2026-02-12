"""
Pytest configuration and shared fixtures for algoshort tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlc_data() -> pd.DataFrame:
    """
    Create sample OHLC data for testing.

    Returns:
        pd.DataFrame with columns: date, open, high, low, close, volume
    """
    np.random.seed(42)
    n_days = 100

    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')

    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)
    close_prices = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close
    data = {
        'date': dates,
        'open': close_prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': close_prices * (1 + np.random.uniform(0, 0.02, n_days)),
        'low': close_prices * (1 - np.random.uniform(0, 0.02, n_days)),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    }

    df = pd.DataFrame(data)

    # Ensure high >= open, close and low <= open, close
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def sample_ohlc_with_signal(sample_ohlc_data) -> pd.DataFrame:
    """
    Create sample OHLC data with a trading signal column.

    Returns:
        pd.DataFrame with OHLC data plus 'signal' column
    """
    df = sample_ohlc_data.copy()
    # Create alternating signal: 1 (long), -1 (short), 0 (neutral)
    df['signal'] = np.where(
        df.index % 3 == 0, 1,
        np.where(df.index % 3 == 1, -1, 0)
    )
    return df


@pytest.fixture
def benchmark_data() -> pd.DataFrame:
    """
    Create sample benchmark data (e.g., SPY) for relative calculations.

    Returns:
        pd.DataFrame with date and close columns
    """
    np.random.seed(123)
    n_days = 100

    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    base_price = 450.0
    returns = np.random.normal(0.0005, 0.01, n_days)
    close_prices = base_price * np.cumprod(1 + returns)

    return pd.DataFrame({
        'date': dates,
        'close': close_prices
    })
