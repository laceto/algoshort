"""
Plotting utilities for financial data visualization.

This module provides functions for visualizing trading signals, equity curves,
regime detection, and performance metrics.

Example:
    >>> from algoshort.plots import plot_abs_rel, plot_equity_amount
    >>> plot_abs_rel(df, 'AAPL', 'SPY')
    >>> plot_equity_amount(df, 'AAPL', 'momentum')
"""

# Standard library
import logging
from typing import List, Optional, Tuple, Union

# Third-party
import matplotlib.pyplot as plt
import pandas as pd

# Module logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_FIG_WIDTH = 20
DEFAULT_FIG_HEIGHT = 8
DEFAULT_FIGSIZE = (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT)
DATE_COLUMN = 'date'
CLOSE_COLUMN = 'close'
RCLOSE_COLUMN = 'rclose'

__all__ = [
    'plot_abs_rel',
    'plot_signal_bo',
    'plot_signal_tt',
    'plot_signal_ma',
    'plot_signal_abs',
    'plot_signal_rel',
    'plot_regime_abs',
    'plot_regime_rel',
    'plot_profit_loss',
    'plot_price_signal_cumreturns',
    'plot_equity_risk',
    'plot_shares_signal',
    'plot_equity_amount',
]


def _validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    func_name: str
) -> pd.DataFrame:
    """
    Validate DataFrame input and return a copy with date index.

    Args:
        df: Input DataFrame to validate
        required_columns: List of column names that must exist
        func_name: Name of calling function for error messages

    Returns:
        Copy of DataFrame with 'date' as index (if date column exists)

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If df is empty or missing required columns
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{func_name}: df must be a pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError(f"{func_name}: DataFrame cannot be empty")

    # Work on a copy to avoid modifying original
    df = df.copy()

    # Handle date indexing
    if DATE_COLUMN in df.columns:
        df = df.set_index(DATE_COLUMN)
    elif df.index.name != DATE_COLUMN:
        logger.warning("%s: No '%s' column found, using existing index", func_name, DATE_COLUMN)

    # Check required columns (after potential index change)
    missing = [col for col in required_columns if col not in df.columns and col != DATE_COLUMN]
    if missing:
        available = df.columns.tolist()
        raise ValueError(
            f"{func_name}: Missing required columns: {missing}. "
            f"Available columns: {available}"
        )

    return df


def _validate_ticker(ticker: Optional[str], func_name: str) -> str:
    """Validate ticker parameter."""
    if ticker is None or (isinstance(ticker, str) and not ticker.strip()):
        raise ValueError(f"{func_name}: ticker cannot be None or empty")
    return str(ticker)


def _validate_positive_int(value: int, name: str, func_name: str) -> int:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{func_name}: {name} must be a positive integer, got {value}")
    return value


def _close_figure() -> None:
    """Close current figure to prevent memory leaks."""
    plt.close()


def plot_abs_rel(
    df: pd.DataFrame,
    ticker: str,
    bm_name: str
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot absolute vs relative price comparison.

    Args:
        df: DataFrame with 'date', 'close', and 'rclose' columns
        ticker: Ticker symbol for title
        bm_name: Benchmark name for title

    Returns:
        Tuple of (Figure, Axes) objects

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing or df is empty
    """
    func_name = "plot_abs_rel"
    ticker = _validate_ticker(ticker, func_name)
    required = [CLOSE_COLUMN, RCLOSE_COLUMN]
    df = _validate_dataframe(df, required, func_name)

    logger.debug("Plotting absolute vs relative for %s", ticker)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    df[[CLOSE_COLUMN, RCLOSE_COLUMN]].plot(
        ax=ax,
        grid=True,
        title=f'{ticker} Absolute vs relative to {bm_name} rebased'
    )
    plt.show()
    _close_figure()

    return fig, ax


def plot_signal_bo(
    df: pd.DataFrame,
    window: int,
    ticker: str,
    relative: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot breakout signal with high/low channels.

    Args:
        df: DataFrame with OHLC and breakout signal columns
        window: Lookback window for channel calculation
        ticker: Ticker symbol for title
        relative: If True, use relative price columns

    Returns:
        Tuple of (Figure, Axes) objects

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing or parameters invalid
    """
    func_name = "plot_signal_bo"
    ticker = _validate_ticker(ticker, func_name)
    window = _validate_positive_int(window, "window", func_name)

    # Determine column prefixes based on relative flag
    if relative:
        prefix_h = 'rhi_'
        prefix_l = 'rlo_'
        prefix_bo = 'rbo_'
        close = RCLOSE_COLUMN
    else:
        prefix_h = 'hi_'
        prefix_l = 'lo_'
        prefix_bo = 'bo_'
        close = CLOSE_COLUMN

    # Build required column names
    hi_col = f'{prefix_h}{window}'
    lo_col = f'{prefix_l}{window}'
    bo_col = f'{prefix_bo}{window}'

    required = [close, hi_col, lo_col, bo_col]
    df = _validate_dataframe(df, required, func_name)

    logger.debug("Plotting breakout signal for %s, window=%d", ticker, window)

    fig, ax = plt.subplots(figsize=(DEFAULT_FIG_WIDTH, 5))
    df[[close, hi_col, lo_col, bo_col]].plot(
        ax=ax,
        secondary_y=[bo_col],
        style=['k', 'g:', 'r:', 'b-.'],
        title=f'{ticker.upper()} {window} days high/low'
    )
    plt.show()
    _close_figure()

    return fig, ax


def plot_signal_tt(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    ticker: str = ""
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot turtle trading signal.

    Args:
        df: DataFrame with close and turtle signal columns
        fast: Fast channel window
        slow: Slow channel window
        ticker: Ticker symbol for title (optional)

    Returns:
        Tuple of (Figure, Axes) objects

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing or parameters invalid
    """
    func_name = "plot_signal_tt"
    fast = _validate_positive_int(fast, "fast", func_name)
    slow = _validate_positive_int(slow, "slow", func_name)

    tt_col = f'turtle_{slow}{fast}'
    required = [CLOSE_COLUMN, tt_col]
    df = _validate_dataframe(df, required, func_name)

    logger.debug("Plotting turtle signal, fast=%d, slow=%d", fast, slow)

    fig, ax = plt.subplots(figsize=(DEFAULT_FIG_WIDTH, 5))
    df[[CLOSE_COLUMN, tt_col]].plot(
        ax=ax,
        secondary_y=[tt_col],
        style=['k', 'b-.'],
        title=f'{ticker.upper()} Turtle Signal {tt_col}' if ticker else f'Turtle Signal {tt_col}'
    )
    plt.show()
    _close_figure()

    return fig, ax


def plot_signal_ma(
    df: pd.DataFrame,
    st: int,
    mt: int,
    lt: int,
    ticker: str = ""
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot SMA and EMA crossover signals.

    Args:
        df: DataFrame with close and moving average signal columns
        st: Short-term period
        mt: Medium-term period
        lt: Long-term period
        ticker: Ticker symbol for title (optional)

    Returns:
        Tuple of (Figure, (Axes1, Axes2)) objects

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing or parameters invalid
    """
    func_name = "plot_signal_ma"
    st = _validate_positive_int(st, "st", func_name)
    mt = _validate_positive_int(mt, "mt", func_name)
    lt = _validate_positive_int(lt, "lt", func_name)

    sma_col = f'sma_{st}{mt}{lt}'
    ema_col = f'ema_{st}{mt}{lt}'
    required = [CLOSE_COLUMN, sma_col, ema_col]
    df = _validate_dataframe(df, required, func_name)

    logger.debug("Plotting MA signals, periods=%d/%d/%d", st, mt, lt)

    title_prefix = f'{ticker.upper()} ' if ticker else ''

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DEFAULT_FIG_WIDTH, 10))

    df[[CLOSE_COLUMN, sma_col]].plot(
        ax=ax1,
        secondary_y=sma_col,
        style=['k', 'b-.'],
        title=f'{title_prefix}SMA Signal {sma_col}'
    )

    df[[CLOSE_COLUMN, ema_col]].plot(
        ax=ax2,
        secondary_y=ema_col,
        style=['k', 'b-.'],
        title=f'{title_prefix}EMA Signal {ema_col}'
    )

    plt.tight_layout()
    plt.show()
    _close_figure()

    return fig, (ax1, ax2)


def plot_signal_abs(
    df: pd.DataFrame,
    ticker: str
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot absolute price signal with floor/ceiling indicators.

    Args:
        df: DataFrame with price and signal columns
        ticker: Ticker symbol for title

    Returns:
        Tuple of (Figure, Axes) objects

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing
    """
    func_name = "plot_signal_abs"
    ticker = _validate_ticker(ticker, func_name)

    plot_cols = [CLOSE_COLUMN, 'hi3', 'lo3', 'clg', 'flr', 'rg_ch', 'rg']
    plot_style = ['k', 'ro', 'go', 'kv', 'k^', 'b:', 'b--']
    y2_cols = ['rg']

    df = _validate_dataframe(df, plot_cols, func_name)

    logger.debug("Plotting absolute signal for %s", ticker)

    fig, ax = plt.subplots(figsize=(15, 8))
    df[plot_cols].plot(
        ax=ax,
        secondary_y=y2_cols,
        title=f'{ticker.upper()} Absolute',
        style=plot_style
    )
    plt.show()
    _close_figure()

    return fig, ax


def plot_signal_rel(
    df: pd.DataFrame,
    ticker: str
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot relative price signal with floor/ceiling indicators.

    Args:
        df: DataFrame with relative price and signal columns
        ticker: Ticker symbol for title

    Returns:
        Tuple of (Figure, Axes) objects

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing
    """
    func_name = "plot_signal_rel"
    ticker = _validate_ticker(ticker, func_name)

    plot_cols = [RCLOSE_COLUMN, 'rh3', 'rl3', 'rclg', 'rflr', 'rrg_ch', 'rrg']
    plot_style = ['grey', 'ro', 'go', 'yv', 'y^', 'm:', 'm--']
    y2_cols = ['rrg']

    df = _validate_dataframe(df, plot_cols, func_name)

    logger.debug("Plotting relative signal for %s", ticker)

    fig, ax = plt.subplots(figsize=(15, 8))
    df[plot_cols].plot(
        ax=ax,
        secondary_y=y2_cols,
        title=f'{ticker.upper()} Relative',
        style=plot_style
    )
    plt.show()
    _close_figure()

    return fig, ax


def plot_regime_abs(
    df: pd.DataFrame,
    ticker: str
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot absolute regime detection.

    Args:
        df: DataFrame with price and regime columns
        ticker: Ticker symbol for title

    Returns:
        Tuple of (Figure, Axes) objects

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing

    Note:
        This function plots regime data. The full graph_regime_combo
        visualization requires additional implementation.
    """
    func_name = "plot_regime_abs"
    ticker = _validate_ticker(ticker, func_name)

    # Required columns for regime plotting
    required = [CLOSE_COLUMN, 'rg', 'lo3', 'hi3', 'clg', 'flr', 'rg_ch']
    df = _validate_dataframe(df, required, func_name)

    logger.debug("Plotting absolute regime for %s", ticker)

    # Simplified regime plot (graph_regime_combo not available)
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    df[[CLOSE_COLUMN, 'rg']].plot(
        ax=ax,
        secondary_y=['rg'],
        style=['k', 'b--'],
        title=f'{ticker.upper()} Regime (Absolute)'
    )

    # Add floor/ceiling markers
    if 'clg' in df.columns and 'flr' in df.columns:
        ceiling_points = df[df['clg'].notna()]
        floor_points = df[df['flr'].notna()]
        if not ceiling_points.empty:
            ax.scatter(ceiling_points.index, ceiling_points[CLOSE_COLUMN],
                      marker='v', color='red', s=50, label='Ceiling', zorder=5)
        if not floor_points.empty:
            ax.scatter(floor_points.index, floor_points[CLOSE_COLUMN],
                      marker='^', color='green', s=50, label='Floor', zorder=5)
        ax.legend()

    plt.show()
    _close_figure()

    return fig, ax


def plot_regime_rel(
    df: pd.DataFrame,
    ticker: str
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot relative regime detection.

    Args:
        df: DataFrame with relative price and regime columns
        ticker: Ticker symbol for title

    Returns:
        Tuple of (Figure, Axes) objects

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing

    Note:
        This function plots regime data. The full graph_regime_combo
        visualization requires additional implementation.
    """
    func_name = "plot_regime_rel"
    ticker = _validate_ticker(ticker, func_name)

    # Required columns for regime plotting
    required = [RCLOSE_COLUMN, 'rrg', 'rl3', 'rh3', 'rclg', 'rflr', 'rrg_ch']
    df = _validate_dataframe(df, required, func_name)

    logger.debug("Plotting relative regime for %s", ticker)

    # Simplified regime plot (graph_regime_combo not available)
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    df[[RCLOSE_COLUMN, 'rrg']].plot(
        ax=ax,
        secondary_y=['rrg'],
        style=['grey', 'm--'],
        title=f'{ticker.upper()} Regime (Relative)'
    )

    # Add floor/ceiling markers
    if 'rclg' in df.columns and 'rflr' in df.columns:
        ceiling_points = df[df['rclg'].notna()]
        floor_points = df[df['rflr'].notna()]
        if not ceiling_points.empty:
            ax.scatter(ceiling_points.index, ceiling_points[RCLOSE_COLUMN],
                      marker='v', color='red', s=50, label='Ceiling', zorder=5)
        if not floor_points.empty:
            ax.scatter(floor_points.index, floor_points[RCLOSE_COLUMN],
                      marker='^', color='green', s=50, label='Floor', zorder=5)
        ax.legend()

    plt.show()
    _close_figure()

    return fig, ax


def plot_profit_loss(
    df: pd.DataFrame,
    ticker: str,
    method: str = ""
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot cumulative P&L and daily changes.

    Args:
        df: DataFrame with P&L columns
        ticker: Ticker symbol for title
        method: Method/strategy name for title

    Returns:
        Tuple of (Figure, Axes) objects

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing
    """
    func_name = "plot_profit_loss"
    ticker = _validate_ticker(ticker, func_name)

    required = ['tt_PL_cum', 'tt_chg1D']
    df = _validate_dataframe(df, required, func_name)

    logger.debug("Plotting P&L for %s", ticker)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    df[['tt_PL_cum', 'tt_chg1D']].plot(
        ax=ax,
        secondary_y=['tt_chg1D'],
        style=['b', 'c:'],
        title=f'{ticker} Daily P&L & Cumulative P&L {method}'
    )
    plt.show()
    _close_figure()

    return fig, ax


# Alias for backward compatibility
plot_PL = plot_profit_loss


def plot_price_signal_cumreturns(
    df: pd.DataFrame,
    ticker: str,
    signal: str,
    method: str = ""
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot price, signal, and cumulative returns.

    Args:
        df: DataFrame with price, signal, and return columns
        ticker: Ticker symbol for title
        signal: Signal column name
        method: Method/strategy name for title

    Returns:
        Tuple of (Figure, Axes) objects

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing
    """
    func_name = "plot_price_signal_cumreturns"
    ticker = _validate_ticker(ticker, func_name)

    required = [CLOSE_COLUMN, 'stop_loss', signal, 'tt_cumul']
    df = _validate_dataframe(df, required, func_name)

    logger.debug("Plotting price/signal/returns for %s", ticker)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    df[[CLOSE_COLUMN, 'stop_loss', signal, 'tt_cumul']].plot(
        ax=ax,
        secondary_y=[signal, 'tt_cumul'],
        style=['k', 'r--', 'b:', 'b'],
        title=f'{ticker} Close Price, signal, cumulative returns {method}'
    )
    plt.show()
    _close_figure()

    return fig, ax


def plot_equity_risk(
    df: pd.DataFrame,
    ticker: str,
    method: str = ""
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot equity risk metrics including drawdown and risk levels.

    Args:
        df: DataFrame with equity and risk columns
        ticker: Ticker symbol for title
        method: Method/strategy name for title

    Returns:
        Tuple of (Figure, (Axes1, Axes2)) objects

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing
    """
    func_name = "plot_equity_risk"
    ticker = _validate_ticker(ticker, func_name)

    required1 = [CLOSE_COLUMN, 'peak_eqty', 'tolerance', 'drawdown']
    required2 = [CLOSE_COLUMN, 'peak_eqty', 'tolerance',
                 'constant_risk', 'convex_risk', 'concave_risk']
    required = list(set(required1 + required2))
    df = _validate_dataframe(df, required, func_name)

    logger.debug("Plotting equity risk for %s", ticker)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DEFAULT_FIG_WIDTH, 12))

    df[[CLOSE_COLUMN, 'peak_eqty', 'tolerance', 'drawdown']].plot(
        ax=ax1,
        style=['k', 'g-.', 'r-.', 'm:'],
        secondary_y=['drawdown'],
        grid=True,
        title=f'{ticker} Drawdown {method}'
    )

    df[[CLOSE_COLUMN, 'peak_eqty', 'tolerance',
        'constant_risk', 'convex_risk', 'concave_risk']].plot(
        ax=ax2,
        grid=True,
        secondary_y=['constant_risk', 'convex_risk', 'concave_risk'],
        style=['k', 'g-.', 'r-.', 'b:', 'y-.', 'orange'],
        title=f'{ticker} Equity Risk {method}'
    )

    plt.tight_layout()
    plt.show()
    _close_figure()

    return fig, (ax1, ax2)


def plot_shares_signal(
    df: pd.DataFrame,
    ticker: str,
    signal: str,
    method: str = ""
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot share quantities for different position sizing methods.

    Args:
        df: DataFrame with share quantity columns
        ticker: Ticker symbol for title
        signal: Signal column name
        method: Method/strategy name for title

    Returns:
        Tuple of (Figure, Axes) objects

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing
    """
    func_name = "plot_shares_signal"
    ticker = _validate_ticker(ticker, func_name)

    required = ['shs_eql', 'shs_fxd', 'shs_ccv', 'shs_cvx', signal]
    df = _validate_dataframe(df, required, func_name)

    logger.debug("Plotting shares signal for %s", ticker)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    df[['shs_eql', 'shs_fxd', 'shs_ccv', 'shs_cvx', signal]].plot(
        ax=ax,
        secondary_y=[signal],
        style=['k', 'r--', 'b:', 'b', 'y'],
        title=f'{ticker} Shares {method}'
    )
    plt.show()
    _close_figure()

    return fig, ax


def plot_equity_amount(
    df: pd.DataFrame,
    ticker: str,
    method: str = ""
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot equity curves for different position sizing strategies.

    Args:
        df: DataFrame with equity columns
        ticker: Ticker symbol for title
        method: Method/strategy name for title

    Returns:
        Tuple of (Figure, Axes) objects

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing
    """
    func_name = "plot_equity_amount"
    ticker = _validate_ticker(ticker, func_name)

    required = ['constant', 'concave', 'convex', 'equal_weight', 'tt_PL_cum_fx']
    df = _validate_dataframe(df, required, func_name)

    logger.debug("Plotting equity amount for %s", ticker)

    fig, ax = plt.subplots(figsize=(DEFAULT_FIG_WIDTH, 10))
    df[['constant', 'concave', 'convex', 'equal_weight', 'tt_PL_cum_fx']].plot(
        ax=ax,
        grid=True,
        style=['y.-', 'm--', 'g-.', 'b:', 'b'],
        secondary_y='tt_PL_cum_fx',
        title=f'{ticker} Cumulative P&L, concave, convex, constant equity at risk, equal weight {method}'
    )
    plt.show()
    _close_figure()

    return fig, ax
