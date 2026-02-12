"""
Trading readiness summary module for pre-market analysis.

This module provides functions to generate trading summaries showing
current positions, signals, stop losses, and position sizing recommendations.

Example:
    >>> from algoshort.trading_summary import get_trading_summary, print_trading_summary
    >>> summary = get_trading_summary(df, 'AAPL', signal_col='hybrid_signal')
    >>> print_trading_summary(summary)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'get_trading_summary',
    'get_multi_symbol_summary',
    'print_trading_summary',
    'print_multi_symbol_summary',
]


def get_trading_summary(
    df: pd.DataFrame,
    ticker: str,
    signal_col: str = 'hybrid_signal',
    stop_loss_col: str = 'stop_loss',
    close_col: str = 'close',
    position_cols: Optional[Dict[str, str]] = None,
    lookback: int = 5
) -> Dict[str, Any]:
    """
    Generate a trading readiness summary for a single symbol.

    Args:
        df: DataFrame with signal, price, and optional position sizing columns.
        ticker: Symbol/ticker name.
        signal_col: Name of the signal column (1=long, -1=short, 0=flat).
        stop_loss_col: Name of the stop loss column.
        close_col: Name of the close price column.
        position_cols: Dict mapping sizing method names to column names.
            Example: {'equal': 'shs_eql', 'fixed': 'shs_fxd'}
        lookback: Number of recent bars to include in history.

    Returns:
        Dict containing trading summary with keys:
        - ticker: Symbol name
        - timestamp: Current datetime
        - last_date: Last date in data
        - current_price: Latest close price
        - current_signal: Latest signal value
        - position_direction: 'LONG', 'SHORT', or 'FLAT'
        - trade_action: Recommended action for tomorrow
        - stop_loss: Current stop loss level
        - risk_pct: Percentage risk from entry to stop
        - position_sizes: Dict of position sizes by method
        - recent_history: Recent signal changes
        - signal_changed: Whether signal changed on last bar
    """
    if df.empty:
        raise ValueError(f"{ticker}: DataFrame cannot be empty")

    # Work with a copy
    df = df.copy()

    # Handle date index
    if 'date' in df.columns:
        df = df.set_index('date')

    # Validate required columns
    required = [close_col, signal_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: Missing required columns: {missing}")

    # Get latest values
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    current_price = float(latest[close_col])
    current_signal = int(latest[signal_col]) if pd.notna(latest[signal_col]) else 0
    prev_signal = int(prev[signal_col]) if pd.notna(prev[signal_col]) else 0

    # Determine position direction
    if current_signal == 1:
        position_direction = 'LONG'
    elif current_signal == -1:
        position_direction = 'SHORT'
    else:
        position_direction = 'FLAT'

    # Determine trade action
    signal_changed = current_signal != prev_signal
    if signal_changed:
        if prev_signal == 0 and current_signal == 1:
            trade_action = 'ENTER LONG'
        elif prev_signal == 0 and current_signal == -1:
            trade_action = 'ENTER SHORT'
        elif prev_signal == 1 and current_signal == 0:
            trade_action = 'EXIT LONG'
        elif prev_signal == -1 and current_signal == 0:
            trade_action = 'EXIT SHORT'
        elif prev_signal == 1 and current_signal == -1:
            trade_action = 'FLIP: LONG → SHORT'
        elif prev_signal == -1 and current_signal == 1:
            trade_action = 'FLIP: SHORT → LONG'
        else:
            trade_action = 'SIGNAL CHANGED'
    else:
        if current_signal == 1:
            trade_action = 'HOLD LONG'
        elif current_signal == -1:
            trade_action = 'HOLD SHORT'
        else:
            trade_action = 'STAY FLAT'

    # Get stop loss
    stop_loss = None
    risk_pct = None
    if stop_loss_col in df.columns and pd.notna(latest[stop_loss_col]):
        stop_loss = float(latest[stop_loss_col])
        if current_signal == 1 and stop_loss < current_price:
            risk_pct = ((current_price - stop_loss) / current_price) * 100
        elif current_signal == -1 and stop_loss > current_price:
            risk_pct = ((stop_loss - current_price) / current_price) * 100

    # Get position sizes
    position_sizes = {}
    if position_cols:
        for method_name, col_name in position_cols.items():
            if col_name in df.columns and pd.notna(latest[col_name]):
                position_sizes[method_name] = int(latest[col_name])

    # Check for common position sizing columns
    default_pos_cols = {
        'equal_weight': 'shs_eql',
        'fixed': 'shs_fxd',
        'concave': 'shs_ccv',
        'convex': 'shs_cvx',
    }
    for method_name, col_name in default_pos_cols.items():
        if col_name in df.columns and method_name not in position_sizes:
            if pd.notna(latest[col_name]):
                position_sizes[method_name] = int(latest[col_name])

    # Recent history
    recent_df = df.tail(lookback)
    recent_history = []
    for idx, row in recent_df.iterrows():
        sig = int(row[signal_col]) if pd.notna(row[signal_col]) else 0
        sl = float(row[stop_loss_col]) if stop_loss_col in df.columns and pd.notna(row[stop_loss_col]) else None
        recent_history.append({
            'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
            'close': float(row[close_col]),
            'signal': sig,
            'direction': 'LONG' if sig == 1 else 'SHORT' if sig == -1 else 'FLAT',
            'stop_loss': sl,
        })

    # Build summary
    summary = {
        'ticker': ticker,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'last_date': recent_history[-1]['date'] if recent_history else None,
        'current_price': current_price,
        'current_signal': current_signal,
        'position_direction': position_direction,
        'trade_action': trade_action,
        'signal_changed': signal_changed,
        'stop_loss': stop_loss,
        'risk_pct': risk_pct,
        'position_sizes': position_sizes,
        'recent_history': recent_history,
    }

    logger.info(f"{ticker}: {trade_action} at ${current_price:.2f}")

    return summary


def get_multi_symbol_summary(
    data_dict: Dict[str, pd.DataFrame],
    signal_col: str = 'hybrid_signal',
    stop_loss_col: str = 'stop_loss',
    close_col: str = 'close',
    position_cols: Optional[Dict[str, str]] = None,
    lookback: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate trading summaries for multiple symbols.

    Args:
        data_dict: Dict mapping ticker names to DataFrames.
        signal_col: Name of the signal column.
        stop_loss_col: Name of the stop loss column.
        close_col: Name of the close price column.
        position_cols: Dict mapping sizing method names to column names.
        lookback: Number of recent bars to include in history.

    Returns:
        List of trading summaries, one per symbol.
    """
    summaries = []
    for ticker, df in data_dict.items():
        try:
            summary = get_trading_summary(
                df=df,
                ticker=ticker,
                signal_col=signal_col,
                stop_loss_col=stop_loss_col,
                close_col=close_col,
                position_cols=position_cols,
                lookback=lookback,
            )
            summaries.append(summary)
        except Exception as e:
            logger.error(f"{ticker}: Error generating summary - {e}")
            summaries.append({
                'ticker': ticker,
                'error': str(e),
            })

    return summaries


def print_trading_summary(summary: Dict[str, Any], detailed: bool = True) -> None:
    """
    Print a formatted trading summary to console.

    Args:
        summary: Trading summary dict from get_trading_summary().
        detailed: If True, include recent history.
    """
    if 'error' in summary:
        print(f"\n{'='*60}")
        print(f"  {summary['ticker']}: ERROR - {summary['error']}")
        print(f"{'='*60}")
        return

    ticker = summary['ticker']
    direction = summary['position_direction']
    action = summary['trade_action']

    # Header with color-coding based on direction
    direction_symbol = '🟢' if direction == 'LONG' else '🔴' if direction == 'SHORT' else '⚪'

    print(f"\n{'='*60}")
    print(f"  {direction_symbol} {ticker} - TRADING SUMMARY")
    print(f"{'='*60}")
    print(f"  Generated: {summary['timestamp']}")
    print(f"  Last Data: {summary['last_date']}")
    print()

    # Current status
    print(f"  CURRENT STATUS")
    print(f"  {'-'*40}")
    print(f"  Price:      ${summary['current_price']:,.2f}")
    print(f"  Position:   {direction}")
    print(f"  Signal:     {summary['current_signal']}")
    print()

    # Trade action (highlighted if changed)
    if summary['signal_changed']:
        print(f"  ⚠️  ACTION REQUIRED: {action}")
    else:
        print(f"  Action:     {action}")
    print()

    # Stop loss and risk
    if summary['stop_loss']:
        print(f"  RISK MANAGEMENT")
        print(f"  {'-'*40}")
        print(f"  Stop Loss:  ${summary['stop_loss']:,.2f}")
        if summary['risk_pct']:
            print(f"  Risk:       {summary['risk_pct']:.2f}%")
        print()

    # Position sizes
    if summary['position_sizes']:
        print(f"  POSITION SIZING (shares)")
        print(f"  {'-'*40}")
        for method, shares in summary['position_sizes'].items():
            print(f"  {method:12s}: {shares:,} shares")
        print()

    # Recent history
    if detailed and summary['recent_history']:
        print(f"  RECENT HISTORY")
        print(f"  {'-'*40}")
        print(f"  {'Date':<12} {'Close':>10} {'Signal':>8} {'Direction':<8}")
        print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8}")
        for bar in summary['recent_history']:
            sl_str = f" SL:${bar['stop_loss']:.2f}" if bar['stop_loss'] else ""
            print(f"  {bar['date']:<12} ${bar['close']:>9,.2f} {bar['signal']:>8} {bar['direction']:<8}{sl_str}")

    print(f"{'='*60}\n")


def print_multi_symbol_summary(
    summaries: List[Dict[str, Any]],
    detailed: bool = False
) -> None:
    """
    Print a formatted multi-symbol trading summary.

    Args:
        summaries: List of trading summaries from get_multi_symbol_summary().
        detailed: If True, include recent history for each symbol.
    """
    print("\n" + "="*70)
    print("  📊 TRADING READINESS REPORT")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("="*70)

    # Quick overview table
    print("\n  SUMMARY TABLE")
    print("  " + "-"*66)
    print(f"  {'Ticker':<8} {'Price':>10} {'Position':<8} {'Action':<20} {'Stop':>10}")
    print("  " + "-"*66)

    for s in summaries:
        if 'error' in s:
            print(f"  {s['ticker']:<8} {'ERROR':<50}")
            continue

        direction_sym = '🟢' if s['position_direction'] == 'LONG' else '🔴' if s['position_direction'] == 'SHORT' else '⚪'
        action_flag = '⚠️' if s['signal_changed'] else '  '
        stop_str = f"${s['stop_loss']:,.2f}" if s['stop_loss'] else '-'

        print(f"  {s['ticker']:<8} ${s['current_price']:>9,.2f} {direction_sym}{s['position_direction']:<6} {action_flag}{s['trade_action']:<18} {stop_str:>10}")

    print("  " + "-"*66)

    # Action items (signals that changed)
    actions = [s for s in summaries if 'error' not in s and s.get('signal_changed')]
    if actions:
        print("\n  ⚠️  ACTION ITEMS FOR TOMORROW")
        print("  " + "-"*40)
        for s in actions:
            print(f"  • {s['ticker']}: {s['trade_action']}")
            if s['position_sizes']:
                for method, shares in list(s['position_sizes'].items())[:2]:
                    print(f"      {method}: {shares:,} shares")

    # Detailed view if requested
    if detailed:
        for s in summaries:
            print_trading_summary(s, detailed=True)

    print()


def create_trading_dashboard(
    data_dict: Dict[str, pd.DataFrame],
    signal_col: str = 'hybrid_signal',
    stop_loss_col: str = 'stop_loss',
    close_col: str = 'close',
) -> pd.DataFrame:
    """
    Create a DataFrame dashboard of trading statuses.

    Args:
        data_dict: Dict mapping ticker names to DataFrames.
        signal_col: Name of the signal column.
        stop_loss_col: Name of the stop loss column.
        close_col: Name of the close price column.

    Returns:
        DataFrame with one row per symbol showing trading status.
    """
    summaries = get_multi_symbol_summary(
        data_dict=data_dict,
        signal_col=signal_col,
        stop_loss_col=stop_loss_col,
        close_col=close_col,
        lookback=1,
    )

    rows = []
    for s in summaries:
        if 'error' in s:
            rows.append({
                'ticker': s['ticker'],
                'error': s['error'],
            })
            continue

        row = {
            'ticker': s['ticker'],
            'last_date': s['last_date'],
            'price': s['current_price'],
            'signal': s['current_signal'],
            'position': s['position_direction'],
            'action': s['trade_action'],
            'signal_changed': s['signal_changed'],
            'stop_loss': s['stop_loss'],
            'risk_pct': s['risk_pct'],
        }

        # Add position sizes
        for method, shares in s.get('position_sizes', {}).items():
            row[f'shares_{method}'] = shares

        rows.append(row)

    return pd.DataFrame(rows)
