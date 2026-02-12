"""
algoshort - Algorithmic Trading Tools Package

This package provides tools for algorithmic trading including:
- Data handling (YFinance integration)
- OHLC processing
- Regime detection (Moving Average, Breakout, Turtle, Floor/Ceiling)
- Stop-loss calculations
- Position sizing
- Returns calculation
- Signal generation

Example:
    >>> from algoshort import RegimeDetector, YFinanceDataHandler
    >>> from algoshort import PositionSizing, StopLossCalculator
"""

# Regime detection (unified interface)
from algoshort.regimes import (
    RegimeDetector,
    MovingAverageCrossover,
    BreakoutRegime,
    FloorCeilingRegime,
)

# Data handling
from algoshort.yfinance_handler import YFinanceDataHandler

# Processing
from algoshort.ohlcprocessor import OHLCProcessor

# Trading utilities
from algoshort.stop_loss import StopLossCalculator
from algoshort.position_sizing import PositionSizing, run_position_sizing_parallel
from algoshort.returns import ReturnsCalculator

# Trading summary
from algoshort.trading_summary import (
    get_trading_summary,
    get_multi_symbol_summary,
    print_trading_summary,
    print_multi_symbol_summary,
    create_trading_dashboard,
)

# Legacy regime modules (for backward compatibility)
from algoshort.regime_ma import TripleMACrossoverRegime
from algoshort.regime_bo import RegimeBO
from algoshort.regime_fc import RegimeFC


__version__ = "0.2.0"

__all__ = [
    # Unified regime detection
    "RegimeDetector",
    "MovingAverageCrossover",
    "BreakoutRegime",
    "FloorCeilingRegime",
    # Data handling
    "YFinanceDataHandler",
    # Processing
    "OHLCProcessor",
    # Trading utilities
    "StopLossCalculator",
    "PositionSizing",
    "run_position_sizing_parallel",
    "ReturnsCalculator",
    # Trading summary
    "get_trading_summary",
    "get_multi_symbol_summary",
    "print_trading_summary",
    "print_multi_symbol_summary",
    "create_trading_dashboard",
    # Legacy (backward compatibility)
    "TripleMACrossoverRegime",
    "RegimeBO",
    "RegimeFC",
]
