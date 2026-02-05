import logging
import pandas as pd
import os
from typing import Any
from algoshort.regime_fc import RegimeFC
from algoshort.returns import ReturnsCalculator
from algoshort.stop_loss import StopLossCalculator   # your stop-loss module
# from algoshort.position_sizing import PositionSizing  # your position sizing module
from algoshort.utils import load_config
# from algoshort.optimizer import get_equity, StrategyOptimizer
import numpy as np
from algoshort.yfinance_handler import YFinanceDataHandler
from algoshort.ohlcprocessor import OHLCProcessor
from datetime import date
from algoshort.wrappers import generate_signals

handler = YFinanceDataHandler(cache_dir="./cache", enable_logging=False)
handler.download_data(['BC.MI', 'FTSEMIB.MI'], start='2016-01-01', end=date.today(), use_cache=True)
df = handler.get_ohlc_data('BC.MI')
df['fx'] = 1
df.set_index('date')
bmk = handler.get_ohlc_data('FTSEMIB.MI')
df.columns.name = None
bmk.columns.name = None
bmk.set_index('date')

from algoshort.ohlcprocessor import OHLCProcessor
processor = OHLCProcessor()
df = processor.calculate_relative_prices(
    stock_data= df,
    benchmark_data= bmk
    )

tt_search_space = {
    'fast': [20],
    'slow': [50]
}

bo_search_space = [100]

ma_search_space = {
    'short_ma': [50],
    'medium_ma': [100],
    'long_ma': [150]
}


df, signal_columns = generate_signals(
    df=df,
    tt_search_space=tt_search_space,
    bo_search_space=bo_search_space,
    ma_search_space=ma_search_space,
)
print(df.columns)

signal_columns = [x for x in signal_columns if x != "rrg"]
print(signal_columns)

from algoshort.combiner import SignalGridSearch
searcher = SignalGridSearch(
    df=df,
    available_signals=signal_columns,
    direction_col='rrg'
)

# Run with default settings (all cores, multiprocessing)
df = searcher.run_grid_search_parallel(
    allow_flips=True,
    require_regime_alignment=True
)

print(df.columns.tolist())