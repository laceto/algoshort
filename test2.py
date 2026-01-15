from datetime import date
import pandas as pd
import numpy as np
from algoshort.yfinance_handler import YFinanceDataHandler
from algoshort.ohlcprocessor import OHLCProcessor
from algoshort.regime_fc import RegimeFC
from algoshort.regime_bo import RegimeBO
from algoshort.utils import load_config
from algoshort.wrappers import calculate_return
from algoshort.stop_loss import StopLossCalculator
from algoshort.position_sizing import PositionSizing

config_path = './config.json'
config = load_config(config_path)

# download data
handler = YFinanceDataHandler(cache_dir="./cache")
# handler.download_data(['MONC.MI', 'FTSEMIB.MI'], start='2016-01-01', end=date.today(), use_cache=True)
handler.download_data(['MONC.MI', 'FTSEMIB.MI'], start='2021-01-07', end='2022-09-01', use_cache=False)
df = handler.get_ohlc_data('MONC.MI')
df['fx'] = 1
bmk = handler.get_ohlc_data('FTSEMIB.MI')

df.columns.name = None
bmk.columns.name = None

# get relative price
processor = OHLCProcessor()
df = processor.calculate_relative_prices(
    stock_data= df,
    benchmark_data= bmk
    )

# triage stock
import logging
regime_fc = RegimeFC(df, logging.WARNING)
df = regime_fc.compute_regime(
        relative = True,
        lvl = config['regimes']['floor_ceiling']['lvl'],
        vlty_n = config['regimes']['floor_ceiling']['vlty_n'],
        threshold = config['regimes']['floor_ceiling']['threshold'],
        dgt = config['regimes']['floor_ceiling']['dgt'],
        d_vol = config['regimes']['floor_ceiling']['d_vol'],
        dist_pct = config['regimes']['floor_ceiling']['dist_pct'],
        retrace_pct = config['regimes']['floor_ceiling']['retrace_pct'],
        r_vol = config['regimes']['floor_ceiling']['r_vol']
    )

regime_bo = RegimeBO(ohlc_stock=df)
regime_bo.compute_regime(regime_type='turtle', fast_window=10,
                             window=50,
                             relative=True, inplace=True)

df = calculate_return(df, config_path=config_path, signal_columns=['rtt_5010'])

from algoshort.optimizer import StrategyOptimizer

def get_equity(is_data, signal, i, price_col = 'close'):

    pos = PositionSizing(is_data)
    df = pos.compare_position_sizing(df=is_data, signal=signal, price_col=price_col, stop_loss_col=signal + '_stop_loss', daily_change_col = signal + '_chg1D_fx', inplace=False)
    # df.to_excel(f"{i}output.xlsx")
    metrics_df = df[['constant', 'concave', 'convex', 'equal_weight']]
    # metrics_df = df
    row = metrics_df.iloc[-1].to_dict()
    return row

calc = StopLossCalculator(df)
optimizer = StrategyOptimizer(df, calc, get_equity)

# param_grid = {"window": [10, 20, 30, 50]}
param_grid={"window": [10,14,20,30], "multiplier": [1.5,2.0,2.5,3.0,4.0]}

print(param_grid)
print({k: type(v[0]) for k,v in param_grid.items() if v})

results = optimizer.rolling_walk_forward(
    signals=["rtt_5010"],
    stop_method="atr",
    param_grid=param_grid,
    n_jobs=-1,
)

print(pd.DataFrame(results))



# results = optimizer.rolling_walk_forward(
#     signals="position_signal_v1",
#     stop_method="breakout_channel",
#     param_grid={"window": [10, 20, 30, 50]},
#     n_jobs=-1,
# )

# # Multiple signals
# multi_results = optimizer.rolling_walk_forward(
#     signals=["signal_regime_bull", "signal_regime_all", "signal_ma_cross", "signal_rsi_oversold"],
#     stop_method="atr",
#     param_grid={"window": [10,14,20], "multiplier": [1.8, 2.2, 2.8, 3.5]},
#     n_jobs=-1,
#     verbose=True,
# )

# # Quick comparison table
# comparison = optimizer.compare_signals(
#     signals=["signal_v1", "signal_v2", "signal_v3"],
#     stop_method="breakout_channel",
#     param_grid={"window": range(15, 61, 5)},
#     n_jobs=-1,
# )
# print(comparison[["convex", "sharpe", "n_segments_valid"]])