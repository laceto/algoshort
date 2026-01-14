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

# tt strategy grid search
regime_bo = RegimeBO(ohlc_stock=df)

search_space = {
    'fast': [10],
    'slow': [50]
}

for w_val, m_val in zip(*search_space.values()):
    print(f"Index Match -> Window: {w_val}, Multiplier: {m_val}")
    regime_bo.compute_regime(regime_type='turtle', fast_window=w_val,
                             window=m_val,
                             relative=True, inplace=True)

# Includes any column starting with 'rtt_' OR exactly matching 'rrg'
signal_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in ['rtt_'])]
signal_columns = ['rtt_5010']
# calculate return
df = calculate_return(df, config_path=config_path, signal_columns=signal_columns)

calc = StopLossCalculator(df)
price_col = 'close'
df = calc.atr_stop_loss('rtt_5010', window=10, multiplier=1.5, price_col=price_col)
pos = PositionSizing(df)
pos.compare_position_sizing(df=df, signal='rtt_5010', price_col='close', stop_loss_col='rtt_5010_stop_loss', daily_change_col = 'rtt_5010_chg1D_fx', inplace=True)
df.to_excel('output.xlsx')


# shs_cols = [
#     'shs_fxd',
#     'shs_ccv',
#     'shs_cvx',
#     'shs_eql'
# ]


# def get_equity(is_data, signal, i, price_col = 'close'):

#     pos = PositionSizing(is_data)
#     df = pos.compare_position_sizing(df=is_data, signal=signal, price_col=price_col, stop_loss_col=signal + '_stop_loss', daily_change_col = signal + '_chg1D_fx', inplace=False)
#     df.to_excel(f"{i}output.xlsx")
#     metrics_df = df[['constant', 'concave', 'convex', 'equal_weight']]
#     # metrics_df = df
#     row = metrics_df.iloc[-1].to_dict()
#     return row

# # def get_equity(is_data, signal, price_col = 'close'):

# #     pos = PositionSizing(is_data)
# #     df = pos.compare_position_sizing(df=is_data, signal=signal, price_col=price_col, stop_loss_col=signal + '_stop_loss', daily_change_col = signal + '_chg1D_fx', inplace=False)
# #     # metrics_df = df[[signal, signal + '_stop_loss', shs_cols,'constant', 'concave', 'convex', 'equal_weight']]
# #     # metrics_df = df[['constant', 'concave', 'convex', 'equal_weight']]
# #     # metrics_df = df
# #     # row = metrics_df.iloc[-1].to_dict()
# #     return df.to_dict()


# from algoshort.optimizer import StrategyOptimizer

# # 1. Setup
# calc = StopLossCalculator(df)
# optimizer = StrategyOptimizer(df, calc, get_equity)

# # 2. Run Rolling Walk-Forward
# windows = [10]
# multipliers = [1.5, 2.5]
# oos_df, stability, param_history = optimizer.rolling_walk_forward(
#     signal='rtt_5010', 
#     close_col = 'close',
#     windows=windows, 
#     multipliers=multipliers,
#     n_segments=5
# )
# print('oos')
# print(oos_df)

# # print(param_history)

# # print(stability)

# # sens = optimizer.sensitivity_analysis(signal='rtt_5020', best_w=15, best_m=1.5, variance=0.2)
# # print(sens)

