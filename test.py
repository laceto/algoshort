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
handler.download_data(['MONC.MI', 'FTSEMIB.MI'], start='2016-01-01', end=date.today(), use_cache=True)
# df = handler.get_data('MONC.MI')
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
regime_fc = RegimeFC(df)
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
    'fast': [10, 20],
    'slow': [50, 50]
}

for w_val, m_val in zip(*search_space.values()):
    print(f"Index Match -> Window: {w_val}, Multiplier: {m_val}")
    regime_bo.compute_regime(regime_type='turtle', fast_window=w_val,
                             window=m_val,
                             relative=True, inplace=True)

# Includes any column starting with 'rtt_' OR exactly matching 'rrg'
signal_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in ['rtt_'])]

# calculate return
df = calculate_return(df, config_path=config_path, signal_columns=signal_columns)



# # calculate pos size and equity curves
# pos = PositionSizing(df)
# df = pos.compare_position_sizing(df=df, signal=s, price_col=price_col, stop_loss_col=stop_loss_name, daily_change_col = change_name, inplace=False)
# print(df[['constant', 'concave', 'convex', 'equal_weight']].tail(1))

def get_equity(is_data, signal, price_col = 'close'):

    pos = PositionSizing(is_data)
    df = pos.compare_position_sizing(df=is_data, signal=signal, price_col=price_col, stop_loss_col=signal + '_stop_loss', daily_change_col = signal + '_chg1D_fx', inplace=False)
    metrics_df = df[[signal, signal + '_stop_loss', signal + '_chg1D_fx','constant', 'concave', 'convex', 'equal_weight']]
    # metrics_df = df[['constant', 'concave', 'convex', 'equal_weight']]
    row = metrics_df.iloc[-1].to_dict()
    return row


from algoshort.optimizer import StrategyOptimizer

# 1. Setup
calc = StopLossCalculator(df)
optimizer = StrategyOptimizer(df, calc, get_equity)

# 2. Run Rolling Walk-Forward
windows = [10]
multipliers = [1.5, 2.5]
oos_df, stability, param_history = optimizer.rolling_walk_forward(
    signal='rtt_5010', 
    close_col = 'close',
    windows=windows, 
    multipliers=multipliers,
    n_segments=5
)
print('oos')
print(oos_df)

print(param_history)

# print(stability)

# sens = optimizer.sensitivity_analysis(signal='rtt_5020', best_w=15, best_m=1.5, variance=0.2)
# print(sens)

# n_segments = 5
# segment_size = len(df) // (n_segments + 1)
# oos_results = []
# param_history = []
# data = df
# i = 3
# windows = [10]
# multipliers = [1.5, 2.5]

# is_data = data.iloc[i * segment_size : (i + 1) * segment_size]
# oos_data = data.iloc[(i + 1) * segment_size : (i + 2) * segment_size]

# calc = StopLossCalculator(is_data)
# optimizer = StrategyOptimizer(is_data, calc, get_equity)
# signal = 'rtt_5010'
# windows = [10,]
# multipliers = [1.5, 2.5]

# equity_by_sl = optimizer.run_grid_search(is_data=is_data, signal=signal, windows=windows, multipliers=multipliers)
# # print(f"--- is data: {equity_by_sl[signal_columns]}")
# # print(equity_by_sl)

# best_row = equity_by_sl.sort_values('convex', ascending=False).iloc[0]
# w_best, m_best = int(best_row['window']), best_row['multiplier']
# # w_best, m_best

# print(f"--- best window is: {w_best}")
# print(f"--- best multiplier is: {m_best}")

# # calc = StopLossCalculator(oos_data)
# price_col = 'close'
# calc.data = oos_data
# final_oos = calc.atr_stop_loss(signal, window=w_best, multiplier=m_best, price_col=price_col)

# print(get_equity(final_oos, signal='rtt_5010', price_col=price_col))