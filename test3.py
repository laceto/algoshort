import logging
import pandas as pd
import os
from typing import Any
from algoshort.regime_fc import RegimeFC
from algoshort.returns import ReturnsCalculator
from algoshort.stop_loss import StopLossCalculator   # your stop-loss module
from algoshort.position_sizing import PositionSizing  # your position sizing module
from algoshort.utils import load_config
from algoshort.optimizer import get_equity, StrategyOptimizer
import numpy as np
from algoshort.yfinance_handler import YFinanceDataHandler
from algoshort.ohlcprocessor import OHLCProcessor
from datetime import date
from algoshort.regime_bo import RegimeBO
from algoshort.wrappers import calculate_return

handler = YFinanceDataHandler(cache_dir="./cache", enable_logging=False)
symbols = ['SPM.MI']
handler.download_data(symbols, start='2016-01-01', end=date.today(), use_cache=True)
df = handler.get_ohlc_data('SPM.MI')
df['fx'] = 1
df.set_index('date')


regime_bo = RegimeBO(ohlc_stock=df)
regime_bo.compute_regime(regime_type='turtle', fast_window=10,
                             window=50,
                             relative=False, inplace=True)

from algoshort.returns import ReturnsCalculator

ret = ReturnsCalculator(df)
df = ret.get_returns(df=df, signal='tt_5010', relative=False)

print(df.columns)

def risk_appetite(eqty, tolerance, mn, mx, span, shape):
    '''
    eqty: equity curve series
    tolerance: tolerance for drawdown (<0)
    mn: min risk
    mx: max risk
    span: exponential moving average to smoothe the risk_appetite
    shape: convex (>45 deg diagonal) = 1, concave (<diagonal) = -1, else: simple risk_appetite
    '''
    # drawdown rebased
    eqty = pd.Series(eqty)
    watermark = eqty.expanding().max() # all-time-high peak equity
    drawdown = eqty / watermark - 1 # drawdown from peak
    ddr = 1 - np.minimum(drawdown / tolerance,1) # drawdown rebased to tolerance from 0 to 1
    avg_ddr = ddr.ewm(span = span).mean() # span rebased drawdown
    
    # Shape of the curve
    if shape == 1: # 
        _power = mx/mn # convex 
    elif shape == -1 :
        _power = mn/mx # concave
    else:
        _power = 1 # raw, straight line
    ddr_power = avg_ddr ** _power # ddr 
    
    # mn + adjusted delta
    risk_appetite = mn + (mx - mn) * ddr_power 
    
    return risk_appetite

def eqty_risk_shares(px,sl,eqty,risk,fx,lot):
    r = sl - px
    if fx > 0:
        budget = eqty * risk * fx
    else:
        budget = eqty * risk
    shares = round(budget // (r *lot) * lot,0)
#     print(r,budget,round(budget/r,0))
    return shares

starting_capital = 1000000
lot = 100
mn = -0.0025
mx = -0.0075
avg = (mn + mx) / 2
tolerance= -0.1
equal_weight = 0.05
shs_fxd = shs_ccv = shs_cvx = shs_eql = 0
df.loc[df.index[0],'constant'] = df.loc[df.index[0],'concave'] = starting_capital
df.loc[df.index[0],'convex'] = df.loc[df.index[0],'equal_weight'] = starting_capital

# Initialize columns
df['ccv'] = 0.0
df['cvx'] = 0.0
df['shs_eql'] = 0.0
df['shs_fxd'] = 0.0
df['shs_ccv'] = 0.0
df['shs_cvx'] = 0.0

# Cache column references for faster access
equal_weight_col = df['equal_weight']
constant_col = df['constant']
concave_col = df['concave']
convex_col = df['convex']
tt_5010_chg1D_fx_col = df['tt_5010_chg1D_fx']
tt_5010_col = df['tt_5010']
close_col = df['close']

for i in range(1, len(df)):
    # Cache previous row values to avoid repeated lookups
    prev_equal_weight = df.at[i-1, 'equal_weight']
    prev_constant = df.at[i-1, 'constant']
    prev_concave = df.at[i-1, 'concave']
    prev_convex = df.at[i-1, 'convex']
    curr_chg = df.at[i, 'tt_5010_chg1D_fx']
    
    # Update equity columns using .at (faster than .loc for scalars)
    df.at[i, 'equal_weight'] = prev_equal_weight + curr_chg * shs_eql
    df.at[i, 'constant'] = prev_constant + curr_chg * shs_fxd
    df.at[i, 'concave'] = prev_concave + curr_chg * shs_ccv
    df.at[i, 'convex'] = prev_convex + curr_chg * shs_cvx
    
    # Calculate risk appetite values using iloc slicing (more efficient)
    ccv = risk_appetite(eqty=concave_col.iloc[:i], tolerance=tolerance, 
                        mn=mn, mx=mx, span=5, shape=-1)
    cvx = risk_appetite(eqty=convex_col.iloc[:i], tolerance=tolerance, 
                        mn=mn, mx=mx, span=5, shape=1)
    
    # Store risk appetite values
    ccv_val = ccv.iloc[-1]
    cvx_val = cvx.iloc[-1]
    df.at[i, 'ccv'] = ccv_val
    df.at[i, 'cvx'] = cvx_val

    # Check condition with cached values
    if (df.at[i-1, 'tt_5010'] == 0) and (df.at[i, 'tt_5010'] != 0):
        px = df.at[i, 'close']
        sl = px * 0.9
        fx = 1
        
        # Calculate equal weight shares
        shs_eql = (df.at[i, 'equal_weight'] * equal_weight * fx // (px * lot)) * lot
        
        if px != sl:
            # Calculate risk-based shares
            shs_fxd = eqty_risk_shares(px, sl, eqty=df.at[i, 'constant'],
                                        risk=avg, fx=fx, lot=100)
            shs_ccv = eqty_risk_shares(px, sl, eqty=df.at[i, 'concave'],
                                        risk=ccv_val, fx=fx, lot=100)
            shs_cvx = eqty_risk_shares(px, sl, eqty=df.at[i, 'convex'],
                                        risk=cvx_val, fx=fx, lot=100)
    
    # Store share values
    df.at[i, 'shs_eql'] = shs_eql
    df.at[i, 'shs_fxd'] = shs_fxd
    df.at[i, 'shs_ccv'] = shs_ccv
    df.at[i, 'shs_cvx'] = shs_cvx

# # Initialize the ccv, cvx, and share columns
# df['ccv'] = 0.0  # or np.nan if you prefer
# df['cvx'] = 0.0  # or np.nan if you prefer
# df['shs_eql'] = 0.0
# df['shs_fxd'] = 0.0
# df['shs_ccv'] = 0.0
# df['shs_cvx'] = 0.0

# for i in range(1, len(df)):
#     # Use .loc instead of .iat for assignment
#     df.loc[i, 'equal_weight'] = df.loc[i-1, 'equal_weight'] + df.loc[i, 'tt_5010_chg1D_fx'] * shs_eql
#     df.loc[i, 'constant'] = df.loc[i-1, 'constant'] + df.loc[i, 'tt_5010_chg1D_fx'] * shs_fxd
#     df.loc[i, 'concave'] = df.loc[i-1, 'concave'] + df.loc[i, 'tt_5010_chg1D_fx'] * shs_ccv
#     df.loc[i, 'convex'] = df.loc[i-1, 'convex'] + df.loc[i, 'tt_5010_chg1D_fx'] * shs_cvx
    
#     # Calculate risk appetite values
#     ccv = risk_appetite(eqty=df['concave'][:i], tolerance=tolerance, 
#                         mn=mn, mx=mx, span=5, shape=-1)
#     cvx = risk_appetite(eqty=df['convex'][:i], tolerance=tolerance, 
#                         mn=mn, mx=mx, span=5, shape=1)
    
#     # Store the latest ccv and cvx values in the DataFrame
#     df.loc[i, 'ccv'] = ccv.iloc[-1]
#     df.loc[i, 'cvx'] = cvx.iloc[-1]

#     if (df.loc[i-1, 'tt_5010'] == 0) & (df.loc[i, 'tt_5010'] != 0):
#         px = df.loc[i, 'close']
#         sl = df.loc[i, 'close'] * 0.9
#         # sl = df.loc[i, 'stop_loss']
#         fx = 1
#         shs_eql = (df.loc[i, 'equal_weight'] * equal_weight * fx // (px * lot)) * lot
#         if px != sl:
#             shs_fxd = eqty_risk_shares(px, sl, eqty=df.loc[i, 'constant'],
#                                         risk=avg, fx=fx, lot=100)
#             shs_ccv = eqty_risk_shares(px, sl, eqty=df.loc[i, 'concave'],
#                                         risk=df.loc[i, 'ccv'], fx=fx, lot=100)
#             shs_cvx = eqty_risk_shares(px, sl, eqty=df.loc[i, 'convex'],
#                                         risk=df.loc[i, 'cvx'], fx=fx, lot=100)
    
#     # Store the current share values in the DataFrame
#     df.loc[i, 'shs_eql'] = shs_eql
#     df.loc[i, 'shs_fxd'] = shs_fxd
#     df.loc[i, 'shs_ccv'] = shs_ccv
#     df.loc[i, 'shs_cvx'] = shs_cvx
            
# for i in range(1, len(df)):
#     # Use .loc instead of .iat for assignment to avoid chained assignment warnings
#     df.loc[i, 'equal_weight'] = df.loc[i-1, 'equal_weight'] + df.loc[i, 'tt_5010_chg1D_fx'] * shs_eql
#     df.loc[i, 'constant'] = df.loc[i-1, 'constant'] + df.loc[i, 'tt_5010_chg1D_fx'] * shs_fxd
#     df.loc[i, 'concave'] = df.loc[i-1, 'concave'] + df.loc[i, 'tt_5010_chg1D_fx'] * shs_ccv
#     df.loc[i, 'convex'] = df.loc[i-1, 'convex'] + df.loc[i, 'tt_5010_chg1D_fx'] * shs_cvx
    
#     ccv = risk_appetite(eqty=df['concave'][:i], tolerance=tolerance, 
#                         mn=mn, mx=mx, span=5, shape=-1)
#     cvx = risk_appetite(eqty=df['convex'][:i], tolerance=tolerance, 
#                         mn=mn, mx=mx, span=5, shape=1)

#     if (df.loc[i-1, 'tt_5010'] == 0) & (df.loc[i, 'tt_5010'] != 0):
#         px = df.loc[i, 'close']
#         sl = df.loc[i, 'close'] * 0.9
#         # sl = df.loc[i, 'stop_loss']
#         fx = 1
#         shs_eql = (df.loc[i, 'equal_weight'] * equal_weight * fx // (px * lot)) * lot
#         if px != sl:
#             shs_fxd = eqty_risk_shares(px, sl, eqty=df.loc[i, 'constant'],
#                                         risk=avg, fx=fx, lot=100)
#             shs_ccv = eqty_risk_shares(px, sl, eqty=df.loc[i, 'concave'],
#                                         risk=ccv.iloc[-1], fx=fx, lot=100)
#             shs_cvx = eqty_risk_shares(px, sl, eqty=df.loc[i, 'convex'],
#                                         risk=cvx.iloc[-1], fx=fx, lot=100)
            

print(df)