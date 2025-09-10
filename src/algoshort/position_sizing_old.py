import pandas as pd
import numpy as np

def pyramid(position, root=2): 
    ''' 
    position is the number of positions 
    power is root n.  

    Conservative = 1, aggressive = position, default = 2 
    ''' 
    return 1 / (1+position) ** (1/root) 
 
def amortized_weight(raw_weight, amortization): 
    ''' 
    raw_weight is the initial position size 
    amortization is pyramid(position,root=2) 
    ''' 
    return raw_weight * amortization 

def eqty_risk_shares(px,sl,eqty,risk,fx,lot):
    r = sl - px
    if fx > 0:
        budget = eqty * risk * fx
    else:
        budget = eqty * risk
    shares = round(budget // (r *lot) * lot,0)
#     print(r,budget,round(budget/r,0))
    return shares

def peak_equity(eqty):
    '''
    calculates peak equity
    
    '''
    equity = pd.Series(eqty)
    peak = equity.expanding().max()
    return peak


def risk_budget(eqty,appetite,fx):
    '''
    eqty: eqty to be deployed. Deault at peak equity
    risk_appetite: risk to allocate to  
    fx: currency conversion
    '''
    budget = peak_equity(eqty) * appetite * fx
    return budget

def risk_unit(price,stop_loss):
    '''
    Van Tharp's R: distance to stop loss in $
    '''
    r = price - stop_loss
    return r 

def shares_roundlot(budget,fx,r,lot):
    fx_budget = fx * budget
    rounded_shares = fx_budget // (r * lot)
    shares = rounded_shares * lot
    return shares

def target_price(price, stop_loss, r_multiplier):
    r = price - stop_loss
    return price + r * r_multiplier

def partial_exit(qty, r_multiplier):
    if (qty * r_multiplier)!= 0:
        fraction = qty / r_multiplier
    else:
        fraction = 0
    return fraction

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

def concave(ddr, floor):
    '''
    For demo purpose only
    '''
    if floor == 0:
        concave = ddr
    else:
        concave = ddr ** (floor)
    return concave

# obtuse 
def convex(ddr, floor):
    '''
    obtuse = 1 - acute
    '''
    if floor == 0:
        convex = ddr
    else:
        convex = ddr ** (1/floor)
    return convex