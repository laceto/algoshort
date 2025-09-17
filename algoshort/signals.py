import numpy as np
import pandas as pd
from algoshort.utils import lower_upper_OHLC
from algoshort.regime_bo import regime_breakout, regime_ema, regime_sma, turtle_trader

# def regime_sma(df,_c,st,lt):
#     '''
#     bull +1: sma_st >= sma_lt , bear -1: sma_st <= sma_lt
#     '''
#     sma_lt = df[_c].rolling(lt).mean()
#     sma_st = df[_c].rolling(st).mean()
#     rg_sma = np.sign(sma_st - sma_lt)
#     return rg_sma

# def regime_ema(df,_c,st,lt):
#     '''
#     bull +1: ema_st >= ema_lt , bear -1: ema_st <= ema_lt
#     '''
#     ema_st = df[_c].ewm(span=st,min_periods = st).mean()
#     ema_lt = df[_c].ewm(span=lt,min_periods = lt).mean()
#     rg_ema = np.sign(ema_st - ema_lt)
#     return rg_ema

# def turtle_trader(df, _h, _l, slow, fast):
#     '''
#     _slow: Long/Short direction
#     _fast: trailing stop loss
#     '''
#     _slow = regime_breakout(df,_h,_l,window = slow)
#     _fast = regime_breakout(df,_h,_l,window = fast)
#     turtle = pd. Series(index= df.index, 
#                         data = np.where(_slow == 1,np.where(_fast == 1,1,0), 
#                                 np.where(_slow == -1, np.where(_fast ==-1,-1,0),0)))
#     return turtle

# def regime_breakout(df,_h,_l,window):
#     hl =  np.where(df[_h] == df[_h].rolling(window).max(),1,
#                                 np.where(df[_l] == df[_l].rolling(window).min(), -1,np.nan))
#     roll_hl = pd.Series(index= df.index, data= hl).ffill()
#     return roll_hl

def signal_bo(df, window):
    
    df['hi_'+str(window)] = df['high'].rolling(window).max()
    df['lo_'+str(window)] = df['low'].rolling(window).min()
    df['bo_'+ str(window)]= regime_breakout(df= df,_h= 'high',_l= 'low',window= window)
    return df

def signal_rbo(df, window, relative=False):
    
    _o,_h,_l,_c = lower_upper_OHLC(df,relative = relative)
    
    prefix_h = 'hi_'
    prefix_l = 'lo_'
    prefix_bo = 'bo_'
    if relative:
        prefix_h = 'rhi_'
        prefix_l = 'rlo_'
        prefix_bo = 'rbo_'  
    
    df[prefix_h + str(window)] = df[_h].rolling(window).max()
    df[prefix_l + str(window)] = df[_l].rolling(window).min()
    df[prefix_bo + str(window)]= regime_breakout(df= df,_h= _h,_l= _l,window= window)
    return df

def signal_tt(df, slow, fast):
    
    _o,_h,_l,_c = lower_upper_OHLC(df,relative = False)

    df['bo_'+ str(slow)] = regime_breakout(df,_h,_l,window = slow)
    df['bo_'+ str(fast)] = regime_breakout(df,_h,_l,window = fast)
    df['turtle_'+ str(slow) + str(fast)] = turtle_trader(df, _h='high', _l='low', slow=slow, fast=fast) 
    
    return df

def signal_rtt(df, slow, fast, relative=False):
    
    _o,_h,_l,_c = lower_upper_OHLC(df,relative = relative)

    prefix_bo = 'tt_'
    if relative:
        prefix_bo = 'rtt_'  

    df[prefix_bo + str(slow)] = regime_breakout(df,_h,_l,window = slow)
    df[prefix_bo + str(fast)] = regime_breakout(df,_h,_l,window = fast)
    df[prefix_bo + str(slow) + str(fast)] = turtle_trader(df, _h='high', _l='low', slow=slow, fast=fast) 
    
    return df

def signal_sma(df, st, mt, lt):
    df['sma_' + str(st) + str(mt)] = regime_sma(df, _c='close', st= st, lt= mt)
    df['sma_' + str(mt) + str(lt)] = regime_sma(df, _c='close', st= mt, lt= lt)
    df['sma_' + str(st) + str(mt) + str(lt)] = df['sma_' + str(st) + str(mt)] * df['sma_' + str(mt) + str(lt)]
    
    return df

def signal_ema(df, st, mt, lt):
    df['ema_' + str(st) + str(mt)] = regime_ema(df, _c='close', st= st, lt= mt)
    df['ema_' + str(mt) + str(lt)] = regime_ema(df, _c='close', st= mt, lt= lt)
    df['ema_' + str(st) + str(mt) + str(lt)] = df['ema_' + str(st) + str(mt)] * df['ema_' + str(mt) + str(lt)]
    
    return df

def signal_rema(df, st, mt, lt, relative=False):  
    prefix = 'ema_'  
    _c = 'close'  
    if relative:  
        prefix = 'rema_'  
        _c = 'rclose'  
          
    df[prefix + str(st) + str(mt)] = regime_ema(df, _c=_c, st=st, lt=mt)  
    df[prefix + str(mt) + str(lt)] = regime_ema(df, _c=_c, st=mt, lt=lt)  
    df[prefix + str(st) + str(mt) + str(lt)] = df[prefix + str(st) + str(mt)] * df[prefix + str(mt) + str(lt)]  
      
    return df 

def signal_rsma(df, st, mt, lt, relative=False):  
    prefix = 'sma_'  
    _c = 'close'  
    if relative:  
        prefix = 'rsma_'  
        _c = 'rclose'  
          
    df[prefix + str(st) + str(mt)] = regime_sma(df, _c=_c, st=st, lt=mt)  
    df[prefix + str(mt) + str(lt)] = regime_sma(df, _c=_c, st=mt, lt=lt)  
    df[prefix + str(st) + str(mt) + str(lt)] = df[prefix + str(st) + str(mt)] * df[prefix + str(mt) + str(lt)]  
    
    return df
