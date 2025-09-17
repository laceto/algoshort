import matplotlib.pyplot as plt

def plot_abs_rel(df, ticker, bm_name):
    
    # ohlc = ['open','high','low','close']
    # _o,_h,_l,_c = [ohlc[h] for h in range(len(ohlc))]

    # bm_col = 'close'
    # dgt = 5

    # df = relative(df,_o,_h,_l,_c, bm_df, bm_col, dgt, rebase=True)
    df = df.set_index('date')
    plot1 = df[['close','rclose']].plot(figsize=(20,8),grid=True, title= ticker +  ' Absolute vs relative to ' + bm_name + ' rebased' )
    plt.show(plot1)

def plot_signal_bo(df, window, ticker, relative):

    _o,_h,_l,_c = lower_upper_OHLC(df,relative = relative)
    
    prefix_h = 'hi_'
    prefix_l = 'lo_'
    prefix_bo = 'bo_'
    close = 'close'
    if relative:
        prefix_h = 'rhi_'
        prefix_l = 'rlo_'
        prefix_bo = 'rbo_'  
        close = 'rclose'
    df = df.set_index('date')
    df[[close, prefix_h + str(window), prefix_l + str(window), prefix_bo + str(window)]].plot(
        secondary_y= [prefix_bo + str(window)], figsize=(20,5), style=['k','g:','r:','b-.'], 
        title = str.upper(ticker)+' '+str(window) +' days high/low')
    plt.show()


def plot_signal_tt(df, fast, slow):
    
    rg_cols = ['turtle_'+ str(sselow)+str(fast)]

    df[['close','turtle_'+ str(slow)+str(fast)] ].plot(
        secondary_y= rg_cols,figsize=(20,5), style=['k','b-.'], 
        title = str.upper('')+' '+str(rg_cols))
    plt.show()  


def plot_signal_ma(df, st, mt, lt):

    df[['close','sma_'+ str(st) + str(mt) + str(lt)] ].plot(
        secondary_y= 'sma_'+ str(st) + str(mt) + str(lt),figsize=(20,5), style=['k','b-.'], 
        title = str.upper('')+' '+str(['sma_'+ str(st) + str(mt) + str(lt)]))
    
    df[['close','ema_'+ str(st) + str(mt) + str(lt)] ].plot(
        secondary_y= 'ema_'+ str(st) + str(mt) + str(lt),figsize=(20,5), style=['k','b-.'], 
        title = str.upper('')+' '+str(['ema_'+ str(st) + str(mt) + str(lt)]))
        
    plt.show() 


def plot_signal_abs(df, ticker):
    
    plot_abs_cols = ['close','hi3', 'lo3','clg','flr','rg_ch','rg']
    plot_abs_style = ['k', 'ro', 'go', 'kv', 'k^','b:','b--']
    y2_abs = ['rg']
    plot_rel_cols = ['rclose','rh3', 'rl3','rclg','rflr','rrg_ch','rrg']
    plot_rel_style = ['grey', 'ro', 'go', 'yv', 'y^','m:','m--']
    y2_rel = ['rrg']

    # df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    df[plot_abs_cols].plot(secondary_y= y2_abs,figsize=(15,8),
                title = str.upper(ticker)+ ' Absolute',# grid=True,
                style=plot_abs_style)
    plt.show() 

    
def plot_signal_rel(df, ticker):
    
    plot_abs_cols = ['close','hi3', 'lo3','clg','flr','rg_ch','rg']
    plot_abs_style = ['k', 'ro', 'go', 'kv', 'k^','b:','b--']
    y2_abs = ['rg']
    plot_rel_cols = ['rclose','rh3', 'rl3','rclg','rflr','rrg_ch','rrg']
    plot_rel_style = ['grey', 'ro', 'go', 'yv', 'y^','m:','m--']
    y2_rel = ['rrg']

    # df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    df[plot_rel_cols].plot(secondary_y=y2_rel,figsize=(15,8),
            title = str.upper(ticker)+ ' Relative',# grid=True,
            style=plot_rel_style)
    plt.show() 
    
def plot_regime_abs(df, ticker):
    
    ma_st = ma_mt = ma_lt = lt_lo = lt_hi = st_lo = st_hi = 0
    df = df.set_index('date')

    rg_combo = ['close','rg','lo3','hi3','lo3','hi3','clg','flr','rg_ch']
    _c,rg,lo,hi,slo,shi,clg,flr,rg_ch =[rg_combo[r] for r in range(len(rg_combo)) ]

    graph_regime_combo(ticker,df,_c,rg,lo,hi,slo,shi,clg,flr,rg_ch,ma_st,ma_mt,ma_lt,lt_lo,lt_hi,st_lo,st_hi)
    plt.show()

    
def plot_regime_rel(df, ticker):
    
    ma_st = ma_mt = ma_lt = lt_lo = lt_hi = st_lo = st_hi = 0
    df = df.set_index('date')
    rrg_combo = ['rclose','rrg','rl3','rh3','rl3','rh3','rclg','rflr','rrg_ch']
    _c,rg,lo,hi,slo,shi,clg,flr,rg_ch =[rrg_combo[r] for r in range(len(rrg_combo)) ]

    graph_regime_combo(ticker,df,_c,rg,lo,hi,slo,shi,clg,flr,rg_ch,ma_st,ma_mt,ma_lt,lt_lo,lt_hi,st_lo,st_hi)
    plt.show()

def plot_PL(df, ticker, m):
    
    df[['tt_PL_cum','tt_chg1D']].plot(secondary_y=['tt_chg1D'],figsize=(20,8),style= ['b','c:'],title= str(ticker) +' Daily P&L & Cumulative P&L ' + str(m))
    plt.show()
    
def plot_price_signal_cumreturns(df, ticker, signal, m):
    df[['close','stop_loss',signal,'tt_cumul']].plot(secondary_y=[signal,'tt_cumul'],figsize=(20,8),style= ['k','r--','b:','b'],
                                                     title= str(ticker) + ' Close Price, signal, cumulative returns ' + str(m))
    plt.show()

def plot_equity_risk(df, ticker, m):
    
    df[['close','peak_eqty','tolerance', 'drawdown'] ].plot(style = ['k','g-.','r-.','m:'] ,
            secondary_y=['drawdown'], figsize=(20,8),grid=True)
    
    df[['close', 'peak_eqty', 'tolerance',
        'constant_risk','convex_risk','concave_risk']].plot(figsize= (20,8),grid=True,
    secondary_y=['constant_risk','convex_risk','concave_risk'],
    style= ['k','g-.','r-.','b:','y-.', 'orange'], 
    title= str(ticker) + ' equity risk ' + str(m))
    
    plt.show()

def plot_shares_signal(df, ticker, signal, m):
    df[['shs_eql','shs_fxd','shs_ccv','shs_cvx', signal]].plot(secondary_y=[signal],figsize=(20,8),style= ['k','r--','b:','b', 'y'],
                                                         title= str(ticker)+' shares ' + str(m))
    plt.show()

def plot_equity_amount(df, ticker, m):
    # df[['constant','convex','concave','equal_weight']].plot(figsize=(20,8),style= ['k','r--','b:','b'],
    #                                                  title= str(ticker) + ' equity amount ' + str(m))
    
    df[['constant','concave','convex','equal_weight', 'tt_PL_cum_fx']].plot(
        figsize = (20,10), 
        grid=True,
        style=['y.-','m--','g-.','b:', 'b'],
        secondary_y='tt_PL_cum_fx', title= str(ticker) + ' cumulative P&L, concave, convex, constant equity at risk, equal weight ' + str(m))
    plt.show()
