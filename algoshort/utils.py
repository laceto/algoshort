import re
import pandas as pd
import json
import os
import numpy as np

def lower_upper_OHLC(df,relative = False):
    if relative==True:
        rel = 'r'
    else:
        rel= ''      
    if 'Open' in df.columns:
        ohlc = [rel+'Open',rel+'High',rel+'Low',rel+'Close']       
    elif 'open' in df.columns:
        ohlc = [rel+'open',rel+'high',rel+'low',rel+'close']
        
    try:
        _o,_h,_l,_c = [ohlc[h] for h in range(len(ohlc))]
    except:
        _o=_h=_l=_c= np.nan
    return _o,_h,_l,_c

def  regime_args(df,lvl,relative= False):
    if ('Low' in df.columns) & (relative == False):
        reg_val = ['Lo1','Hi1','Lo'+str(lvl),'Hi'+str(lvl),'rg','clg','flr','rg_ch']
    elif ('low' in df.columns) & (relative == False):
        reg_val = ['lo1','hi1','lo'+str(lvl),'hi'+str(lvl),'rg','clg','flr','rg_ch']
    elif ('Low' in df.columns) & (relative == True):
        reg_val = ['rL1','rH1','rL'+str(lvl),'rH'+str(lvl),'rrg','rclg','rflr','rrg_ch']
    elif ('low' in df.columns) & (relative == True):
        reg_val = ['rl1','rh1','rl'+str(lvl),'rh'+str(lvl),'rrg','rclg','rflr','rrg_ch']
    
    try: 
        rt_lo,rt_hi,slo,shi,rg,clg,flr,rg_ch = [reg_val[s] for s in range(len(reg_val))]
    except:
        rt_lo=rt_hi=slo=shi=rg=clg=flr=rg_ch= np.nan
    return rt_lo,rt_hi,slo,shi,rg,clg,flr,rg_ch
  
### RELATIVE
def relative(df,_o,_h,_l,_c, bm_df, bm_col, dgt,rebase=True):
    '''
    df: df
    bm_df, bm_col: df benchmark dataframe & column name
    dgt: rounding decimal
    # start/end: string or offset
    rebase: boolean rebase to beginning or continuous series
    '''
    bm_df.rename(columns={bm_col:'bm'},inplace=True)

    # print(df)
    # print(bm_df)
    
    df = pd.merge(df, bm_df[['date', 'bm']],how='left', on='date') 
    # df = pd.merge(df, bm_df[['date', 'bm']],how='inner', on='date') 
    
    df['bmfx'] = round(df['bm'],dgt).ffill()
    if rebase == True:
        df['bmfx'] = df['bmfx'].div(df['bmfx'][0])
    else:
        df['bmfx'] = df['bmfx']


    # Divide absolute price by fxcy adjustment factor and rebase to first value
    df['r' + str(_o)] = round(df[_o].div(df['bmfx']),dgt)
    df['r' + str(_h)] = round(df[_h].div(df['bmfx']),dgt)
    df['r'+ str(_l)] = round(df[_l].div(df['bmfx']),dgt)
    df['r'+ str(_c)] = round(df[_c].div(df['bmfx']),dgt)
    df = df.drop(['bm','bmfx'],axis=1)
    
    return df


def read_csv(filename): 
    ''' 
    position is the number of positions 
    power is root n.  

    Conservative = 1, aggressive = position, default = 2 
    ''' 
    df = pd.read_csv(filename)
    df.columns = map(str.lower, df.columns)
    
    # Clean the file names  
    df = clean_column_names(df)  
    df['date'] = pd.to_datetime(df['date']) 
    
    return df

def read_xlsx(filename):      
    bm_df = pd.read_excel(filename)
    bm_df.columns= bm_df.columns.str.strip().str.lower()
    bm_df['date'] = pd.to_datetime(bm_df['date'])  
    bm_df = clean_column_names(bm_df)  
    bm_df['close'] = pd.to_numeric(bm_df['close'], errors='coerce')
    
    
    return bm_df


def clean_column_names(df):  
    """  
    Cleans the column names of a Pandas DataFrame by replacing whitespaces with underscores,  
    replacing one or more dots with underscores, converting names to lowercase,  
    removing special characters, and removing trailing underscores from the names.  
    """  
    # Regular expression pattern to match non-alphanumeric characters  
    pattern = re.compile(r'[^a-zA-Z0-9_]+')  
    new_columns = []  
    for col in df.columns:  
        # Replace whitespaces with underscores  
        col = col.replace(' ', '_')  
        # Replace one or more dots with underscores  
        col = re.sub(r'\.+', '_', col)  
        # Convert to lowercase  
        col = col.lower()  
        # Remove special characters  
        col = pattern.sub('', col)  
        # Remove trailing underscores  
        col = col.rstrip('_')  
        new_columns.append(col)  
    df.columns = new_columns  
    return df  




def load_config(config_path='config.json'):
    """
    Load JSON configuration with error handling and validation.
    
    Args:
        config_path (str): Path to JSON config file.
    
    Returns:
        dict: Full configuration dictionary.
    
    Raises:
        FileNotFoundError: If config file is missing.
        ValueError: If required keys are missing or invalid.
    """
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required sections
        required_sections = ['regimes', 'relative', 'stop_loss', 'returns', 'metrics', 'position_sizing', 'benchmark']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing section '{section}' in config file.")
        
        # Validate regime parameters
        if 'breakout' not in config['regimes'] or 'bo_window' not in config['regimes']['breakout']:
            raise ValueError("Missing 'breakout.bo_window' in config.")
        if 'turtle' not in config['regimes'] or 'slow_window' not in config['regimes']['turtle'] or 'fast_window' not in config['regimes']['turtle']:
            raise ValueError("Missing 'turtle.slow_window' or 'turtle.fast_window' in config.")
        if 'ma_crossover' not in config['regimes'] or 'short_window' not in config['regimes']['ma_crossover']:
            raise ValueError("Missing 'ma_crossover.short_window' in config.")
        
        # Validate parameter values
        if config['regimes']['turtle']['slow_window'] <= config['regimes']['turtle']['fast_window']:
            raise ValueError("turtle.slow_window must be greater than fast_window.")
        if config['regimes']['ma_crossover']['short_window'] >= config['regimes']['ma_crossover']['medium_window'] or \
           config['regimes']['ma_crossover']['medium_window'] >= config['regimes']['ma_crossover']['long_window']:
            raise ValueError("ma_crossover windows must satisfy short_window < medium_window < long_window.")
        if config['stop_loss']['atr_window'] <= 0 or config['stop_loss']['atr_multiplier'] <= 0:
            raise ValueError("stop_loss parameters must be positive.")
        if config['stop_loss']['retracement_level'] < 0 or config['stop_loss']['retracement_level'] > 1:
            raise ValueError("stop_loss.retracement_level must be between 0 and 1.")
        if config['stop_loss'].get('magnitude_level', 1) not in [1, 2, 3]:
            raise ValueError("stop_loss.magnitude_level must be 1, 2, or 3.")
        if config['relative']['dgt'] < 0:
            raise ValueError("relative.dgt must be non-negative.")
        if config['metrics']['risk_window'] <= 0 or config['metrics']['percentile'] <= 0 or config['metrics']['percentile'] >= 1:
            raise ValueError("metrics parameters must be valid.")
        
        return config
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading config {config_path}: {str(e)}")
    
def extract_signal_name(config_path='config.json'):
    """
    Extract signal column names based on config parameters.
    
    Args:
        config_path (str): Path to JSON config file.
    
    Returns:
        list: List of signal column names (e.g., ['rbo_150', 'rtt_5020', 'rsma_50100150', 'rema_50100150']).
    
    Raises:
        ValueError: If config is invalid or parameters are missing.
    """
    config = load_config(config_path)
    
    signal_names = []

    # Floor ceiling
    fc = 'rrg' if config['regimes']['breakout']['relative'] else 'rg'
    signal_names.append(fc)
    
    # Breakout signal name
    bo_window = config['regimes']['breakout']['bo_window']
    prefix_bo = 'rbo_' if config['regimes']['breakout']['relative'] else 'bo_'
    signal_names.append(f'{prefix_bo}{bo_window}')
    
    # Turtle signal name
    slow_window = config['regimes']['turtle']['slow_window']
    fast_window = config['regimes']['turtle']['fast_window']
    prefix_tt = 'rtt_' if config['regimes']['turtle']['relative'] else 'tt_'
    signal_names.append(f'{prefix_tt}{slow_window}{fast_window}')
    
    # MA Crossover signal names
    short_window = config['regimes']['ma_crossover']['short_window']
    medium_window = config['regimes']['ma_crossover']['medium_window']
    long_window = config['regimes']['ma_crossover']['long_window']
    ma_types = config['regimes']['ma_crossover'].get('ma_type', ['sma', 'ema'])
    prefix_ma = 'r' if config['regimes']['ma_crossover']['relative'] else ''
    for ma_type in ma_types:
        signal_names.append(f'{prefix_ma}{ma_type}_{short_window}{medium_window}{long_window}')
    
    return signal_names