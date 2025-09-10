import re
import pandas as pd

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