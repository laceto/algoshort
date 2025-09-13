import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# ==============================================================================
# ORIGINAL (FLAWED) FUNCTIONS - DO NOT USE FOR TRADING
# ==============================================================================

def hilo_alternation_original(hilo, dist=None, hurdle=None):
    """
    Original, flawed version of the hilo_alternation function.
    
    This function has multiple issues including:
    - An unreliable while loop condition.
    - Incorrect logic for keeping extreme values.
    - Returning from inside the loop, preventing full execution.
    - Incorrect use of dist and hurdle parameters.
    """
    i = 0
    while (np.sign(hilo.shift(1)) == np.sign(hilo)).any():
        
        # This block contains contradictory logic
        hilo.loc[(np.sign(hilo.shift(1)) != np.sign(hilo)) & (hilo.shift(1)<0) & (np.abs(hilo.shift(1)) < np.abs(hilo))] = np.nan
        hilo.loc[(np.sign(hilo.shift(1)) != np.sign(hilo)) & (hilo.shift(1)>0) & (np.abs(hilo) < hilo.shift(1))] = np.nan
        
        # This block incorrectly handles consecutive swings
        hilo.loc[(np.sign(hilo.shift(1)) == np.sign(hilo)) & (hilo.shift(1) < hilo)] = np.nan
        hilo.loc[(np.sign(hilo.shift(-1)) == np.sign(hilo)) & (hilo.shift(-1) < hilo)] = np.nan
        
        if pd.notnull(dist):
            hilo.loc[(np.sign(hilo.shift(1)) != np.sign(hilo)) & (np.abs(hilo + hilo.shift(1)).div(dist, fill_value=1) < hurdle)] = np.nan
        
        hilo = hilo.dropna().copy()
        i += 1
        if i == 4: # Break to prevent infinite loop
            break
    return hilo

def historical_swings_original(df,_o,_h,_l,_c, dist= None, hurdle= None):
    """
    Original, flawed version of the historical_swings function.
    
    This function has issues including:
    - Incorrect logic for creating the hilo series.
    - Fails to capture the return value from hilo_alternation.
    - Incorrect reduction logic for subsequent loops.
    """
    reduction = df[[_o,_h,_l,_c]].copy()
    reduction['avg_px'] = round(reduction[[_h,_l,_c]].mean(axis=1),2)
    highs = reduction['avg_px'].values
    lows = - reduction['avg_px'].values
    reduction_target = len(reduction) // 100
    
    n = 0
    while len(reduction) >= reduction_target:
        highs_list = find_peaks(highs, distance = 1, width = 0)
        lows_list = find_peaks(lows, distance = 1, width = 0)
        
        # This line is incorrect and will cause an error or unexpected behavior
        try:
            hilo = reduction.iloc[lows_list[0]][_l].sub(reduction.iloc[highs_list[0]][_h],fill_value=0)
        except IndexError:
            print("Original function failed to find peaks and build hilo series.")
            return df # Exit early on failure

        # This line does not capture the return value, so the changes are lost
        hilo_alternation_original(hilo, dist= None, hurdle= None)
        reduction['hilo'] = hilo

        n += 1
        reduction[str(_h)[:2]+str(n)] = reduction.loc[reduction['hilo']<0 ,_h]
        reduction[str(_l)[:2]+str(n)] = reduction.loc[reduction['hilo']>0 ,_l]
        df[str(_h)[:2]+str(n)] = reduction.loc[reduction['hilo']<0 ,_h]
        df[str(_l)[:2]+str(n)] = reduction.loc[reduction['hilo']>0 ,_l]
        
        reduction = reduction.dropna(subset= ['hilo'])
        reduction.fillna(method='ffill', inplace = True)
        highs = reduction[str(_h)[:2]+str(n)].values
        lows = -reduction[str(_l)[:2]+str(n)].values
        
        if n >= 9:
            break
            
    return df

# ==============================================================================
# CORRECTED FUNCTIONS
# ==============================================================================

def hilo_alternation_corrected(hilo):
    """
    Corrected version of the hilo_alternation function.
    """
    hilo = hilo.dropna().copy()
    
    while True:
        initial_count = len(hilo)
        hilo.loc[(np.sign(hilo.shift(1)) == np.sign(hilo)) & (hilo.shift(1) < 0) & (hilo.shift(1) > hilo)] = np.nan
        hilo.loc[(np.sign(hilo.shift(1)) == np.sign(hilo)) & (hilo.shift(1) > 0) & (hilo.shift(1) < hilo)] = np.nan
        hilo = hilo.dropna()
        if len(hilo) == initial_count:
            break
            
    return hilo

def historical_swings_corrected(df, _h, _l, n_levels=5):
    """
    Corrected version of the historical_swings function.
    """
    reduction = df[[_h, _l]].copy()
    
    for n in range(1, n_levels + 1):
        high_indices, _ = find_peaks(reduction[_h].values)
        low_indices, _ = find_peaks(-reduction[_l].values)
        
        # Create a combined series of highs and lows with original prices
        hilo_series = pd.Series(dtype='float64')
        hilo_series = pd.concat([
            pd.Series(reduction.iloc[high_indices][_h].values, index=reduction.iloc[high_indices].index),
            pd.Series(reduction.iloc[low_indices][_l].values, index=reduction.iloc[low_indices].index)
        ]).sort_index()

        # Mark highs with a negative sign for alternation logic
        hilo_series.loc[hilo_series.index.isin(reduction.iloc[high_indices].index)] *= -1

        # Apply the corrected alternation logic
        hilo_series = hilo_alternation_corrected(hilo_series)
        
        # Populate the original and reduction dataframes
        high_column_name = f'h{n}'
        low_column_name = f'l{n}'
        
        # Use reindex to align the series with the original dataframe index
        highs = hilo_series.where(hilo_series < 0).reindex(df.index)
        lows = hilo_series.where(hilo_series > 0).reindex(df.index)
        
        # Assign the absolute values to the new columns
        df[high_column_name] = highs.abs()
        df[low_column_name] = lows

        # Reduce the reduction DataFrame for the next level
        reduction = df.loc[hilo_series.index, [_h, _l]].copy()
        
    return df

if __name__ == "__main__":
    # Create sample data with consecutive highs and lows to test the logic
    data = {'Open': [10, 11, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 12, 11, 10, 9, 8, 7, 8, 9],
            'High': [12, 13, 14, 12, 11, 10, 9, 10, 11, 12, 13, 14, 13, 12, 11, 10, 9, 8, 9, 10],
            'Low': [8, 9, 10, 9, 8, 7, 6, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6, 5, 6, 7],
            'Close': [11, 12, 13, 10, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 7, 8]}
    
    df = pd.DataFrame(data)

    # --- Run Original Functions ---
    print("--- Running Original (Flawed) Functions ---")
    df_original = df.copy()
    try:
        result_original = historical_swings_original(df_original, 'Open', 'High', 'Low', 'Close')
        print("Original Function Output:")
        print(result_original)
    except Exception as e:
        print(f"Original function execution failed with an error: {e}")
        print("This highlights the unreliability of the original code.")
    print("\n" + "="*50 + "\n")

    # --- Run Corrected Functions ---
    print("--- Running Corrected Functions ---")
    df_corrected = df.copy()
    result_corrected = historical_swings_corrected(df_corrected, 'High', 'Low', n_levels=2)
    print("Corrected Function Output:")
    print(result_corrected)
    print("\n" + "="*50 + "\n")

    # --- Comparison ---
    print("--- Comparison of Outputs ---")
    print("Original code often fails or produces incorrect output due to logical errors.")
    print("The corrected code, on the other hand, should produce a clean, logical")
    print("series of fractal swing levels. Notice how the 'h1' and 'l1' columns")
    print("in the corrected output show the first level of swings, which you can then")
    print("use for your trading strategy.")
    print("For example, the original code may not find any swings at all, whereas the corrected code does.")
