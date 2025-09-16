import numpy as np
import pandas as pd
import logging
from scipy.signal import find_peaks
from typing import List, Tuple, Optional, Union, Dict, Any

class Regime_fc:
    """
    Floor/Ceiling Methodology for Swing Analysis
    
    This class contains seven core methods for swing detection and cleanup:
    1. hilo_alternation: Reduces data to alternating highs and lows
    2. historical_swings: Creates multiple levels of swing analysis  
    3. cleanup_latest_swing: Eliminates false positives from latest swings
    4. latest_swing_variables: Instantiates arguments for the latest swing
    5. test_distance: Tests sufficient distance from the last swing
    6. average_true_range: Calculates the Average True Range (ATR) for volatility
    7. retest_swing: Identifies swings based on retest logic
    8. retracement_swing: Identifies swings based on retracement logic
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize the Floor/Ceiling swing analyzer
        
        Parameters:
        -----------
        log_level : int, default=logging.INFO
            Logging level for the class operations
        """
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info("Initialized regime_fc swing analyzer")
    
    def hilo_alternation(self, 
                        hilo: pd.Series, 
                        dist: Optional[pd.Series] = None, 
                        hurdle: Optional[float] = None) -> pd.Series:
        """
        Reduces a series to a succession of highs and lows by eliminating consecutive 
        same-side extremes and keeping only the most extreme values.
        
        This method eliminates same-side consecutive highs and lows where:
        - Highs are assigned a minus sign 
        - Lows are assigned a positive sign
        - The most extreme value is kept when duplicates exist
        
        Parameters:
        -----------
        hilo : pd.Series
            Series containing high/low values with appropriate signs 
            (negative for highs, positive for lows)
        dist : pd.Series, optional
            Distance series for noise filtering (default: None)
        hurdle : float, optional
            Threshold for noise filtering based on distance (default: None)
                
        Returns:
        --------
        pd.Series
            Reduced series with alternating highs and lows
            
        Raises:
        -------
        ValueError
            If input data is invalid or empty
        TypeError
            If hilo is not a pandas Series
        """
        try:
            self.logger.debug(f"Starting hilo_alternation with {len(hilo)} data points")
            
            # Input validation
            if not isinstance(hilo, pd.Series):
                self.logger.error("hilo must be a pandas Series")
                raise TypeError("hilo must be a pandas Series")
            
            if hilo.empty:
                self.logger.warning("Empty hilo series provided")
                return pd.Series(dtype=float)
            
            # Create a copy to avoid modifying original data
            initial_length = len(hilo.dropna())
            hilo = hilo.copy()
            i = 0
            
            self.logger.debug(f"Initial non-NaN data points: {initial_length}")
            
            while (np.sign(hilo.shift(1)) == np.sign(hilo)).any(): # runs until duplicates are eliminated
                self.logger.debug(f"Iteration {i+1}: Processing {len(hilo.dropna())} points")
                
                # removes swing lows > swing highs
                condition1 = ((np.sign(hilo.shift(1)) != np.sign(hilo)) &  # hilo alternation test
                            (hilo.shift(1) < 0) &  # previous datapoint: high
                            (np.abs(hilo.shift(1)) < np.abs(hilo))) # high[-1] < low, eliminate low 
                hilo.loc[condition1] = np.nan
                self.logger.debug(f"Removed {condition1.sum()} swing lows > swing highs")
                
                # removes swing highs < swing lows
                condition2 = ((np.sign(hilo.shift(1)) != np.sign(hilo)) &  # hilo alternation
                            (hilo.shift(1) > 0) &  # previous swing: low
                            (np.abs(hilo) < hilo.shift(1))) # swing high < swing low[-1]
                hilo.loc[condition2] = np.nan
                self.logger.debug(f"Removed {condition2.sum()} swing highs < swing lows")
                
                # alternation test: removes duplicate swings & keep extremes
                condition3 = ((np.sign(hilo.shift(1)) == np.sign(hilo)) & # same sign
                            (hilo.shift(1) < hilo)) # keep lower one
                hilo.loc[condition3] = np.nan
                self.logger.debug(f"Removed {condition3.sum()} backward duplicate swings")
                
                condition4 = ((np.sign(hilo.shift(-1)) == np.sign(hilo)) & # same sign, forward looking 
                            (hilo.shift(-1) < hilo)) # keep forward one
                hilo.loc[condition4] = np.nan
                self.logger.debug(f"Removed {condition4.sum()} forward duplicate swings")
                
                # removes noisy swings: distance test
                if pd.notnull(dist).any() and hurdle is not None:
                    try:
                        distance_condition = ((np.sign(hilo.shift(1)) != np.sign(hilo)) &
                                            (np.abs(hilo + hilo.shift(1)).div(dist, fill_value=1) < hurdle))
                        hilo.loc[distance_condition] = np.nan
                        self.logger.debug(f"Applied distance filtering, removed {distance_condition.sum()} noisy swings with hurdle={hurdle}")
                    except Exception as e:
                        self.logger.warning(f"Distance filtering failed: {e}")
                
                # reduce hilo after each pass
                hilo = hilo.dropna().copy() 
                i += 1
                if i == 4: # breaks infinite loop
                    self.logger.warning("Maximum iterations (4) reached in hilo_alternation")
                    break
            
            final_length = len(hilo)
            reduction_ratio = (initial_length - final_length) / initial_length * 100 if initial_length > 0 else 0
            self.logger.info(f"hilo_alternation completed: reduced from {initial_length} to {final_length} "
                            f"points ({reduction_ratio:.1f}% reduction) in {i} iterations")
            
            return hilo
        
        except Exception as e:
            self.logger.error(f"Error in hilo_alternation: {e}")
            raise
    
    def historical_swings(self, 
                        df: pd.DataFrame,
                        _o: str = 'open',
                        _h: str = 'high', 
                        _l: str = 'low',
                        _c: str = 'close',
                        dist: Optional[pd.Series] = None,
                        hurdle: Optional[float] = None) -> pd.DataFrame:
        """
        Creates multiple levels of swing highs and lows using fractal analysis.
        
        This method implements a fractal approach to swing detection by:
        1. Finding peaks in average price data
        2. Creating alternating high/low sequences
        3. Iteratively reducing the dataset to find higher-level swings
        4. Populating multiple swing level columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing OHLC price data
        _o : str, default='open'
            Name of the open price column
        _h : str, default='high'
            Name of the high price column
        _l : str, default='low'
            Name of the low price column
        _c : str, default='close'
            Name of the close price column
        dist : pd.Series, optional
            Distance series for noise filtering in hilo_alternation (default: None)
        hurdle : float, optional
            Threshold for noise filtering (default: None)
            
        Returns:
        --------
        pd.DataFrame
            Original DataFrame with additional swing level columns (hi1, lo1, hi2, lo2, etc.)
            
        Raises:
        -------
        ValueError
            If required columns are missing or data is insufficient
        TypeError
            If df is not a pandas DataFrame
        """
        try:
            self.logger.info(f"Starting historical_swings analysis on {len(df)} data points")
            
            # Input validation
            if not isinstance(df, pd.DataFrame):
                self.logger.error("df must be a pandas DataFrame")
                raise TypeError("df must be a pandas DataFrame")
            
            required_cols = [_o, _h, _l, _c]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            if len(df) < 10:
                self.logger.error("Insufficient data: need at least 10 data points")
                raise ValueError("Insufficient data: need at least 10 data points")
            
            # Create working copy and calculate average price
            reduction = df[[_o, _h, _l, _c]].copy() 
            try:
                reduction['avg_px'] = round(reduction[[_h, _l, _c]].mean(axis=1), 2)
                self.logger.debug("Calculated average price column")
            except Exception as e:
                self.logger.error(f"Failed to calculate average price: {e}")
                raise ValueError(f"Error calculating average price: {e}")
            
            highs = reduction['avg_px'].values
            lows = -reduction['avg_px'].values
            reduction_target = len(reduction) // 100
            self.logger.debug(f"Reduction target set to {reduction_target}")
            
            n = 0
            swing_levels_info = []
            
            while len(reduction) >= reduction_target: 
                self.logger.debug(f"Level {n+1}: Processing {len(reduction)} data points")
                
                # Find peaks and valleys
                highs_list = find_peaks(highs, distance=1, width=0)
                lows_list = find_peaks(lows, distance=1, width=0)
                self.logger.debug(f"Found {len(highs_list[0])} highs and {len(lows_list[0])} lows")
                
                if len(highs_list[0]) == 0 or len(lows_list[0]) == 0:
                    self.logger.warning(f"No peaks found at level {n+1}, stopping")
                    break
                
                # Create hilo series
                hilo = reduction.iloc[lows_list[0]][_l].sub(reduction.iloc[highs_list[0]][_h], fill_value=0)
                self.logger.debug(f"Created hilo series with {len(hilo)} points")
                
                # Reduction dataframe and alternation loop
                hilo = self.hilo_alternation(hilo, dist=dist, hurdle=hurdle)
                reduction['hilo'] = hilo
                self.logger.debug(f"Applied hilo_alternation, reduced to {len(hilo)} points")
                
                if len(hilo) < 2:
                    self.logger.warning(f"Insufficient alternated points at level {n+1}, stopping")
                    break
                
                # Populate reduction df
                n += 1        
                high_col_name = f"{_h[:2]}{n}"
                low_col_name = f"{_l[:2]}{n}"
                reduction[high_col_name] = reduction.loc[reduction['hilo'] < 0, _h]
                reduction[low_col_name] = reduction.loc[reduction['hilo'] > 0, _l]
                self.logger.debug(f"Populated reduction columns: {high_col_name}, {low_col_name}")
                
                # Populate main dataframe
                df[high_col_name] = reduction.loc[reduction['hilo'] < 0, _h]
                df[low_col_name] = reduction.loc[reduction['hilo'] > 0, _l]
                self.logger.debug(f"Populated main dataframe columns: {high_col_name}, {low_col_name}")
                
                # Store level information
                swing_levels_info.append({
                    'level': n,
                    'data_points': len(reduction),
                    'swing_points': len(hilo),
                    'high_points': (reduction['hilo'] < 0).sum(),
                    'low_points': (reduction['hilo'] > 0).sum()
                })
                
                # Reduce reduction
                reduction = reduction.dropna(subset=['hilo'])
                reduction.fillna(method='ffill', inplace=True)
                self.logger.debug(f"Reduced dataset to {len(reduction)} points")
                
                highs = reduction[high_col_name].values
                lows = -reduction[low_col_name].values
                self.logger.debug(f"Prepared next iteration with {len(highs)} highs and {len(lows)} lows")
                
                if n >= 9:
                    self.logger.warning("Maximum swing levels (9) reached, stopping")
                    break
            
            self.logger.info(f"Historical swings analysis completed: created {n} levels, "
                            f"final reduction to {len(reduction)} points")
            
            for level_info in swing_levels_info:
                self.logger.debug(f"Level {level_info['level']}: {level_info['swing_points']} swings "
                                f"({level_info['high_points']} highs, {level_info['low_points']} lows)")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error in historical_swings: {e}")
            raise
    
    def cleanup_latest_swing(self, 
                           df: pd.DataFrame, 
                           shi: str, 
                           slo: str, 
                           rt_hi: str, 
                           rt_lo: str) -> pd.DataFrame:
        """
        Removes false positives from the latest swing high and low.
        
        The function takes the following steps:
        1. Identifies the latest swing low and high
        2. Identifies the most recent swing
        3. If a false positive is detected, assigns NaN
        
        A false positive is identified when:
        - For swing highs: subsequent real-time high exceeds the swing high, or swing high < swing low
        - For swing lows: subsequent real-time low falls below the swing low, or swing high < swing low
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing swing and real-time price data
        shi : str
            Column name for swing highs
        slo : str
            Column name for swing lows  
        rt_hi : str
            Column name for real-time highs
        rt_lo : str
            Column name for real-time lows
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with false positive swings removed (set to NaN)
            
        Raises:
        -------
        ValueError
            If required columns are missing or no swing data found
        TypeError
            If df is not a pandas DataFrame
        """
        try:
            self.logger.debug("Starting cleanup_latest_swing analysis")
            
            # Input validation
            if not isinstance(df, pd.DataFrame):
                raise TypeError("df must be a pandas DataFrame")
            
            required_cols = [shi, slo, rt_hi, rt_lo]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Create working copy
            df_work = df.copy()
            
            # Check if there are any swing points
            shi_mask = pd.notnull(df_work[shi])
            slo_mask = pd.notnull(df_work[slo])
            
            if not shi_mask.any() and not slo_mask.any():
                self.logger.warning("No swing points found in data")
                return df_work
            
            initial_shi_count = shi_mask.sum()
            initial_slo_count = slo_mask.sum()
            
            # Process up to 2 iterations for cleanup
            for iteration in range(2):
                self.logger.debug(f"Cleanup iteration {iteration + 1}")
                
                # Recalculate masks after potential changes
                shi_mask = pd.notnull(df_work[shi])
                slo_mask = pd.notnull(df_work[slo])
                
                if not shi_mask.any() and not slo_mask.any():
                    self.logger.debug("No swing points remaining after cleanup")
                    break
                
                # Get latest swing high data
                shi_dt = None
                s_hi = None
                len_shi_dt = 0
                
                if shi_mask.any():
                    try:
                        shi_dt = df_work.loc[shi_mask, shi].index[-1]
                        s_hi = df_work.loc[shi_mask, shi].iloc[-1]
                        len_shi_dt = len(df_work[:shi_dt])
                        self.logger.debug(f"Latest swing high: {s_hi} at index {shi_dt} (position {len_shi_dt})")
                    except Exception as e:
                        self.logger.warning(f"Error getting latest swing high: {e}")
                
                # Get latest swing low data  
                slo_dt = None
                s_lo = None
                len_slo_dt = 0
                
                if slo_mask.any():
                    try:
                        slo_dt = df_work.loc[slo_mask, slo].index[-1]
                        s_lo = df_work.loc[slo_mask, slo].iloc[-1]
                        len_slo_dt = len(df_work[:slo_dt])
                        self.logger.debug(f"Latest swing low: {s_lo} at index {slo_dt} (position {len_slo_dt})")
                    except Exception as e:
                        self.logger.warning(f"Error getting latest swing low: {e}")
                
                # Reset false positives to np.nan
                false_positive_detected = False
                
                # Check swing high false positive
                if (len_shi_dt > len_slo_dt) and shi_dt is not None and s_hi is not None:
                    try:
                        # Get data from swing high date onwards
                        future_highs = df_work.loc[shi_dt:, rt_hi]
                        max_future_high = future_highs.max() if not future_highs.empty else s_hi
                        
                        # Check conditions for false positive
                        high_exceeded = max_future_high > s_hi
                        cross_condition = s_lo is not None and s_hi < s_lo
                        
                        if high_exceeded or cross_condition:
                            df_work.loc[shi_dt, shi] = np.nan
                            len_shi_dt = 0
                            false_positive_detected = True
                            self.logger.debug(f"Removed false positive swing high: high_exceeded={high_exceeded}, "
                                            f"cross_condition={cross_condition}")
                    except Exception as e:
                        self.logger.warning(f"Error checking swing high false positive: {e}")
                
                # Check swing low false positive  
                elif (len_slo_dt > len_shi_dt) and slo_dt is not None and s_lo is not None:
                    try:
                        # Get data from swing low date onwards
                        future_lows = df_work.loc[slo_dt:, rt_lo]
                        min_future_low = future_lows.min() if not future_lows.empty else s_lo
                        
                        # Check conditions for false positive
                        low_breached = min_future_low < s_lo
                        cross_condition = s_hi is not None and s_hi < s_lo
                        
                        if low_breached or cross_condition:
                            df_work.loc[slo_dt, slo] = np.nan
                            len_slo_dt = 0
                            false_positive_detected = True
                            self.logger.debug(f"Removed false positive swing low: low_breached={low_breached}, "
                                            f"cross_condition={cross_condition}")
                    except Exception as e:
                        self.logger.warning(f"Error checking swing low false positive: {e}")
                
                if not false_positive_detected:
                    self.logger.debug(f"No false positives detected in iteration {iteration + 1}")
                    break
            
            # Calculate cleanup statistics
            final_shi_count = pd.notnull(df_work[shi]).sum()
            final_slo_count = pd.notnull(df_work[slo]).sum()
            
            removed_shi = initial_shi_count - final_shi_count
            removed_slo = initial_slo_count - final_slo_count
            
            self.logger.info(f"Cleanup completed: removed {removed_shi} swing highs and {removed_slo} swing lows")
            self.logger.debug(f"Remaining: {final_shi_count} swing highs, {final_slo_count} swing lows")
            
            return df_work
            
        except Exception as e:
            self.logger.error(f"Error in cleanup_latest_swing: {e}")
            raise
    
    def latest_swing_variables(self, 
                             df: pd.DataFrame, 
                             shi: str, 
                             slo: str, 
                             rt_hi: str, 
                             rt_lo: str, 
                             _h: str, 
                             _l: str, 
                             _c: str) -> Tuple[int, float, pd.Timestamp, str, str, float, pd.Timestamp]:
        """
        Instantiates arguments for the latest swing analysis.
        
        This function calculates the variables used in subsequent swing-related functions:
        1. ud: Direction, up (+1) or down (-1)
        2. bs: Base, either swing low or high
        3. bs_dt: The swing date
        4. _rt: The series name used to detect swing (real-time high or low)
        5. _swg: The series to assign the value (shi for swing high, slo for swing low)
        6. hh_ll: Either the lowest low or highest high
        7. hh_ll_dt: The date of the highest high or lowest low
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing swing and real-time price data
        shi : str
            Column name for swing highs
        slo : str
            Column name for swing lows
        rt_hi : str
            Column name for real-time highs
        rt_lo : str
            Column name for real-time lows
        _h : str
            Column name for high prices
        _l : str
            Column name for low prices
        _c : str
            Column name for close prices
            
        Returns:
        --------
        Tuple[int, float, pd.Timestamp, str, str, float, pd.Timestamp]
            Tuple containing (ud, bs, bs_dt, _rt, _swg, hh_ll, hh_ll_dt)
            
        Raises:
        -------
        ValueError
            If required columns are missing or no swing data found
        TypeError
            If df is not a pandas DataFrame
        """
        try:
            self.logger.debug("Starting latest_swing_variables analysis")
            
            # Input validation
            if not isinstance(df, pd.DataFrame):
                raise TypeError("df must be a pandas DataFrame")
            
            required_cols = [shi, slo, rt_hi, rt_lo, _h, _l, _c]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Get latest swing high and low dates and values
            shi_mask = pd.notnull(df[shi])
            slo_mask = pd.notnull(df[slo])
            
            if not shi_mask.any() or not slo_mask.any():
                self.logger.warning("No swing points found in data")
                raise ValueError("No valid swing high or low data found")
            
            shi_dt = df.loc[shi_mask, shi].index[-1]
            slo_dt = df.loc[slo_mask, slo].index[-1]
            s_hi = df.loc[shi_mask, shi].iloc[-1]
            s_lo = df.loc[slo_mask, slo].iloc[-1]
            
            self.logger.debug(f"Latest swing high: {s_hi} at {shi_dt}")
            self.logger.debug(f"Latest swing low: {s_lo} at {slo_dt}")
            
            # Determine swing variables based on latest swing
            if slo_dt > shi_dt:
                swg_var = [
                    1,  # ud: up direction
                    s_lo,  # bs: swing low
                    slo_dt,  # bs_dt: swing low date
                    rt_lo,  # _rt: real-time low series
                    shi,  # _swg: swing high series
                    df.loc[slo_dt:, _h].max(),  # hh_ll: highest high
                    df.loc[slo_dt:, _h].idxmax()  # hh_ll_dt: date of highest high
                ]
            elif shi_dt > slo_dt:
                swg_var = [
                    -1,  # ud: down direction
                    s_hi,  # bs: swing high
                    shi_dt,  # bs_dt: swing high date
                    rt_hi,  # _rt: real-time high series
                    slo,  # _swg: swing low series
                    df.loc[shi_dt:, _l].min(),  # hh_ll: lowest low
                    df.loc[shi_dt:, _l].idxmin()  # hh_ll_dt: date of lowest low
                ]
            else:
                self.logger.warning("Swing high and low have the same date")
                swg_var = [0, np.nan, None, None, None, np.nan, None]
            
            # Unpack swing variables
            ud, bs, bs_dt, _rt, _swg, hh_ll, hh_ll_dt = swg_var
            
            self.logger.info(f"Latest swing variables: ud={ud}, bs={bs}, bs_dt={bs_dt}, "
                           f"_rt={_rt}, _swg={_swg}, hh_ll={hh_ll}, hh_ll_dt={hh_ll_dt}")
            
            return ud, bs, bs_dt, _rt, _swg, hh_ll, hh_ll_dt
            
        except Exception as e:
            self.logger.error(f"Error in latest_swing_variables: {e}")
            raise
    
    def test_distance(self, 
                     ud: int, 
                     bs: float, 
                     hh_ll: float, 
                     dist_vol: float, 
                     dist_pct: float) -> int:
        """
        Tests if the distance from the last swing is sufficient.
        
        The function performs two built-in tests with priority:
        1. Distance as a multiple of volatility (dist_vol)
        2. Distance as a fixed percentage (dist_pct)
        3. Default (returns 0 if neither condition is met)
        
        Parameters:
        -----------
        ud : int
            Direction of the swing (+1 for up, -1 for down)
        bs : float
            Base swing value (swing high or low)
        hh_ll : float
            Highest high or lowest low
        dist_vol : float
            Volatility-based distance threshold (multiple of ATR)
        dist_pct : float
            Percentage-based distance threshold
            
        Returns:
        --------
        int
            Result of the distance test (0 if not met, ud if met)
            
        Raises:
        -------
        ValueError
            If input parameters are invalid
        """
        try:
            self.logger.debug(f"Starting test_distance with ud={ud}, bs={bs}, hh_ll={hh_ll}, "
                            f"dist_vol={dist_vol}, dist_pct={dist_pct}")
            
            # Input validation
            if not isinstance(ud, int) or ud not in [-1, 0, 1]:
                raise ValueError("ud must be an integer in [-1, 0, 1]")
            if not all(isinstance(x, (int, float)) for x in [bs, hh_ll, dist_vol, dist_pct]):
                raise ValueError("bs, hh_ll, dist_vol, and dist_pct must be numeric")
            
            # Priority: 1. Volatility 2. Percentage 3. Default
            if dist_vol > 0:
                distance_test = np.sign(abs(hh_ll - bs) - dist_vol)
                self.logger.debug(f"Volatility-based distance test: {distance_test}")
            elif dist_pct > 0:
                distance_test = np.sign(abs(hh_ll / bs - 1) - dist_pct)
                self.logger.debug(f"Percentage-based distance test: {distance_test}")
            else:
                distance_test = np.sign(dist_pct)
                self.logger.debug(f"Default distance test: {distance_test}")
            
            result = int(max(distance_test, 0) * ud)
            self.logger.info(f"Distance test result: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in test_distance: {e}")
            raise
    
    def average_true_range(self, 
                          df: pd.DataFrame, 
                          _h: str, 
                          _l: str, 
                          _c: str, 
                          n: int) -> pd.Series:
        """
        Calculates the Average True Range (ATR) for volatility measurement.
        
        Reference: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing price data
        _h : str
            Column name for high prices
        _l : str
            Column name for low prices
        _c : str
            Column name for close prices
        n : int
            Window size for rolling mean
            
        Returns:
        --------
        pd.Series
            Series containing ATR values
            
        Raises:
        -------
        ValueError
            If required columns are missing or n is invalid
        TypeError
            If df is not a pandas DataFrame
        """
        try:
            self.logger.debug(f"Starting average_true_range with window={n}")
            
            # Input validation
            if not isinstance(df, pd.DataFrame):
                raise TypeError("df must be a pandas DataFrame")
            
            required_cols = [_h, _l, _c]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            if not isinstance(n, int) or n <= 0:
                raise ValueError("n must be a positive integer")
            
            # Calculate True Range
            tr = (df[_h].combine(df[_c].shift(), max) - 
                  df[_l].combine(df[_c].shift(), min))
            
            # Calculate ATR as rolling mean
            atr = tr.rolling(window=n).mean()
            
            self.logger.info(f"Calculated ATR with {len(atr)} values")
            return atr
            
        except Exception as e:
            self.logger.error(f"Error in average_true_range: {e}")
            raise
    
    def retest_swing(self, 
                    df: pd.DataFrame, 
                    _sign: int, 
                    _rt: str, 
                    hh_ll_dt: pd.Timestamp, 
                    hh_ll: float, 
                    _c: str, 
                    _swg: str) -> pd.DataFrame:
        """
        Identifies swings based on retest logic.
        
        For swing highs:
        1. Detect the highest high from swing low
        2. Identify the highest retest low
        3. When price closes below the highest retest low, assign swing high
        
        For swing lows:
        1. Detect the lowest low from swing high
        2. Identify the lowest retest high
        3. When price closes above the lowest retest high, assign swing low
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing price data
        _sign : int
            Direction of the swing (+1 for up, -1 for down)
        _rt : str
            Column name for real-time series (high or low)
        hh_ll_dt : pd.Timestamp
            Date of highest high or lowest low
        hh_ll : float
            Value of highest high or lowest low
        _c : str
            Column name for close prices
        _swg : str
            Column name for swing series (shi or slo)
            
        Returns:
        --------
        pd.DataFrame
            Updated DataFrame with swing assignments
            
        Raises:
        -------
        ValueError
            If required columns are missing or inputs are invalid
        TypeError
            If df is not a pandas DataFrame
        """
        try:
            self.logger.debug(f"Starting retest_swing with _sign={_sign}, hh_ll_dt={hh_ll_dt}, hh_ll={hh_ll}")
            
            # Input validation
            if not isinstance(df, pd.DataFrame):
                raise TypeError("df must be a pandas DataFrame")
            
            required_cols = [_rt, _c, _swg]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            if _sign not in [-1, 1]:
                raise ValueError("_sign must be +1 or -1")
            
            # Get real-time segment from hh_ll_dt
            rt_sgmt = df.loc[hh_ll_dt:, _rt]
            
            if rt_sgmt.count() > 0 and _sign != 0:
                if _sign == 1:
                    rt_list = [
                        rt_sgmt.idxmax(),  # Date of highest retest low
                        rt_sgmt.max(),  # Highest retest low
                        df.loc[rt_sgmt.idxmax():, _c].cummin()  # Cumulative min of close prices
                    ]
                elif _sign == -1:
                    rt_list = [
                        rt_sgmt.idxmin(),  # Date of lowest retest high
                        rt_sgmt.min(),  # Lowest retest high
                        df.loc[rt_sgmt.idxmin():, _c].cummax()  # Cumulative max of close prices
                    ]
                
                rt_dt, rt_hurdle, rt_px = rt_list
                self.logger.debug(f"Retest: rt_dt={rt_dt}, rt_hurdle={rt_hurdle}")
                
                # Assign retest hurdle to appropriate column
                if str(_c).startswith('r'):
                    df.loc[rt_dt, 'rrt'] = rt_hurdle
                else:
                    df.loc[rt_dt, 'rt'] = rt_hurdle
                
                # Check if price crosses the retest hurdle
                if (np.sign(rt_px - rt_hurdle) == -_sign).any():
                    df.at[hh_ll_dt, _swg] = hh_ll
                    self.logger.debug(f"Assigned swing at {hh_ll_dt} to {_swg}: {hh_ll}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in retest_swing: {e}")
            raise
    
    def retracement_swing(self, 
                         df: pd.DataFrame, 
                         _sign: int, 
                         _swg: str, 
                         _c: str, 
                         hh_ll_dt: pd.Timestamp, 
                         hh_ll: float, 
                         vlty: float, 
                         retrace_vol: float, 
                         retrace_pct: float) -> pd.DataFrame:
        """
        Identifies swings based on retracement logic.
        
        The function calculates the retracement from the extreme value and tests if it meets
        the volatility or percentage threshold.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing price data
        _sign : int
            Direction of the swing (+1 for up, -1 for down)
        _swg : str
            Column name for swing series (shi or slo)
        _c : str
            Column name for close prices
        hh_ll_dt : pd.Timestamp
            Date of highest high or lowest low
        hh_ll : float
            Value of highest high or lowest low
        vlty : float
            Volatility measure (e.g., ATR)
        retrace_vol : float
            Volatility-based retracement threshold
        retrace_pct : float
            Percentage-based retracement threshold
            
        Returns:
        --------
        pd.DataFrame
            Updated DataFrame with swing assignments
            
        Raises:
        -------
        ValueError
            If required columns are missing or inputs are invalid
        TypeError
            If df is not a pandas DataFrame
        """
        try:
            self.logger.debug(f"Starting retracement_swing with _sign={_sign}, hh_ll_dt={hh_ll_dt}, "
                            f"hh_ll={hh_ll}, vlty={vlty}, retrace_vol={retrace_vol}, retrace_pct={retrace_pct}")
            
            # Input validation
            if not isinstance(df, pd.DataFrame):
                raise TypeError("df must be a pandas DataFrame")
            
            required_cols = [_c, _swg]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            if _sign not in [-1, 1, 0]:
                raise ValueError("_sign must be +1, -1, or 0")
            
            if _sign == 1:
                retracement = df.loc[hh_ll_dt:, _c].min() - hh_ll
                self.logger.debug(f"Calculated retracement for swing low: {retracement}")
                
                if (vlty > 0) and (retrace_vol > 0) and ((abs(retracement / vlty) - retrace_vol) > 0):
                    df.at[hh_ll_dt, _swg] = hh_ll
                    self.logger.debug(f"Assigned swing low at {hh_ll_dt} to {_swg}: {hh_ll} (volatility test)")
                elif (retrace_pct > 0) and ((abs(retracement / hh_ll) - retrace_pct) > 0):
                    df.at[hh_ll_dt, _swg] = hh_ll
                    self.logger.debug(f"Assigned swing low at {hh_ll_dt} to {_swg}: {hh_ll} (percentage test)")
            
            elif _sign == -1:
                retracement = df.loc[hh_ll_dt:, _c].max() - hh_ll
                self.logger.debug(f"Calculated retracement for swing high: {retracement}")
                
                if (vlty > 0) and (retrace_vol > 0) and ((round(retracement / vlty, 1) - retrace_vol) > 0):
                    df.at[hh_ll_dt, _swg] = hh_ll
                    self.logger.debug(f"Assigned swing high at {hh_ll_dt} to {_swg}: {hh_ll} (volatility test)")
                elif (retrace_pct > 0) and ((round(retracement / hh_ll, 4) - retrace_pct) > 0):
                    df.at[hh_ll_dt, _swg] = hh_ll
                    self.logger.debug(f"Assigned swing high at {hh_ll_dt} to {_swg}: {hh_ll} (percentage test)")
            
            else:
                retracement = 0
                self.logger.debug("No valid swing direction, retracement set to 0")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in retracement_swing: {e}")
            raise


from algoshort.yfinance_handler import YFinanceDataHandler
handler = YFinanceDataHandler(cache_dir="./cache")
symbols = handler.list_cached_symbols()
handler.download_data(symbols, use_cache=True)

stock_data = handler.get_ohlc_data('A2A.MI')


# stock_data = stock_data.head(100)
print(stock_data)
stock_data['avg_px'] = round(stock_data[['high', 'low', 'close']].mean(axis=1), 2)
print(stock_data['avg_px'])

highs = stock_data['avg_px'].values
lows = -stock_data['avg_px'].values

reduction_target = max(len(stock_data) // 100, 5)  # Ensure minimum target

hilo_series = pd.Series(index=stock_data.index, dtype=float)
highs_list = find_peaks(highs, distance=1, width=0)
lows_list = find_peaks(lows, distance=1, width=0)

if len(lows_list[0]) > 0:
    hilo_series.iloc[lows_list[0]] = stock_data.iloc[lows_list[0]]['low']
if len(highs_list[0]) > 0:
    hilo_series.iloc[highs_list[0]] = -stock_data.iloc[highs_list[0]]['high']

print(hilo_series)
fc = Regime_fc()
# fc_ha = fc.hilo_alternation(hilo_series.dropna())
# fc = Regime_fc()
fc_ha = fc.historical_swings(stock_data)

print(fc_ha)