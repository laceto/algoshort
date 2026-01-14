import numpy as np
import pandas as pd
import logging
from scipy.signal import find_peaks
from typing import List, Tuple, Optional, Union, Dict, Any
from algoshort.utils import lower_upper_OHLC, regime_args, load_config

class RegimeFC:
    """
    Floor/Ceiling Methodology for Swing Analysis
    
    This class contains seven core methods for swing detection and cleanup, orchestrated
    through the public 'regime' method. The OHLC DataFrame is stored as an instance attribute.
    
    Methods:
    1. __hilo_alternation: Reduces data to alternating highs and lows
    2. __historical_swings: Creates multiple levels of swing analysis  
    3. __cleanup_latest_swing: Eliminates false positives from latest swings
    4. __latest_swing_variables: Instantiates arguments for the latest swing
    5. __test_distance: Tests sufficient distance from the last swing
    6. __average_true_range: Calculates the Average True Range (ATR) for volatility
    7. __retest_swing: Identifies swings based on retest logic
    8. __retracement_swing: Identifies swings based on retracement logic
    9. __regime_floor_ceiling: Detects floor/ceiling levels and regime changes
    10. __swings: Wrapper for swing analysis
    11. regime: Public method for regime analysis
    """
    
    def __init__(self, df: pd.DataFrame, log_level: int = logging.INFO):
        """
        Initialize the Floor/Ceiling swing analyzer with an OHLC DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLC DataFrame with columns 'open', 'high', 'low', 'close' (or relative equivalents).
        log_level : int, default=logging.INFO
            Logging level for the class operations.
            - logging.CRITICAL + 1 (or logging.NOTSET) for no logs at all
            - logging.WARNING to keep only warning, error and critical
            
        Raises:
        -------
        TypeError
            If df is not a pandas DataFrame.
        ValueError
            If required OHLC columns are missing.
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
        
        # Validate and store DataFrame
        if not isinstance(df, pd.DataFrame):
            self.logger.error("df must be a pandas DataFrame")
            raise TypeError("df must be a pandas DataFrame")
        
        # Check for OHLC columns (absolute or relative)
        required_cols = ['open', 'high', 'low', 'close']
        rel_cols = ['ropen', 'rhigh', 'rlow', 'rclose']
        if not (all(col in df.columns for col in required_cols) or 
                all(col in df.columns for col in rel_cols)):
            self.logger.error("DataFrame must contain OHLC columns (absolute or relative)")
            raise ValueError("Missing required OHLC columns")
        
        self.df = df.copy()  # Store a copy to prevent external modifications
        self.logger.info(f"Initialized Regime_fc with DataFrame of shape {self.df.shape}")

    def _hilo_alternation(self, 
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
        self.logger.debug(f"Starting hilo_alternation with {len(hilo)} data points")
        
        # Input validation
        if not isinstance(hilo, pd.Series):
            self.logger.error("hilo must be a pandas Series")
            raise TypeError("hilo must be a pandas Series")
        
        if hilo.empty:
            self.logger.warning("Empty hilo series provided")
            return pd.Series(dtype=float)
        
        i=0    
        while (np.sign(hilo.shift(1)) == np.sign(hilo)).any(): # runs until duplicates are eliminated

            # removes swing lows > swing highs
            hilo.loc[(np.sign(hilo.shift(1)) != np.sign(hilo)) &  # hilo alternation test
                    (hilo.shift(1)<0) &  # previous datapoint:  high
                    (np.abs(hilo.shift(1)) < np.abs(hilo) )] = np.nan # high[-1] < low, eliminate low 

            hilo.loc[(np.sign(hilo.shift(1)) != np.sign(hilo)) &  # hilo alternation
                    (hilo.shift(1)>0) &  # previous swing: low
                    (np.abs(hilo ) < hilo.shift(1))] = np.nan # swing high < swing low[-1]

            # alternation test: removes duplicate swings & keep extremes
            hilo.loc[(np.sign(hilo.shift(1)) == np.sign(hilo)) & # same sign
                    (hilo.shift(1) < hilo )] = np.nan # keep lower one

            hilo.loc[(np.sign(hilo.shift(-1)) == np.sign(hilo)) & # same sign, forward looking 
                    (hilo.shift(-1) < hilo )] = np.nan # keep forward one

            # removes noisy swings: distance test
            if pd.notnull(dist):
                hilo.loc[(np.sign(hilo.shift(1)) != np.sign(hilo))&\
                    (np.abs(hilo + hilo.shift(1)).div(dist, fill_value=1)< hurdle)] = np.nan

            # reduce hilo after each pass
            hilo = hilo.dropna().copy() 
            i+=1
            if i == 4: # breaks infinite loop
                break 
            return hilo
        
    def _historical_swings(self, 
                        relative: bool = True,
                        dist: Optional[pd.Series] = None,
                        hurdle: Optional[float] = None) -> None:
        """
        Perform multi-level swing analysis on stored OHLC data using a floor/ceiling methodology.

        This method computes multiple levels of swing highs and lows using:
        1. Average price calculation (from high, low, close)
        2. Identification of peaks and troughs
        3. Alternating high/low reduction using `__hilo_alternation`
        4. Populating swing levels iteratively in self.df

        Parameters
        ----------
        relative : bool, default=True
            Whether to use relative price adjustments.
        dist : pd.Series, optional
            Distance series for noise filtering in `__hilo_alternation`.
        hurdle : float, optional
            Threshold for noise filtering in `__hilo_alternation`.

        Returns
        -------
        None
            Updates self.df with additional swing level columns (e.g., hi1, lo1, hi2, lo2).

        Notes
        -----
        Iterates up to 9 levels or until the dataset is too small.

        """
        self.logger.info(f"Starting historical_swings analysis (relative={relative})")

        _o, _h, _l, _c = lower_upper_OHLC(self.df, relative=relative)

        try:
            reduction = self.df[[_o, _h, _l, _c]].copy()
            reduction['avg_px'] = round(reduction[[_h, _l, _c]].mean(axis=1), 2)
            highs = reduction['avg_px'].values
            lows = -reduction['avg_px'].values
            reduction_target = len(reduction) // 100

            self.logger.debug(f"Reduction target set to {reduction_target} rows")
            n = 0

            while len(reduction) >= reduction_target:
                self.logger.debug(f"Iteration {n+1}, reduction size: {len(reduction)}")

                highs_list = find_peaks(highs, distance=1, width=0)
                lows_list = find_peaks(lows, distance=1, width=0)

                if len(highs_list[0]) == 0 or len(lows_list[0]) == 0:
                    self.logger.warning("No peaks found, breaking loop")
                    break

                hilo = reduction.iloc[lows_list[0]][_l].sub(reduction.iloc[highs_list[0]][_h], fill_value=0)
                self.logger.debug(f"Initial hilo computed, {hilo.notna().sum()} valid swings")

                # Apply alternation
                # hilo = self._hilo_alternation(hilo, dist=dist, hurdle=hurdle)
                self._hilo_alternation(hilo, dist=dist, hurdle=hurdle)
                reduction['hilo'] = hilo

                # Populate reduction dataframe
                n += 1
                high_col = f"{_h[:2]}{n}"
                low_col = f"{_l[:2]}{n}"
                reduction[high_col] = reduction.loc[reduction['hilo'] < 0, _h]
                reduction[low_col] = reduction.loc[reduction['hilo'] > 0, _l]

                # Populate main dataframe
                self.df[high_col] = reduction[high_col]
                self.df[low_col] = reduction[low_col]

                # Reduce for next iteration
                reduction = reduction.dropna(subset=['hilo']).copy()
                reduction = reduction.ffill()
                highs = reduction[high_col].values
                lows = -reduction[low_col].values

                if n >= 9:
                    self.logger.info("Maximum swing levels reached, stopping iteration")
                    break

            self.logger.info(f"historical_swings completed with {n} swing levels")
            # return df

        except Exception as e:
            self.logger.exception(f"Error in historical_swings: {e}")
            raise


    def _cleanup_latest_swing(self, shi: str, slo: str, rt_hi: str, rt_lo: str) -> None:
        """
        Remove false positives from the latest swing high/low levels.

        Parameters
        ----------
        shi : str
            Swing high column name.
        slo : str
            Swing low column name.
        rt_hi : str
            Retest high column name.
        rt_lo : str
            Retest low column name.

        Returns
        -------
        None
            Updates self.df with cleaned swing columns.

        """
        self.logger.debug(f"Cleaning latest swings: {shi}, {slo}")
        try:
            shi_dt = self.df.loc[pd.notnull(self.df[shi]), shi].index[-1]
            s_hi = self.df.loc[pd.notnull(self.df[shi]), shi].iloc[-1]  
            slo_dt = self.df.loc[pd.notnull(self.df[slo]), slo].index[-1] 
            s_lo = self.df.loc[pd.notnull(self.df[slo]), slo].iloc[-1]  
            len_shi_dt = len(self.df[:shi_dt])
            len_slo_dt = len(self.df[:slo_dt])

            for _ in range(2):
                if (len_shi_dt > len_slo_dt) and ((self.df.loc[shi_dt:, rt_hi].max() > s_hi) or (s_hi < s_lo)):
                    self.df.loc[shi_dt, shi] = np.nan
                    len_shi_dt = 0
                elif (len_slo_dt > len_shi_dt) and ((self.df.loc[slo_dt:, rt_lo].min() < s_lo) or (s_hi < s_lo)):
                    self.df.loc[slo_dt, slo] = np.nan
                    len_slo_dt = 0
            
        except Exception as e:
            self.logger.exception(f"Error in cleanup_latest_swing: {e}")
            raise

    def _latest_swing_variables(self, shi: str, slo: str, rt_hi: str, rt_lo: str, _h: str, _l: str, _c: str) -> Tuple[int, float, pd.Timestamp, str, str, float, pd.Timestamp]:
        """
        Extract latest swing dates and values.

        Returns
        -------
        tuple: (ud, bs, bs_dt, _rt, _swg, hh_ll, hh_ll_dt)
            - ud: direction of the last swing (+1 low, -1 high)
            - bs: last swing value
            - bs_dt: last swing date
            - _rt: corresponding retest column
            - _swg: corresponding swing column
            - hh_ll: extreme high/low for retracement/retest
            - hh_ll_dt: index of hh_ll
        """
        try:
            shi_dt = self.df.loc[pd.notnull(self.df[shi]), shi].index[-1]
            slo_dt = self.df.loc[pd.notnull(self.df[slo]), slo].index[-1]
            s_hi = self.df.loc[pd.notnull(self.df[shi]), shi].iloc[-1]
            s_lo = self.df.loc[pd.notnull(self.df[slo]), slo].iloc[-1]

            if slo_dt > shi_dt:
                swg_var = [1, s_lo, slo_dt, rt_lo, shi, self.df.loc[slo_dt:, _h].max(), self.df.loc[slo_dt:, _h].idxmax()]
            elif shi_dt > slo_dt:
                swg_var = [-1, s_hi, shi_dt, rt_hi, slo, self.df.loc[shi_dt:, _l].min(), self.df.loc[shi_dt:, _l].idxmin()]
            else:
                swg_var = [0, np.nan, np.nan, None, None, np.nan, None]

            ud, bs, bs_dt, _rt, _swg, hh_ll, hh_ll_dt = swg_var
            return ud, bs, bs_dt, _rt, _swg, hh_ll, hh_ll_dt
        except Exception as e:
            self.logger.exception(f"Error in latest_swing_variables: {e}")
            raise

    def _test_distance(self, ud: int, bs: float, hh_ll: float, dist_vol: float, dist_pct: float) -> int:
        """
        Check whether a swing passes distance thresholds.

        Returns
        -------
        int
            Distance test result scaled by swing direction.
        """
        try:
            if dist_vol > 0:
                distance_test = np.sign(abs(hh_ll - bs) - dist_vol)
            elif dist_pct > 0:
                distance_test = np.sign(abs(hh_ll / bs - 1) - dist_pct)
            else:
                distance_test = np.sign(dist_pct)
            return int(max(distance_test, 0) * ud)
        except Exception as e:
            self.logger.exception(f"Error in test_distance: {e}")
            raise

    def _average_true_range(self, _h: str, _l: str, _c: str, n: int) -> pd.Series:
        """
        Calculate Average True Range (ATR) over n periods.

        Returns
        -------
        pd.Series
            Rolling ATR values.
        """
        try:
            atr = (self.df[_h].combine(self.df[_c].shift(), max) - 
                   self.df[_l].combine(self.df[_c].shift(), min)).rolling(window=n).mean()
            return atr
        except Exception as e:
            self.logger.exception(f"Error in average_true_range: {e}")
            raise

    def _retest_swing(self, _sign: int, _rt: str, hh_ll_dt, hh_ll, _c: str, _swg: str) -> None:
        """
        Identify swings based on retest logic.
        """
        try:
            rt_sgmt = self.df.loc[hh_ll_dt:, _rt]
            if (rt_sgmt.count() > 0) and (_sign != 0):
                if _sign == 1:
                    rt_list = [rt_sgmt.idxmax(), rt_sgmt.max(), self.df.loc[rt_sgmt.idxmax():, _c].cummin()]
                elif _sign == -1:
                    rt_list = [rt_sgmt.idxmin(), rt_sgmt.min(), self.df.loc[rt_sgmt.idxmin():, _c].cummax()]

                rt_dt, rt_hurdle, rt_px = rt_list

                col_name = 'rrt' if str(_c)[0] == 'r' else 'rt'
                self.df.loc[rt_dt, col_name] = rt_hurdle

                if (np.sign(rt_px - rt_hurdle) == -np.sign(_sign)).any():
                    self.df.at[hh_ll_dt, _swg] = hh_ll
            
        except Exception as e:
            self.logger.exception(f"Error in retest_swing: {e}")
            raise

    def _retracement_swing(self, _sign: int, _swg: str, _c: str, hh_ll_dt, hh_ll, vlty: float, retrace_vol: float, retrace_pct: float) -> None:
        """
        Identify swings based on retracement logic.
        """
        try:
            if _sign == 1:
                retracement = self.df.loc[hh_ll_dt:, _c].min() - hh_ll
                if (vlty > 0 and retrace_vol > 0) and ((abs(retracement / vlty) - retrace_vol) > 0):
                    self.df.at[hh_ll_dt, _swg] = hh_ll
                elif retrace_pct > 0 and ((abs(retracement / hh_ll) - retrace_pct) > 0):
                    self.df.at[hh_ll_dt, _swg] = hh_ll
            elif _sign == -1:
                retracement = self.df.loc[hh_ll_dt:, _c].max() - hh_ll
                if (vlty > 0 and retrace_vol > 0) and ((round(retracement / vlty, 1) - retrace_vol) > 0):
                    self.df.at[hh_ll_dt, _swg] = hh_ll
                elif retrace_pct > 0 and ((round(retracement / hh_ll, 4) - retrace_pct) > 0):
                    self.df.at[hh_ll_dt, _swg] = hh_ll
            
        except Exception as e:
            self.logger.exception(f"Error in retracement_swing: {e}")
            raise

    def _regime_floor_ceiling(self, _h: str, _l: str, _c: str, slo: str, shi: str,
                             flr: str, clg: str, rg: str, rg_ch: str, stdev: pd.Series, threshold: float) -> None:
        """
        Detect floor/ceiling levels and track regime changes based on swing highs and lows.

        This method:
        - Identifies classic floors and ceilings from swing highs/lows
        - Handles exceptions when price penetrates discovery swings
        - Tracks breakout/breakdown regimes and populates relevant columns
        - Updates regime columns using cumulative min/max logic

        Parameters
        ----------
        _h : str
            High price column.
        _l : str
            Low price column.
        _c : str
            Close price column.
        slo : str
            Swing low column.
        shi : str
            Swing high column.
        flr : str
            Floor column name to populate.
        clg : str
            Ceiling column name to populate.
        rg : str
            Regime column name to populate.
        rg_ch : str
            Regime change column name to populate.
        stdev : pd.Series
            Standard deviation series for threshold scaling.
        threshold : float
            Threshold for floor/ceiling discovery.

        Returns
        -------
        None
            Updates self.df with floor, ceiling, regime, and regime change columns.
        """
        self.logger.info("Starting regime_floor_ceiling analysis")
        try:
            # Lists initialization
            threshold_test, rg_ch_ix_list, rg_ch_list = [], [], []
            floor_ix_list, floor_list, ceiling_ix_list, ceiling_list = [self.df.index[0]], [self.df[_l].iloc[0]], [self.df.index[0]], [self.df[_h].iloc[0]]

            # Boolean flags
            ceiling_found = floor_found = breakdown = breakout = False

            # Swing data
            swing_highs = list(self.df.loc[pd.notnull(self.df[shi]), shi])
            swing_highs_ix = list(self.df.loc[pd.notnull(self.df[shi])].index)
            swing_lows = list(self.df.loc[pd.notnull(self.df[slo]), slo])
            swing_lows_ix = list(self.df.loc[pd.notnull(self.df[slo])].index)
            loop_size = max(len(swing_highs), len(swing_lows))

            for i in range(loop_size):
                # Handle asymmetric swing lists
                s_lo_ix, s_lo = (swing_lows_ix[i], swing_lows[i]) if i < len(swing_lows) else (swing_lows_ix[-1], swing_lows[-1])
                s_hi_ix, s_hi = (swing_highs_ix[i], swing_highs[i]) if i < len(swing_highs) else (swing_highs_ix[-1], swing_highs[-1])
                swing_max_ix = max(s_lo_ix, s_hi_ix)

                # Classic ceiling discovery
                if not ceiling_found:
                    top = self.df.loc[floor_ix_list[-1]:s_hi_ix, _h].max()
                    ceiling_test = round((s_hi - top) / stdev[s_hi_ix], 1)
                    if ceiling_test <= -threshold:
                        ceiling_found, floor_found, breakdown, breakout = True, False, False, False
                        threshold_test.append(ceiling_test)
                        ceiling_list.append(top)
                        ceiling_ix_list.append(self.df.loc[floor_ix_list[-1]:s_hi_ix, _h].idxmax())
                        rg_ch_ix_list.append(s_hi_ix)
                        rg_ch_list.append(s_hi)

                # Ceiling found: update regime
                elif ceiling_found:
                    close_high = self.df.loc[rg_ch_ix_list[-1]:swing_max_ix, _c].cummax()
                    self.df.loc[rg_ch_ix_list[-1]:swing_max_ix, rg] = np.sign(close_high - rg_ch_list[-1])
                    if (self.df.loc[rg_ch_ix_list[-1]:swing_max_ix, rg] > 0).any():
                        ceiling_found, floor_found, breakdown = False, False, False
                        breakout = True

                if breakout:
                    brkout_high_ix = self.df.loc[rg_ch_ix_list[-1]:swing_max_ix, _c].idxmax()
                    brkout_low = self.df.loc[brkout_high_ix:swing_max_ix, _c].cummin()
                    self.df.loc[brkout_high_ix:swing_max_ix, rg] = np.sign(brkout_low - rg_ch_list[-1])

                # Classic floor discovery
                if not floor_found:
                    bottom = self.df.loc[ceiling_ix_list[-1]:s_lo_ix, _l].min()
                    floor_test = round((s_lo - bottom) / stdev[s_lo_ix], 1)
                    if floor_test >= threshold:
                        floor_found, ceiling_found, breakdown, breakout = True, False, False, False
                        threshold_test.append(floor_test)
                        floor_list.append(bottom)
                        floor_ix_list.append(self.df.loc[ceiling_ix_list[-1]:s_lo_ix, _l].idxmin())
                        rg_ch_ix_list.append(s_lo_ix)
                        rg_ch_list.append(s_lo)

                # Floor found: update regime
                elif floor_found:
                    close_low = self.df.loc[rg_ch_ix_list[-1]:swing_max_ix, _c].cummin()
                    self.df.loc[rg_ch_ix_list[-1]:swing_max_ix, rg] = np.sign(close_low - rg_ch_list[-1])
                    if (self.df.loc[rg_ch_ix_list[-1]:swing_max_ix, rg] < 0).any():
                        floor_found, breakout = False, False
                        breakdown = True

                if breakdown:
                    brkdwn_low_ix = self.df.loc[rg_ch_ix_list[-1]:swing_max_ix, _c].idxmin()
                    breakdown_rebound = self.df.loc[brkdwn_low_ix:swing_max_ix, _c].cummax()
                    self.df.loc[brkdwn_low_ix:swing_max_ix, rg] = np.sign(breakdown_rebound - rg_ch_list[-1])

            # Populate final columns
            self.df.loc[floor_ix_list[1:], flr] = floor_list[1:]
            self.df.loc[ceiling_ix_list[1:], clg] = ceiling_list[1:]
            self.df.loc[rg_ch_ix_list, rg_ch] = rg_ch_list
            self.df[rg_ch] = self.df[rg_ch].ffill()
            self.df.loc[swing_max_ix:, rg] = np.where(ceiling_found,
                                                 np.sign(self.df.loc[swing_max_ix:, _c].cummax() - rg_ch_list[-1]),
                                                 np.where(floor_found,
                                                          np.sign(self.df.loc[swing_max_ix:, _c].cummin() - rg_ch_list[-1]),
                                                          np.sign(self.df.loc[swing_max_ix:, _c].rolling(5).mean() - rg_ch_list[-1])))
            self.df[rg] = self.df[rg].ffill()

            self.logger.info("regime_floor_ceiling completed")
            

        except Exception as e:
            self.logger.exception(f"Error in regime_floor_ceiling: {e}")
            raise


    def _swings(self, relative: bool, lvl: int, vlty_n: int, dgt: int, d_vol: int, dist_pct: float, retrace_pct: float, r_vol: float) -> None:
        """
        Perform full swing analysis on stored OHLC data.

        This method:
        - Computes lower/upper OHLC columns (absolute or relative)
        - Computes historical swings
        - Cleans up false-positive latest swings
        - Calculates latest swing variables
        - Applies ATR-based volatility adjustments
        - Performs retest and retracement analysis

        Parameters
        ----------
        relative : bool, default=False
            Whether to use relative price adjustments.
        config_path : str, default='config.json'
            Path to JSON config file.

        Returns
        -------
        None
            Updates self.df with swing columns and retest/retracement analysis.
        """
        self.logger.info(f"Starting swings analysis (relative={relative})")

        try:

            if relative:
                _o, _h, _l, _c = lower_upper_OHLC(self.df, relative=True)
                rt_lo, rt_hi, slo, shi, rg, clg, flr, rg_ch = regime_args(self.df, lvl, relative=True)
            else:
                _o, _h, _l, _c = lower_upper_OHLC(self.df, relative=False)
                rt_lo, rt_hi, slo, shi, rg, clg, flr, rg_ch = regime_args(self.df, lvl, relative=False)

            self._historical_swings(relative = relative, dist= None, hurdle= None)
            self._cleanup_latest_swing(shi, slo, rt_hi, rt_lo)
            ud, bs, bs_dt, _rt, _swg, hh_ll, hh_ll_dt = self._latest_swing_variables(shi, slo, rt_hi, rt_lo, _h, _l, _c)

            vlty = round(self._average_true_range(_h, _l, _c, n=vlty_n).loc[hh_ll_dt], dgt)
            dist_vol = d_vol * vlty
            _sign = self._test_distance(ud, bs, hh_ll, dist_vol, dist_pct)

            self._retest_swing(_sign, _rt, hh_ll_dt, hh_ll, _c, _swg)
            retrace_vol = r_vol * vlty
            self._retracement_swing(_sign, _swg, _c, hh_ll_dt, hh_ll, vlty, retrace_vol, retrace_pct)

            self.logger.info("Completed swings analysis")
            

        except Exception as e:
            self.logger.exception(f"Error in swings: {e}")
            raise

    def compute_regime(self, relative: bool, lvl: int, vlty_n: int, dgt: int, d_vol: int, dist_pct: float, retrace_pct: float, r_vol: float, threshold: float) -> pd.DataFrame:
        """
        Identify regime floor/ceiling levels based on swings.

        This method:
        - Computes lower/upper OHLC columns
        - Determines swing variables
        - Computes rolling standard deviation
        - Applies floor/ceiling and regime analysis

        Parameters
        ----------
        relative : bool, default=False
            Whether to use relative price adjustments.
        config_path : str, default='config.json'
            Path to JSON config file.

        Returns
        -------
        pd.DataFrame
            DataFrame with regime, floor, ceiling, and regime change columns updated.
        """
        self.logger.info(f"Starting regime analysis (relative={relative})")
        # Load config
        # config = load_config(config_path)
        try:
            _o, _h, _l, _c = lower_upper_OHLC(self.df, relative=relative)
            rt_lo, rt_hi, slo, shi, rg, clg, flr, rg_ch = regime_args(self.df, lvl=lvl, relative=relative)
            self._swings(lvl = lvl, vlty_n = vlty_n, dgt=dgt, d_vol=d_vol, dist_pct=dist_pct, retrace_pct=retrace_pct, r_vol=r_vol, relative=relative)
            stdev = self.df[_c].rolling(vlty_n).std(ddof=0)
            self._regime_floor_ceiling(_h=_h, _l=_l, _c=_c, slo=slo, shi=shi, flr=flr, clg=clg, rg=rg, rg_ch=rg_ch, stdev=stdev, threshold=threshold)

            self.logger.info("Completed regime analysis")
            return self.df

        except Exception as e:
            self.logger.exception(f"Error in regime: {e}")
            raise