"""
Floor/Ceiling Regime Detection based on Swing Analysis.

This module provides sophisticated regime detection using swing high/low analysis
with floor and ceiling level identification.

Signals:
    - 1: Bullish (above floor, breakout)
    - -1: Bearish (below ceiling, breakdown)
    - 0: Neutral/consolidation

Example:
    >>> from algoshort.regimes.floor_ceiling import FloorCeilingRegime
    >>> detector = FloorCeilingRegime(df)
    >>> result = detector.floor_ceiling(lvl=1, threshold=1.5)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from algoshort.regimes.base import BaseRegimeDetector, calculate_atr


logger = logging.getLogger(__name__)


class FloorCeilingRegime(BaseRegimeDetector):
    """
    Floor/Ceiling methodology for swing analysis and regime detection.

    This class performs multi-level swing analysis to identify:
    - Swing highs and lows at multiple levels
    - Floor and ceiling levels
    - Regime changes based on breakout/breakdown

    The algorithm:
    1. Identifies peaks and troughs using average price
    2. Reduces to alternating highs/lows (hilo alternation)
    3. Cleans up false positives
    4. Applies retest and retracement logic
    5. Detects floor/ceiling levels and regime changes

    Attributes:
        df (pd.DataFrame): OHLC DataFrame
        _cache (Dict[str, pd.Series]): Cache for computed values
    """

    # Default parameters
    DEFAULT_LEVEL = 1
    DEFAULT_VOLATILITY_WINDOW = 63
    DEFAULT_DIGITS = 2
    DEFAULT_DISTANCE_VOLATILITY = 5
    DEFAULT_DISTANCE_PCT = 0.05
    DEFAULT_RETRACE_PCT = 0.05
    DEFAULT_RETRACE_VOL = 2.5
    DEFAULT_THRESHOLD = 1.5

    def __init__(self, df: pd.DataFrame, log_level: int = logging.INFO):
        """
        Initialize the Floor/Ceiling Regime detector.

        Args:
            df: DataFrame with OHLC columns
            log_level: Logging level

        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If DataFrame is empty or missing OHLC columns
        """
        super().__init__(df, log_level)
        self.logger.info(f"FloorCeilingRegime initialized with {len(self._df)} rows")

    def _get_regime_columns(
        self,
        lvl: int,
        relative: bool = False
    ) -> Tuple[str, str, str, str, str, str, str, str]:
        """
        Get regime-related column names.

        Args:
            lvl: Swing level
            relative: If True, use relative prefixes

        Returns:
            Tuple of column names:
            (rt_lo, rt_hi, slo, shi, rg, clg, flr, rg_ch)
        """
        if relative:
            if 'Low' in self._df.columns:
                return (
                    'rL1', 'rH1', f'rL{lvl}', f'rH{lvl}',
                    'rrg', 'rclg', 'rflr', 'rrg_ch'
                )
            return (
                'rl1', 'rh1', f'rl{lvl}', f'rh{lvl}',
                'rrg', 'rclg', 'rflr', 'rrg_ch'
            )
        else:
            if 'Low' in self._df.columns:
                return (
                    'Lo1', 'Hi1', f'Lo{lvl}', f'Hi{lvl}',
                    'rg', 'clg', 'flr', 'rg_ch'
                )
            return (
                'lo1', 'hi1', f'lo{lvl}', f'hi{lvl}',
                'rg', 'clg', 'flr', 'rg_ch'
            )

    def _hilo_alternation(
        self,
        hilo: pd.Series,
        dist: Optional[pd.Series] = None,
        hurdle: Optional[float] = None
    ) -> pd.Series:
        """
        Reduce series to alternating highs and lows.

        Eliminates consecutive same-side extremes, keeping only the most extreme.
        - Highs: negative sign
        - Lows: positive sign

        Args:
            hilo: Series with signed high/low values
            dist: Distance series for noise filtering
            hurdle: Threshold for distance filtering

        Returns:
            Reduced series with alternating highs/lows
        """
        self.logger.debug(f"Starting hilo_alternation with {len(hilo)} points")

        if not isinstance(hilo, pd.Series):
            raise TypeError("hilo must be a pandas Series")

        if hilo.empty:
            self.logger.warning("Empty hilo series provided")
            return pd.Series(dtype=float)

        i = 0
        while (np.sign(hilo.shift(1)) == np.sign(hilo)).any():
            # Remove swing lows > swing highs
            hilo.loc[
                (np.sign(hilo.shift(1)) != np.sign(hilo)) &
                (hilo.shift(1) < 0) &
                (np.abs(hilo.shift(1)) < np.abs(hilo))
            ] = np.nan

            hilo.loc[
                (np.sign(hilo.shift(1)) != np.sign(hilo)) &
                (hilo.shift(1) > 0) &
                (np.abs(hilo) < hilo.shift(1))
            ] = np.nan

            # Remove duplicate swings, keep extremes
            hilo.loc[
                (np.sign(hilo.shift(1)) == np.sign(hilo)) &
                (hilo.shift(1) < hilo)
            ] = np.nan

            hilo.loc[
                (np.sign(hilo.shift(-1)) == np.sign(hilo)) &
                (hilo.shift(-1) < hilo)
            ] = np.nan

            # Remove noisy swings with distance test
            if dist is not None and hurdle is not None:
                hilo.loc[
                    (np.sign(hilo.shift(1)) != np.sign(hilo)) &
                    (np.abs(hilo + hilo.shift(1)).div(dist, fill_value=1) < hurdle)
                ] = np.nan

            hilo = hilo.dropna().copy()
            i += 1
            if i >= 4:
                break

        return hilo

    def _historical_swings(
        self,
        relative: bool = False,
        dist: Optional[pd.Series] = None,
        hurdle: Optional[float] = None
    ) -> None:
        """
        Perform multi-level swing analysis.

        Updates self._df with swing level columns (hi1, lo1, hi2, lo2, etc.)

        Args:
            relative: If True, use relative OHLC columns
            dist: Distance series for noise filtering
            hurdle: Threshold for distance filtering
        """
        self.logger.info(f"Starting historical_swings (relative={relative})")

        _o, _h, _l, _c = self._get_ohlc_columns(relative=relative)

        try:
            reduction = self._df[[_o, _h, _l, _c]].copy()
            reduction['avg_px'] = round(reduction[[_h, _l, _c]].mean(axis=1), 2)

            highs = reduction['avg_px'].values
            lows = -reduction['avg_px'].values
            reduction_target = max(len(reduction) // 100, 10)

            n = 0
            while len(reduction) >= reduction_target:
                self.logger.debug(f"Iteration {n + 1}, size: {len(reduction)}")

                highs_list = find_peaks(highs, distance=1, width=0)
                lows_list = find_peaks(lows, distance=1, width=0)

                if len(highs_list[0]) == 0 or len(lows_list[0]) == 0:
                    self.logger.warning("No peaks found, breaking")
                    break

                hilo = reduction.iloc[lows_list[0]][_l].sub(
                    reduction.iloc[highs_list[0]][_h], fill_value=0
                )

                hilo = self._hilo_alternation(hilo, dist=dist, hurdle=hurdle)
                reduction['hilo'] = hilo

                n += 1
                high_col = f"{_h[:2]}{n}"
                low_col = f"{_l[:2]}{n}"

                reduction[high_col] = reduction.loc[reduction['hilo'] < 0, _h]
                reduction[low_col] = reduction.loc[reduction['hilo'] > 0, _l]

                self._df[high_col] = reduction[high_col]
                self._df[low_col] = reduction[low_col]

                reduction = reduction.dropna(subset=['hilo']).copy()
                reduction = reduction.ffill()
                highs = reduction[high_col].values
                lows = -reduction[low_col].values

                if n >= 9:
                    self.logger.info("Max swing levels reached")
                    break

            self.logger.info(f"Historical swings: {n} levels computed")

        except Exception as e:
            self.logger.exception(f"Error in historical_swings: {e}")
            raise

    def _cleanup_latest_swing(
        self,
        shi: str,
        slo: str,
        rt_hi: str,
        rt_lo: str
    ) -> None:
        """
        Remove false positives from latest swing levels.

        Args:
            shi: Swing high column name
            slo: Swing low column name
            rt_hi: Retest high column name
            rt_lo: Retest low column name
        """
        self.logger.debug(f"Cleaning latest swings: {shi}, {slo}")

        try:
            if self._df[shi].notna().sum() == 0 or self._df[slo].notna().sum() == 0:
                return

            shi_dt = self._df.loc[self._df[shi].notna(), shi].index[-1]
            s_hi = self._df.loc[self._df[shi].notna(), shi].iloc[-1]
            slo_dt = self._df.loc[self._df[slo].notna(), slo].index[-1]
            s_lo = self._df.loc[self._df[slo].notna(), slo].iloc[-1]

            len_shi_dt = len(self._df[:shi_dt])
            len_slo_dt = len(self._df[:slo_dt])

            for _ in range(2):
                if (len_shi_dt > len_slo_dt) and (
                    (self._df.loc[shi_dt:, rt_hi].max() > s_hi) or (s_hi < s_lo)
                ):
                    self._df.loc[shi_dt, shi] = np.nan
                    len_shi_dt = 0
                elif (len_slo_dt > len_shi_dt) and (
                    (self._df.loc[slo_dt:, rt_lo].min() < s_lo) or (s_hi < s_lo)
                ):
                    self._df.loc[slo_dt, slo] = np.nan
                    len_slo_dt = 0

        except Exception as e:
            self.logger.exception(f"Error in cleanup_latest_swing: {e}")
            raise

    def _latest_swing_variables(
        self,
        shi: str,
        slo: str,
        rt_hi: str,
        rt_lo: str,
        _h: str,
        _l: str,
        _c: str
    ) -> Tuple[int, float, Any, Optional[str], Optional[str], float, Any]:
        """
        Extract latest swing dates and values.

        Returns:
            Tuple: (ud, bs, bs_dt, _rt, _swg, hh_ll, hh_ll_dt)
                - ud: Direction (+1 low, -1 high)
                - bs: Last swing value
                - bs_dt: Last swing date
                - _rt: Retest column
                - _swg: Swing column
                - hh_ll: Extreme value
                - hh_ll_dt: Extreme index
        """
        try:
            if self._df[shi].notna().sum() == 0 or self._df[slo].notna().sum() == 0:
                return (0, np.nan, None, None, None, np.nan, None)

            shi_dt = self._df.loc[self._df[shi].notna(), shi].index[-1]
            slo_dt = self._df.loc[self._df[slo].notna(), slo].index[-1]
            s_hi = self._df.loc[self._df[shi].notna(), shi].iloc[-1]
            s_lo = self._df.loc[self._df[slo].notna(), slo].iloc[-1]

            if slo_dt > shi_dt:
                return (
                    1, s_lo, slo_dt, rt_lo, shi,
                    self._df.loc[slo_dt:, _h].max(),
                    self._df.loc[slo_dt:, _h].idxmax()
                )
            elif shi_dt > slo_dt:
                return (
                    -1, s_hi, shi_dt, rt_hi, slo,
                    self._df.loc[shi_dt:, _l].min(),
                    self._df.loc[shi_dt:, _l].idxmin()
                )
            else:
                return (0, np.nan, None, None, None, np.nan, None)

        except Exception as e:
            self.logger.exception(f"Error in latest_swing_variables: {e}")
            raise

    def _test_distance(
        self,
        ud: int,
        bs: float,
        hh_ll: float,
        dist_vol: float,
        dist_pct: float
    ) -> int:
        """
        Check if swing passes distance thresholds.

        Returns:
            Distance test result scaled by direction
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

    def _average_true_range(
        self,
        _h: str,
        _l: str,
        _c: str,
        n: int
    ) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            _h: High column name
            _l: Low column name
            _c: Close column name
            n: Window periods

        Returns:
            ATR series
        """
        try:
            atr = (
                self._df[_h].combine(self._df[_c].shift(), max) -
                self._df[_l].combine(self._df[_c].shift(), min)
            ).rolling(window=n).mean()
            return atr
        except Exception as e:
            self.logger.exception(f"Error in average_true_range: {e}")
            raise

    def _retest_swing(
        self,
        _sign: int,
        _rt: str,
        hh_ll_dt: Any,
        hh_ll: float,
        _c: str,
        _swg: str
    ) -> None:
        """
        Identify swings based on retest logic.
        """
        try:
            if hh_ll_dt is None or _rt is None:
                return

            rt_sgmt = self._df.loc[hh_ll_dt:, _rt]
            if rt_sgmt.count() > 0 and _sign != 0:
                if _sign == 1:
                    rt_dt = rt_sgmt.idxmax()
                    rt_hurdle = rt_sgmt.max()
                    rt_px = self._df.loc[rt_dt:, _c].cummin()
                elif _sign == -1:
                    rt_dt = rt_sgmt.idxmin()
                    rt_hurdle = rt_sgmt.min()
                    rt_px = self._df.loc[rt_dt:, _c].cummax()
                else:
                    return

                col_name = 'rrt' if str(_c)[0] == 'r' else 'rt'
                self._df.loc[rt_dt, col_name] = rt_hurdle

                if (np.sign(rt_px - rt_hurdle) == -np.sign(_sign)).any():
                    self._df.at[hh_ll_dt, _swg] = hh_ll

        except Exception as e:
            self.logger.exception(f"Error in retest_swing: {e}")
            raise

    def _retracement_swing(
        self,
        _sign: int,
        _swg: str,
        _c: str,
        hh_ll_dt: Any,
        hh_ll: float,
        vlty: float,
        retrace_vol: float,
        retrace_pct: float
    ) -> None:
        """
        Identify swings based on retracement logic.
        """
        try:
            if hh_ll_dt is None or _swg is None:
                return

            if _sign == 1:
                retracement = self._df.loc[hh_ll_dt:, _c].min() - hh_ll
                if vlty > 0 and retrace_vol > 0:
                    if (abs(retracement / vlty) - retrace_vol) > 0:
                        self._df.at[hh_ll_dt, _swg] = hh_ll
                elif retrace_pct > 0:
                    if (abs(retracement / hh_ll) - retrace_pct) > 0:
                        self._df.at[hh_ll_dt, _swg] = hh_ll

            elif _sign == -1:
                retracement = self._df.loc[hh_ll_dt:, _c].max() - hh_ll
                if vlty > 0 and retrace_vol > 0:
                    if (round(retracement / vlty, 1) - retrace_vol) > 0:
                        self._df.at[hh_ll_dt, _swg] = hh_ll
                elif retrace_pct > 0:
                    if (round(retracement / hh_ll, 4) - retrace_pct) > 0:
                        self._df.at[hh_ll_dt, _swg] = hh_ll

        except Exception as e:
            self.logger.exception(f"Error in retracement_swing: {e}")
            raise

    def _regime_floor_ceiling(
        self,
        _h: str,
        _l: str,
        _c: str,
        slo: str,
        shi: str,
        flr: str,
        clg: str,
        rg: str,
        rg_ch: str,
        stdev: pd.Series,
        threshold: float
    ) -> None:
        """
        Detect floor/ceiling levels and regime changes.

        Updates self._df with floor, ceiling, regime, and regime change columns.
        """
        self.logger.info("Starting regime_floor_ceiling analysis")

        try:
            # Initialize lists
            threshold_test = []
            rg_ch_ix_list: List[Any] = []
            rg_ch_list: List[float] = []

            floor_ix_list = [self._df.index[0]]
            floor_list = [self._df[_l].iloc[0]]
            ceiling_ix_list = [self._df.index[0]]
            ceiling_list = [self._df[_h].iloc[0]]

            # Boolean flags
            ceiling_found = floor_found = breakdown = breakout = False

            # Get swing data
            swing_highs = list(self._df.loc[self._df[shi].notna(), shi])
            swing_highs_ix = list(self._df.loc[self._df[shi].notna()].index)
            swing_lows = list(self._df.loc[self._df[slo].notna(), slo])
            swing_lows_ix = list(self._df.loc[self._df[slo].notna()].index)

            if not swing_highs or not swing_lows:
                self.logger.warning("No swing highs or lows found")
                self._df[rg] = 0
                self._df[flr] = np.nan
                self._df[clg] = np.nan
                self._df[rg_ch] = np.nan
                return

            loop_size = max(len(swing_highs), len(swing_lows))

            for i in range(loop_size):
                # Handle asymmetric lists
                if i < len(swing_lows):
                    s_lo_ix, s_lo = swing_lows_ix[i], swing_lows[i]
                else:
                    s_lo_ix, s_lo = swing_lows_ix[-1], swing_lows[-1]

                if i < len(swing_highs):
                    s_hi_ix, s_hi = swing_highs_ix[i], swing_highs[i]
                else:
                    s_hi_ix, s_hi = swing_highs_ix[-1], swing_highs[-1]

                swing_max_ix = max(s_lo_ix, s_hi_ix)

                # Ceiling discovery
                if not ceiling_found:
                    top = self._df.loc[floor_ix_list[-1]:s_hi_ix, _h].max()
                    stdev_val = stdev[s_hi_ix] if s_hi_ix in stdev.index else stdev.iloc[-1]
                    if stdev_val > 0:
                        ceiling_test = round((s_hi - top) / stdev_val, 1)
                        if ceiling_test <= -threshold:
                            ceiling_found = True
                            floor_found = breakdown = breakout = False
                            threshold_test.append(ceiling_test)
                            ceiling_list.append(top)
                            ceiling_ix_list.append(
                                self._df.loc[floor_ix_list[-1]:s_hi_ix, _h].idxmax()
                            )
                            rg_ch_ix_list.append(s_hi_ix)
                            rg_ch_list.append(s_hi)

                elif ceiling_found:
                    close_high = self._df.loc[rg_ch_ix_list[-1]:swing_max_ix, _c].cummax()
                    self._df.loc[rg_ch_ix_list[-1]:swing_max_ix, rg] = np.sign(
                        close_high - rg_ch_list[-1]
                    )
                    if (self._df.loc[rg_ch_ix_list[-1]:swing_max_ix, rg] > 0).any():
                        ceiling_found = floor_found = breakdown = False
                        breakout = True

                if breakout:
                    brkout_high_ix = self._df.loc[rg_ch_ix_list[-1]:swing_max_ix, _c].idxmax()
                    brkout_low = self._df.loc[brkout_high_ix:swing_max_ix, _c].cummin()
                    self._df.loc[brkout_high_ix:swing_max_ix, rg] = np.sign(
                        brkout_low - rg_ch_list[-1]
                    )

                # Floor discovery
                if not floor_found:
                    bottom = self._df.loc[ceiling_ix_list[-1]:s_lo_ix, _l].min()
                    stdev_val = stdev[s_lo_ix] if s_lo_ix in stdev.index else stdev.iloc[-1]
                    if stdev_val > 0:
                        floor_test = round((s_lo - bottom) / stdev_val, 1)
                        if floor_test >= threshold:
                            floor_found = True
                            ceiling_found = breakdown = breakout = False
                            threshold_test.append(floor_test)
                            floor_list.append(bottom)
                            floor_ix_list.append(
                                self._df.loc[ceiling_ix_list[-1]:s_lo_ix, _l].idxmin()
                            )
                            rg_ch_ix_list.append(s_lo_ix)
                            rg_ch_list.append(s_lo)

                elif floor_found:
                    close_low = self._df.loc[rg_ch_ix_list[-1]:swing_max_ix, _c].cummin()
                    self._df.loc[rg_ch_ix_list[-1]:swing_max_ix, rg] = np.sign(
                        close_low - rg_ch_list[-1]
                    )
                    if (self._df.loc[rg_ch_ix_list[-1]:swing_max_ix, rg] < 0).any():
                        floor_found = breakout = False
                        breakdown = True

                if breakdown:
                    brkdwn_low_ix = self._df.loc[rg_ch_ix_list[-1]:swing_max_ix, _c].idxmin()
                    breakdown_rebound = self._df.loc[brkdwn_low_ix:swing_max_ix, _c].cummax()
                    self._df.loc[brkdwn_low_ix:swing_max_ix, rg] = np.sign(
                        breakdown_rebound - rg_ch_list[-1]
                    )

            # Populate final columns
            if len(floor_ix_list) > 1:
                self._df.loc[floor_ix_list[1:], flr] = floor_list[1:]
            if len(ceiling_ix_list) > 1:
                self._df.loc[ceiling_ix_list[1:], clg] = ceiling_list[1:]
            if rg_ch_ix_list:
                self._df.loc[rg_ch_ix_list, rg_ch] = rg_ch_list
                self._df[rg_ch] = self._df[rg_ch].ffill()

                # Fill remaining regime values
                if swing_max_ix in self._df.index:
                    if ceiling_found:
                        self._df.loc[swing_max_ix:, rg] = np.sign(
                            self._df.loc[swing_max_ix:, _c].cummax() - rg_ch_list[-1]
                        )
                    elif floor_found:
                        self._df.loc[swing_max_ix:, rg] = np.sign(
                            self._df.loc[swing_max_ix:, _c].cummin() - rg_ch_list[-1]
                        )
                    else:
                        self._df.loc[swing_max_ix:, rg] = np.sign(
                            self._df.loc[swing_max_ix:, _c].rolling(5).mean() - rg_ch_list[-1]
                        )

            self._df[rg] = self._df[rg].ffill().fillna(0)
            self.logger.info("regime_floor_ceiling completed")

        except Exception as e:
            self.logger.exception(f"Error in regime_floor_ceiling: {e}")
            raise

    def _swings(
        self,
        relative: bool,
        lvl: int,
        vlty_n: int,
        dgt: int,
        d_vol: float,
        dist_pct: float,
        retrace_pct: float,
        r_vol: float
    ) -> None:
        """
        Perform full swing analysis.
        """
        self.logger.info(f"Starting swings analysis (relative={relative})")

        try:
            _o, _h, _l, _c = self._get_ohlc_columns(relative=relative)
            rt_lo, rt_hi, slo, shi, rg, clg, flr, rg_ch = self._get_regime_columns(
                lvl, relative
            )

            self._historical_swings(relative=relative)
            self._cleanup_latest_swing(shi, slo, rt_hi, rt_lo)

            ud, bs, bs_dt, _rt, _swg, hh_ll, hh_ll_dt = self._latest_swing_variables(
                shi, slo, rt_hi, rt_lo, _h, _l, _c
            )

            if hh_ll_dt is not None:
                atr = self._average_true_range(_h, _l, _c, n=vlty_n)
                vlty = round(atr.loc[hh_ll_dt], dgt) if hh_ll_dt in atr.index else 0

                dist_vol = d_vol * vlty
                _sign = self._test_distance(ud, bs, hh_ll, dist_vol, dist_pct)

                self._retest_swing(_sign, _rt, hh_ll_dt, hh_ll, _c, _swg)

                retrace_vol = r_vol * vlty
                self._retracement_swing(
                    _sign, _swg, _c, hh_ll_dt, hh_ll, vlty, retrace_vol, retrace_pct
                )

            self.logger.info("Completed swings analysis")

        except Exception as e:
            self.logger.exception(f"Error in swings: {e}")
            raise

    def compute(
        self,
        lvl: int = DEFAULT_LEVEL,
        vlty_n: int = DEFAULT_VOLATILITY_WINDOW,
        dgt: int = DEFAULT_DIGITS,
        d_vol: float = DEFAULT_DISTANCE_VOLATILITY,
        dist_pct: float = DEFAULT_DISTANCE_PCT,
        retrace_pct: float = DEFAULT_RETRACE_PCT,
        r_vol: float = DEFAULT_RETRACE_VOL,
        threshold: float = DEFAULT_THRESHOLD,
        relative: bool = False,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Compute floor/ceiling regime.

        Args:
            lvl: Swing level (default: 1)
            vlty_n: Volatility window for ATR (default: 63)
            dgt: Decimal digits for rounding (default: 2)
            d_vol: Distance volatility multiplier (default: 5)
            dist_pct: Distance percentage threshold (default: 0.05)
            retrace_pct: Retracement percentage (default: 0.05)
            r_vol: Retracement volatility multiplier (default: 2.5)
            threshold: Floor/ceiling threshold (default: 1.5)
            relative: If True, use relative OHLC columns
            inplace: If True, modify internal DataFrame

        Returns:
            DataFrame with regime columns:
                - hi{n}, lo{n}: Swing high/low levels
                - rg: Regime signal (1, -1, 0)
                - flr: Floor levels
                - clg: Ceiling levels
                - rg_ch: Regime change levels
        """
        self.logger.info(
            f"Computing floor_ceiling regime: lvl={lvl}, threshold={threshold}"
        )

        # Validate parameters
        if lvl < 1:
            raise ValueError("lvl must be >= 1")
        self._validate_window(vlty_n, "vlty_n")
        if threshold <= 0:
            raise ValueError("threshold must be positive")

        # Work on copy if not inplace
        if not inplace:
            self._df = self._df.copy()

        try:
            _o, _h, _l, _c = self._get_ohlc_columns(relative=relative)
            rt_lo, rt_hi, slo, shi, rg, clg, flr, rg_ch = self._get_regime_columns(
                lvl, relative
            )

            # Initialize columns
            self._df[rg] = np.nan
            self._df[flr] = np.nan
            self._df[clg] = np.nan
            self._df[rg_ch] = np.nan

            self._swings(
                relative=relative,
                lvl=lvl,
                vlty_n=vlty_n,
                dgt=dgt,
                d_vol=d_vol,
                dist_pct=dist_pct,
                retrace_pct=retrace_pct,
                r_vol=r_vol
            )

            stdev = self._df[_c].rolling(vlty_n).std(ddof=0)

            self._regime_floor_ceiling(
                _h=_h, _l=_l, _c=_c,
                slo=slo, shi=shi,
                flr=flr, clg=clg,
                rg=rg, rg_ch=rg_ch,
                stdev=stdev,
                threshold=threshold
            )

            # Count regime signals
            regime_col = self._df[rg]
            self.logger.info(
                f"Floor/ceiling computed: "
                f"{(regime_col == 1).sum()} bullish, "
                f"{(regime_col == -1).sum()} bearish, "
                f"{(regime_col == 0).sum()} neutral"
            )

            return self._df

        except Exception as e:
            self.logger.exception(f"Error in compute: {e}")
            raise

    def floor_ceiling(
        self,
        lvl: int = 1,
        threshold: float = 1.5,
        vlty_n: int = 63,
        relative: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute floor/ceiling regime (convenience method).

        Args:
            lvl: Swing level (default: 1)
            threshold: Floor/ceiling threshold (default: 1.5)
            vlty_n: Volatility window (default: 63)
            relative: If True, use relative columns
            **kwargs: Additional parameters for compute()

        Returns:
            DataFrame with floor/ceiling regime columns

        Example:
            >>> detector = FloorCeilingRegime(df)
            >>> result = detector.floor_ceiling(lvl=1, threshold=2.0)
            >>> signal = result['rg']  # Regime signal
        """
        return self.compute(
            lvl=lvl,
            threshold=threshold,
            vlty_n=vlty_n,
            relative=relative,
            inplace=False,
            **kwargs
        )

    def get_signal_column(self, relative: bool = False) -> str:
        """
        Get the regime signal column name.

        Args:
            relative: If True, return relative column name

        Returns:
            Signal column name ('rg' or 'rrg')
        """
        return 'rrg' if relative else 'rg'
