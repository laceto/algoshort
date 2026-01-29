import pandas as pd
import numpy as np
from typing import Optional, Union
from algoshort.returns import ReturnsCalculator

class PositionSizing:
    """
    A class to calculate metrics and position sizes for trading strategies, including equity risk, risk appetite,
    drawdown, expectancy metrics, position sizing calculations (shares, target price, partial exits), and
    comparison of position sizing algorithms (constant, concave, convex, equal weight) for both long and short positions.
    Shares are positive for long positions and negative for short positions.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC columns ['open', 'high', 'low', 'close']
            or ['Open', 'High', 'Low', 'Close'] for absolute mode, and
            ['r_open', 'r_high', 'r_low', 'r_close'] or ['rOpen', 'rHigh', 'rLow', 'rClose']
            for relative mode.

    Raises:
        KeyError: If required OHLC columns are not found.
        ValueError: If DataFrame is empty or contains non-numeric data.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._cache: dict[str, pd.Series] = {}  # Cache for calculations like peak equity and watermark

    def _peak_equity(self, eqty: pd.Series) -> pd.Series:
        """
        Calculates peak equity as the cumulative maximum of the equity series.

        Args:
            eqty (pd.Series): Equity series (e.g., close price or computed equity curve).

        Returns:
            pd.Series: Peak equity series.
        """
        return eqty.expanding().max()

    def _risk_budget(self, eqty: pd.Series, appetite: pd.Series, fx: float) -> pd.Series:
        """
        Calculates risk budget as peak_equity * appetite * fx.

        Args:
            eqty (pd.Series): Equity series (e.g., close price or computed equity curve).
            appetite (pd.Series): Risk appetite series (e.g., from get_risk_appetite).
            fx (float): Currency conversion factor.

        Returns:
            pd.Series: Risk budget series.
        """
        return self._peak_equity(eqty) * appetite * fx

    def _risk_unit(self, price: pd.Series, stop_loss: pd.Series) -> pd.Series:
        """
        Calculates Van Tharp's R: distance to stop loss in dollars.

        Args:
            price (pd.Series): Current price series.
            stop_loss (pd.Series): Stop-loss price series.

        Returns:
            pd.Series: Risk unit (R) series.
        """
        return price - stop_loss

    def _shares_roundlot(self, budget: pd.Series, fx: float, r: pd.Series, lot: float) -> pd.Series:
        """
        Calculates number of shares rounded to the nearest lot size.

        Args:
            budget (pd.Series): Risk budget series.
            fx (float): Currency conversion factor.
            r (pd.Series): Risk unit (R) series.
            lot (float): Lot size (e.g., 100 for stocks).

        Returns:
            pd.Series: Number of shares, rounded to lot size.
        """
        fx_budget = fx * budget
        rounded_shares = fx_budget // (r * lot)
        shares = rounded_shares * lot
        return shares

    def _target_price(self, price: pd.Series, stop_loss: pd.Series, r_multiplier: float) -> pd.Series:
        """
        Calculates target price as price + (price - stop_loss) * r_multiplier.

        Args:
            price (pd.Series): Current price series.
            stop_loss (pd.Series): Stop-loss price series.
            r_multiplier (float): Risk-reward multiplier (e.g., 2.0 for 2R target).

        Returns:
            pd.Series: Target price series.
        """
        r = self._risk_unit(price, stop_loss)
        return price + r * r_multiplier

    def _partial_exit(self, qty: pd.Series, r_multiplier: float) -> pd.Series:
        """
        Calculates partial exit quantity as qty / r_multiplier.

        Args:
            qty (pd.Series): Quantity of shares or contracts.
            r_multiplier (float): Risk-reward multiplier (e.g., 2.0 for 2R target).

        Returns:
            pd.Series: Partial exit quantity series.
        """
        fraction = np.where(qty * r_multiplier != 0, qty / r_multiplier, 0)
        return pd.Series(fraction, index=qty.index)
    
    def _equity_risk_shares(self, price: pd.Series, stop_loss: pd.Series, eqty: pd.Series, 
                            risk: float, fx: float, lot: float) -> pd.Series:
        """
        Calculates number of shares based on equity and fixed risk fraction, rounded to lot size, 
        for both long and short positions. Shares are positive for long positions (signal > 0) 
        and negative for short positions (signal < 0).

        Args:
            price (pd.Series): Current price series.
            stop_loss (pd.Series): Stop-loss price series.
            eqty (pd.Series): Equity series (e.g., portfolio equity or price).
            risk (float): Fixed risk fraction (e.g., -0.005 for 0.5% risk).
            fx (float): Currency conversion factor.
            lot (float): Lot size (e.g., 100 for stocks).

        Returns:
            pd.Series: Number of shares, rounded to lot size, positive for long, negative for short.

        Raises:
            ValueError: If inputs are non-numeric or invalid.
        """
        try:
            # Validate inputs
            if not all(np.issubdtype(s.dtype, np.number) for s in [price, stop_loss, eqty]):
                raise ValueError("price, stop_loss, eqty, and signal must contain numeric data.")
            if not all(len(s) == len(price) for s in [stop_loss, eqty]):
                raise ValueError("price, stop_loss, eqty, and signal must have the same length.")
            if risk >= 0:
                raise ValueError("risk must be negative (e.g., -0.005 for 0.5% risk).")
            if fx <= 0:
                raise ValueError("fx must be positive.")
            if lot <= 0:
                raise ValueError("lot must be positive.")

            r = np.abs(price - stop_loss)  # Absolute risk unit (R) for long or short positions
            r = price - stop_loss
            budget = eqty * risk * fx
            shares = np.where(r != 0, (budget // (r * lot)) * lot, 0)
            # Apply sign based on signal: positive for long (signal > 0), negative for short (signal < 0)
            signed_shares = shares
            return pd.Series(signed_shares, index=price.index)
        except Exception as e:
            raise ValueError(f"Error computing equity risk shares: {str(e)}")

    def _pyramid(self, position: Union[int, np.ndarray], root: Union[float, np.ndarray] = 2) -> Union[float, np.ndarray]:
        """
        Calculates the risk amortization factor based on the number of active positions.

        Args:
            position (Union[int, np.ndarray]): Number of active positions (non-negative).
            root (Union[float, np.ndarray]): Power for amortization (e.g., 1 for linear, 2 for square root, position for aggressive).
                                             Must be positive and match position's shape if array.

        Returns:
            Union[float, np.ndarray]: Amortization factor as 1 / (1 + position) ** (1/root).

        Raises:
            ValueError: If position is negative or root is non-positive or shape mismatch.
        """
        try:
            # Convert inputs to numpy arrays for consistent handling
            position = np.asarray(position)
            root = np.asarray(root)

            # Validate inputs
            if np.any(position < 0):
                raise ValueError("position must be non-negative.")
            if np.any(root <= 0):
                raise ValueError("root must be positive.")
            if root.size > 1 and root.shape != position.shape:
                raise ValueError("root must be a scalar or match position's shape.")

            return 1 / (1 + position) ** (1 / root)
        except Exception as e:
            raise ValueError(f"Error computing pyramid factor: {str(e)}")

    def _amortized_weight(self, raw_weight: float, amortization: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Adjusts the raw position weight by the amortization factor.

        Args:
            raw_weight (float): Initial position weight (e.g., 0.05 for 5%).
            amortization (Union[float, np.ndarray]): Amortization factor from pyramid().

        Returns:
            Union[float, np.ndarray]: Adjusted weight as raw_weight * amortization.

        Raises:
            ValueError: If raw_weight is non-positive or amortization is invalid.
        """
        try:
            # Validate inputs
            if raw_weight <= 0:
                raise ValueError("raw_weight must be positive.")
            amortization = np.asarray(amortization)
            if np.any(amortization < 0):
                raise ValueError("amortization factor must be non-negative.")

            return raw_weight * amortization
        except Exception as e:
            raise ValueError(f"Error computing amortized weight: {str(e)}")

    # def get_equity_risk(self, df: pd.DataFrame, tolerance: float = -0.1, mn: float = -0.0025, 
    #                     mx: float = -0.0075, span: int = 5,
    #                     relative: bool = False, inplace: bool = False) -> pd.DataFrame:
    #     """
    #     Calculates equity risk metrics based on the close price, including peak equity, constant/convex/concave risk, tolerance, and drawdown.

    #     Args:
    #         df (pd.DataFrame): DataFrame containing OHLC data.
    #         tolerance (float): Drawdown tolerance as a negative fraction (e.g., -0.1 for 10% drawdown). Defaults to -0.1.
    #         mn (float): Minimum risk level (e.g., -0.0025 for -0.25%). Defaults to -0.0025.
    #         mx (float): Maximum risk level (e.g., -0.0075 for -0.75%). Defaults to -0.0075.
    #         span (int): Window for exponential moving average in risk appetite. Defaults to 5.
    #         relative (bool): If True, use relative OHLC columns ('r_close', etc.); else absolute ('close', etc.). Defaults to False.
    #         inplace (bool): If True, modify input DataFrame; else return a new one. Defaults to False.

    #     Returns:
    #         pd.DataFrame: DataFrame with additional columns:
    #             - 'peak_eqty': Cumulative maximum of the close price.
    #             - 'constant_risk': Constant risk as -close * (mn + mx)/2
    #             - 'convex_risk': Convex risk appetite scaled by peak equity.
    #             - 'concave_risk': Concave risk appetite scaled by peak equity.
    #             - 'tolerance': Peak equity * (1 + tolerance).
    #             - 'drawdown': Drawdown as (close / peak_eqty - 1).

    #     Raises:
    #         KeyError: If required close column is missing.
    #         ValueError: If DataFrame is empty, parameters are invalid, or data types are non-numeric.
    #     """
    #     try:
    #         # Validate inputs
    #         if df.empty:
    #             raise ValueError("Input DataFrame is empty.")
    #         if tolerance >= 0:
    #             raise ValueError("tolerance must be negative (e.g., -0.1 for 10% drawdown).")
    #         if mn > 0 or mx > 0:
    #             raise ValueError("mn and mx must be non-positive.")
    #         if mn < mx:
    #             raise ValueError("mn must be greater than or equal to mx.")
    #         if span < 1:
    #             raise ValueError("span must be positive.")

    #         # Get OHLC column names
    #         _o, _h, _l, _c = ReturnsCalculator._lower_upper_OHLC(self.data, relative=relative)
            
    #         # Validate close column
    #         if _c not in df.columns:
    #             raise KeyError(f"Close column '{_c}' not found in DataFrame.")
    #         if not np.issubdtype(df[_c].dtype, np.number):
    #             raise ValueError(f"Close column '{_c}' must contain numeric data.")

    #         # Create working DataFrame
    #         result_df = df if inplace else df.copy()

    #         # Calculate average risk
    #         avg = (mn + mx) / 2

    #         # Calculate peak equity
    #         result_df['peak_eqty'] = self._peak_equity(result_df[_c])

    #         # Calculate constant risk, 
    #         result_df['constant_risk_appetite'] = avg
    #         result_df['constant_risk'] = self._risk_budget(-result_df[_c], result_df['constant_risk_appetite'])
            

    #         # Calculate convex and concave risk using risk_appetite
    #         result_df['convex_risk_appetite'] = -self.get_risk_appetite(result_df, price_col=_c, tolerance=tolerance, 
    #                                                           min_risk=mn, max_risk=mx, span=span, shape=1, 
    #                                                           inplace=False)[f'{_c}_risk_appetite']
            
    #         result_df['convex_risk'] = self._risk_budget(result_df['peak_eqty'], result_df['convex_risk_appetite'])

    #         result_df['concave_risk_appetite'] = -self.get_risk_appetite(result_df, price_col=_c, tolerance=tolerance, 
    #                                                            min_risk=mn, max_risk=mx, span=span, shape=-1, 
    #                                                            inplace=False)[f'{_c}_risk_appetite']
    #         result_df['concave_risk'] = self._risk_budget(result_df['peak_eqty'], result_df['concave_risk_appetite'])

    #         # Calculate tolerance and drawdown
    #         # result_df['tolerance'] = result_df['peak_eqty'] * (1 + tolerance)
    #         # result_df['drawdown'] = result_df[_c] / result_df['peak_eqty'] - 1

    #         return result_df

    #     except Exception as e:
    #         raise ValueError(f"Error computing equity risk metrics: {str(e)}")

    def risk_appetite(
        self,
        df: pd.DataFrame,
        eqty: pd.Series,
        tolerance: float,
        min_risk: float,
        max_risk: float,
        span: int = 5,
        shape: int = 0,
        return_all: bool = False
    ) -> pd.Series:
        """
        Calculate dynamic risk appetite based on drawdown from peak equity.

        Parameters:
            eqty: pd.Series - equity curve (can be price or portfolio equity)
            tolerance: float - max acceptable drawdown (negative, e.g. -0.10)
            min_risk: float - least aggressive risk level (e.g. -0.0025)
            max_risk: float - most aggressive risk level (e.g. -0.0075)
            span: int - EMA smoothing period for rebased drawdown
            shape: int - 
                1  → convex (aggressive when drawdown is small)
                -1  → concave (conservative when drawdown is small)
                0  → linear

        Returns:
            pd.Series - risk appetite values (negative), same index as eqty
        """
        if not isinstance(eqty, pd.Series):
            eqty = pd.Series(eqty)

        if eqty.empty:
            raise ValueError("Equity series is empty")

        if tolerance >= 0:
            raise ValueError("tolerance must be negative (max drawdown tolerance)")
        if min_risk > 0 or max_risk > 0:
            raise ValueError("min_risk and max_risk must be ≤ 0")
        if min_risk < max_risk:
            raise ValueError("min_risk should be ≤ max_risk (more negative = more aggressive)")
        if span < 1:
            raise ValueError("span must be ≥ 1")
        if shape not in (-1, 0, 1):
            raise ValueError("shape must be -1, 0 or 1")

        # ── Core calculation ────────────────────────────────────────────────────────
        watermark = eqty.cummax()                      # all-time high
        drawdown = eqty / watermark - 1                # current drawdown
        ddr_raw = drawdown / tolerance                  # normalized (positive when down)
        ddr = 1 - np.minimum(ddr_raw,1) # drawdown rebased to tolerance from 0 to 1  
        avg_ddr = ddr.ewm(span=span, adjust=False).mean()    # smoothed

        # ── Shape transformation ────────────────────────────────────────────────────
        if shape == 1:
            power = max_risk / min_risk if min_risk != 0 else 1.0  # convex
        elif shape == -1:
            power = min_risk / max_risk if max_risk != 0 else 1.0  # concave
        else:
            power = 1.0                                            # linear

        ddr_power = avg_ddr.pow(power)

        # ── Final risk appetite ─────────────────────────────────────────────────────
        appetite = min_risk + (max_risk - min_risk) * ddr_power

    # Preserve index & name
        # appetite.name = "risk_appetite"
        # avg_ddr.name  = "avg_ddr"
        # ddr.name      = "ddr"
        # drawdown.name = "drawdown"
        # watermark.name = "watermark"
        # ddr_power.name = "ddr_power"

        if return_all:
            return {
                'appetite':    appetite,
                'avg_ddr':     avg_ddr,
                'ddr':         ddr,
                'drawdown':    drawdown,
                'watermark':   watermark,
                'ddr_power':   ddr_power,
                'power':       pd.Series(power, index=eqty.index, name=f"power"),
                'min_risk':    pd.Series(min_risk, index=eqty.index, name=f"min_risk"),
                'max_risk':    pd.Series(max_risk, index=eqty.index, name=f"max_risk")
            }
        else:
            return appetite

    def get_risk_appetite(
        self,
        df: pd.DataFrame,
        price_col: str,
        tolerance: float = -0.1,
        min_risk: float = -0.0025,
        max_risk: float = -0.0075,
        span: int = 5,
        shape: int = -1,
        inplace: bool = False,
        prefix: str = None,
    ) -> pd.DataFrame:
        """
        Calculates risk appetite for ALL THREE shapes simultaneously (linear, convex, concave)
        and exposes intermediate steps + all three final risk appetite series.

        Added columns (with prefix or price_col + '_'):
        - watermark
        - drawdown
        - ddr
        - tolerance           # watermark * (1 + tolerance)
        - avg_ddr
        - power_linear
        - power_convex
        - power_concave
        - min_risk
        - max_risk
        - ddr_power_linear
        - ddr_power_convex
        - ddr_power_concave
        - risk_appetite_linear
        - risk_appetite_convex
        - risk_appetite_concave

        Args:
            df: DataFrame containing the price column
            price_col: name of the price series
            tolerance: drawdown tolerance (negative)
            min_risk: least aggressive risk level (closest to zero)
            max_risk: most aggressive risk level (most negative)
            span: EMA span for smoothing
            inplace: modify df in place if True
            prefix: custom prefix for added columns (default: f"{price_col}_")

        Returns:
            DataFrame enriched with all intermediate and three final risk appetite series
        """
        # Validation
        if df.empty:
            raise ValueError("DataFrame is empty")
        if price_col not in df.columns:
            raise KeyError(f"Column {price_col!r} not found")
        if not pd.api.types.is_numeric_dtype(df[price_col]):
            raise ValueError(f"Column {price_col!r} must be numeric")
        if tolerance >= 0:
            raise ValueError("tolerance must be negative")
        if min_risk > 0 or max_risk > 0:
            raise ValueError("min_risk and max_risk must be ≤ 0")
        if min_risk < max_risk:
            raise ValueError("min_risk should be ≤ max_risk (more negative = more aggressive)")
        if span < 1:
            raise ValueError("span must be ≥ 1")

        result_df = df if inplace else df.copy()
        price = result_df[price_col]

        # Column prefix
        p = prefix if prefix is not None else f"{price_col}_"

        # ── Watermark ───────────────────────────────────────────────────────────────
        cache_key = f"watermark_{price_col}"
        if cache_key in self._cache:
            watermark = self._cache[cache_key]
        else:
            watermark = price.cummax()
            self._cache[cache_key] = watermark

        # result_df[f"{p}watermark"] = watermark

        # ── Drawdown & Tolerance line ───────────────────────────────────────────────
        drawdown = price / watermark - 1
        # result_df[f"{p}drawdown"] = drawdown

        tolerance_line = watermark * (1 + tolerance)
        # result_df[f"{p}tolerance"] = tolerance_line

        # ── Rebased drawdown (ddr) ──────────────────────────────────────────────────
        ddr_raw = drawdown / tolerance
        ddr_clipped = ddr_raw.clip(upper=1.0)
        ddr = (1 - ddr_clipped).clip(lower=0.0)
        # result_df[f"{p}ddr"] = ddr

        # ── Smoothed rebased drawdown ───────────────────────────────────────────────
        avg_ddr = ddr.ewm(span=span, adjust=False).mean()

            # Shape of the curve
        if shape == 1: # 
            _power = max_risk/min_risk # convex 
        elif shape == -1 :
            _power = min_risk/max_risk # concave
        else:
            _power = 1 # raw, straight line
        ddr_power = avg_ddr ** _power # ddr 

        risk_appetite =min_risk + (max_risk - min_risk) * ddr_power 

        return risk_appetite

    # def get_drawdown(self, df: pd.DataFrame, equity_col: str, dd_tolerance: float = -0.1, 
    #                  inplace: bool = False) -> pd.DataFrame:
    #     """
    #     Calculates drawdown metrics for an equity curve.

    #     Args:
    #         df (pd.DataFrame): DataFrame containing the equity curve column.
    #         equity_col (str): Name of the equity curve column (e.g., 'bo_5_equity', 'sma_358_equity').
    #         dd_tolerance (float): Drawdown tolerance as a negative fraction (e.g., -0.1 for 10% drawdown). Defaults to -0.1.
    #         inplace (bool): If True, modify input DataFrame; else return a new one. Defaults to False.

    #     Returns:
    #         pd.DataFrame: DataFrame with additional columns:
    #             - '<equity_col>_peak': Cumulative maximum of the equity curve.
    #             - '<equity_col>_tolerance': Peak equity * (1 + dd_tolerance).
    #             - '<equity_col>_drawdown': Drawdown as (equity / peak_equity - 1).

    #     Raises:
    #         KeyError: If equity_col is not found in DataFrame.
    #         ValueError: If DataFrame is empty, dd_tolerance is non-negative, or equity_col contains non-numeric data.
    #     """
    #     try:
    #         # Validate inputs
    #         if df.empty:
    #             raise ValueError("Input DataFrame is empty.")
    #         if equity_col not in df.columns:
    #             raise KeyError(f"Equity column '{equity_col}' not found in DataFrame.")
    #         if not np.issubdtype(df[equity_col].dtype, np.number):
    #             raise ValueError(f"Equity column '{equity_col}' must contain numeric data.")
    #         if dd_tolerance >= 0:
    #             raise ValueError("dd_tolerance must be negative (e.g., -0.1 for 10% drawdown).")

    #         # Create working DataFrame
    #         result_df = df if inplace else df.copy()

    #         # Cache key for peak equity
    #         cache_key_peak = f"peak_{equity_col}"

    #         # Calculate peak equity (cumulative max)
    #         if cache_key_peak in self._cache:
    #             peak_equity = self._cache[cache_key_peak]
    #         else:
    #             peak_equity = result_df[equity_col].cummax()
    #             self._cache[cache_key_peak] = peak_equity

    #         # Calculate drawdown metrics
    #         result_df[f'{equity_col}_peak'] = peak_equity
    #         result_df[f'{equity_col}_tolerance'] = peak_equity * (1 + dd_tolerance)
    #         result_df[f'{equity_col}_drawdown'] = result_df[equity_col] / peak_equity - 1

    #         return result_df

    #     except Exception as e:
    #         raise ValueError(f"Error computing drawdown metrics: {str(e)}")

    # ────────────────────────────────────────────────────────────────
    # Helper functions for compare_position_sizing
    # ────────────────────────────────────────────────────────────────

    # def _initialize_equity_and_shares(
    #     df: pd.DataFrame,
    #     starting_capital: float,
    #     strategies: dict
    # ) -> pd.DataFrame:
    #     """Initialize equity curve and shares columns for all strategies"""
    #     for name, info in strategies.items():
    #         df[name] = float(starting_capital)
    #         df[f"shs_{info['short']}"] = 0.0
    #     return df


    # def _update_all_equity_curves(
    #     df: pd.DataFrame,
    #     idx_prev,
    #     idx_curr,
    #     daily_change_col: str,
    #     current_shares: dict,
    #     strategies: dict
    # ):
    #     """Update equity for every strategy based on previous equity + P&L"""
    #     chg = df.at[idx_curr, daily_change_col]

    #     for strat_name, info in strategies.items():
    #         short = info["short"]
    #         prev_equity = df.at[idx_prev, strat_name]
    #         df.at[idx_curr, strat_name] = prev_equity + chg * current_shares[short]
    #         df.at[idx_curr, f"shs_{short}"] = current_shares[short]


    # def _update_position_count(
    #     signal_prev: float,
    #     signal_curr: float,
    #     current_count: int
    # ) -> tuple[int, bool]:
    #     """Determine if we entered or exited a position and update count"""
    #     entered = (signal_prev == 0) and (signal_curr != 0)
    #     exited  = (signal_curr == 0) and (signal_prev != 0)

    #     if entered:
    #         current_count += 1
    #     elif exited:
    #         current_count = max(0, current_count - 1)

    #     in_position = current_count > 0
    #     return current_count, in_position


    # def _calculate_equal_and_amortized_shares(
    #     df,
    #     idx_curr,
    #     equal_weight: float,
    #     amortized_root: float,
    #     pyramid_func,
    #     amortized_weight_func,
    #     position_count: int,
    #     fx: float,
    #     px: float,
    #     lot: float
    # ) -> tuple[float, float]:
    #     """Calculate shares for equal-weight and amortized strategies"""
    #     current_equity_eql = df.at[idx_curr, "equal"]
    #     current_equity_amz = df.at[idx_curr, "amortized"]

    #     shs_eql = (current_equity_eql * equal_weight * fx // (px * lot)) * lot

    #     amortization_factor = pyramid_func(position_count, root=amortized_root)
    #     am_weight = amortized_weight_func(equal_weight, amortization_factor)
    #     shs_amz = (current_equity_amz * am_weight * fx // (px * lot)) * lot

    #     return shs_eql, shs_amz


    # def _calculate_risk_based_shares(
    #     self,
    #     df,
    #     idx_curr,
    #     strat_name: str,
    #     px: float,
    #     sl: float,
    #     fx: float,
    #     lot: float,
    #     tolerance: float,
    #     min_risk: float,
    #     max_risk: float,
    #     span: int,
    #     *,                   # ← keyword-only separator
    #     constant_risk: float = None
    # ) -> float:
    #     """
    #     Calculate position size for constant / concave / convex strategies.
    #     For constant we use fixed risk; for others we recompute appetite.
    #     """
    #     equity_now = df.at[idx_curr, strat_name]

    #     if strat_name == "constant":
    #         risk_level = constant_risk
    #     else:
    #         # Warning: expensive — called on every bar when entering
    #         appetite_df = self.get_risk_appetite(
    #             df.iloc[: idx_curr + 1],
    #             price_col=strat_name,
    #             tolerance=tolerance,
    #             min_risk=min_risk,
    #             max_risk=max_risk,
    #             span=span,
    #             inplace=False
    #         )
    #         # risk_level = appetite_df[f"{strat_name}_risk_appetite"].iloc[-1]

    #         if strat_name == "concave":
    #             risk_level = appetite_df["risk_appetite_concave"].iloc[-1]
    #         elif strat_name == "convex":
    #             risk_level = appetite_df["risk_appetite_convex"].iloc[-1]
    #         else:
    #             risk_level = constant_risk

    #     shares_series = self._equity_risk_shares(
    #         price=pd.Series(px, index=[df.index[idx_curr]]),
    #         stop_loss=pd.Series(sl, index=[df.index[idx_curr]]),
    #         eqty=pd.Series(equity_now, index=[df.index[idx_curr]]),
    #         risk=risk_level,
    #         fx=fx,
    #         lot=lot
    #     )

    #     return shares_series.iloc[0]


    # def _get_direction(signal_value: float) -> float:
    #     """+1 for long, -1 for short, 0 otherwise"""
    #     return np.sign(signal_value) if signal_value != 0 else 0
        
    # def calculate_position_size(self, df: pd.DataFrame, signal: str, price_col: str = 'close', 
    #                            stop_loss_col: str = None, equity_col: str = None, 
    #                            fx: float = 1.0, lot: float = 100, r_multiplier: float = 2.0, 
    #                            tolerance: float = -0.1, min_risk: float = -0.0025, 
    #                            max_risk: float = -0.0075, span: int = 5, shape: int = 0, 
    #                            use_equity_risk: bool = False, risk: float = -0.005, 
    #                            inplace: bool = False) -> pd.DataFrame:
    #     """
    #     Calculates position sizing metrics, including risk budget, risk unit, shares, target price, partial exits,
    #     and optional equity-based shares using a fixed risk fraction. Shares are positive for long positions and
    #     negative for short positions when using equity risk.

    #     Args:
    #         df (pd.DataFrame): DataFrame containing price, stop-loss, signal, and optional equity columns.
    #         signal (str): Name of the signal column (e.g., 'bo_5', 'tt_52') used for naming output columns and trade direction.
    #         price_col (str): Name of the price column (e.g., 'close'). Defaults to 'close'.
    #         stop_loss_col (str, optional): Name of the stop-loss column (e.g., 'bo_5_stop_loss'). If None, skips position sizing.
    #         equity_col (str, optional): Name of the equity column (e.g., 'bo_5_equity'). If None, uses price_col.
    #         fx (float): Currency conversion factor. Defaults to 1.0.
    #         lot (float): Lot size for rounding shares (e.g., 100 for stocks). Defaults to 100.
    #         r_multiplier (float): Risk-reward multiplier for target price and partial exit (e.g., 2.0 for 2R). Defaults to 2.0.
    #         tolerance (float): Drawdown tolerance for risk appetite (e.g., -0.1 for 10% drawdown). Defaults to -0.1.
    #         min_risk (float): Minimum risk level for risk appetite (e.g., -0.0025). Defaults to -0.0025.
    #         max_risk (float): Maximum risk level for risk appetite (e.g., -0.0075). Defaults to -0.0075.
    #         span (int): Window for exponential moving average in risk appetite. Defaults to 5.
    #         shape (int): Shape of risk appetite curve: 1 (convex), -1 (concave), 0 (linear). Defaults to 0.
    #         use_equity_risk (bool): If True, compute shares using fixed risk fraction instead of risk appetite. Defaults to False.
    #         risk (float): Fixed risk fraction for equity-based shares (e.g., -0.005 for 0.5% risk). Defaults to -0.005.
    #         inplace (bool): If True, modify input DataFrame; else return a new one. Defaults to False.

    #     Returns:
    #         pd.DataFrame: DataFrame with additional columns:
    #             - '<signal>_peak_equity': Peak equity of the specified equity or price column.
    #             - '<signal>_risk_appetite': Risk appetite from get_risk_appetite.
    #             - '<signal>_risk_budget': Risk budget as peak_equity * risk_appetite * fx.
    #             - '<signal>_risk_unit': Risk unit (R) as price - stop_loss.
    #             - '<signal>_shares': Number of shares rounded to lot size (if use_equity_risk=False).
    #             - '<signal>_equity_risk_shares': Number of shares using fixed risk fraction (if use_equity_risk=True, positive for long, negative for short).
    #             - '<signal>_target_price': Target price as price + R * r_multiplier.
    #             - '<signal>_partial_exit': Partial exit quantity as shares / r_multiplier.

    #     Raises:
    #         KeyError: If required columns (price_col, stop_loss_col, equity_col, signal) are missing.
    #         ValueError: If DataFrame is empty, parameters are invalid, or columns contain non-numeric data.
    #     """
    #     try:
    #         # Validate inputs
    #         if df.empty:
    #             raise ValueError("Input DataFrame is empty.")
    #         if price_col not in df.columns:
    #             raise KeyError(f"Price column '{price_col}' not found in DataFrame.")
    #         if stop_loss_col is not None and stop_loss_col not in df.columns:
    #             raise KeyError(f"Stop-loss column '{stop_loss_col}' not found in DataFrame.")
    #         if equity_col is not None and equity_col not in df.columns:
    #             raise KeyError(f"Equity column '{equity_col}' not found in DataFrame.")
    #         if signal not in df.columns:
    #             raise KeyError(f"Signal column '{signal}' not found in DataFrame.")
    #         if not np.issubdtype(df[price_col].dtype, np.number):
    #             raise ValueError(f"Price column '{price_col}' must contain numeric data.")
    #         if stop_loss_col is not None and not np.issubdtype(df[stop_loss_col].dtype, np.number):
    #             raise ValueError(f"Stop-loss column '{stop_loss_col}' must contain numeric data.")
    #         if equity_col is not None and not np.issubdtype(df[equity_col].dtype, np.number):
    #             raise ValueError(f"Equity column '{equity_col}' must contain numeric data.")
    #         if not np.issubdtype(df[signal].dtype, np.number):
    #             raise ValueError(f"Signal column '{signal}' must contain numeric data.")
    #         if fx <= 0 and use_equity_risk:
    #             raise ValueError("fx must be positive when use_equity_risk=True.")
    #         if lot <= 0:
    #             raise ValueError("lot must be positive.")
    #         if r_multiplier <= 0:
    #             raise ValueError("r_multiplier must be positive.")
    #         if tolerance >= 0:
    #             raise ValueError("tolerance must be negative (e.g., -0.1 for 10% drawdown).")
    #         if min_risk > 0 or max_risk > 0:
    #             raise ValueError("min_risk and max_risk must be non-positive.")
    #         if min_risk < max_risk:
    #             raise ValueError("min_risk must be greater than or equal to max_risk.")
    #         if span < 1:
    #             raise ValueError("span must be positive.")
    #         if risk >= 0:
    #             raise ValueError("risk must be negative (e.g., -0.005 for 0.5% risk).")

    #         # Create working DataFrame
    #         result_df = df if inplace else df.copy()

    #         # Get risk appetite
    #         appetite_df = self.get_risk_appetite(
    #             result_df, price_col=price_col, tolerance=tolerance, min_risk=min_risk,
    #             max_risk=max_risk, span=span, shape=shape, inplace=False
    #         )
    #         appetite = appetite_df[f'{price_col}_risk_appetite']

    #         # Select equity series (use equity_col if provided, else price_col)
    #         eqty = result_df[equity_col] if equity_col is not None else result_df[price_col]

    #         # Cache key for position sizing metrics
    #         cache_key_prefix = f"position_size_{signal}_{price_col}_{equity_col}_{fx}_{lot}_{r_multiplier}_{risk}"

    #         # Calculate peak equity
    #         cache_key_peak = f"{cache_key_prefix}_peak_equity"
    #         if cache_key_peak in self._cache:
    #             peak_eqty = self._cache[cache_key_peak]
    #         else:
    #             peak_eqty = self._peak_equity(eqty)
    #             self._cache[cache_key_peak] = peak_eqty
    #         result_df[f'{signal}_peak_equity'] = peak_eqty

    #         # Copy risk appetite
    #         result_df[f'{signal}_risk_appetite'] = appetite

    #         # Initialize output columns
    #         result_df[f'{signal}_risk_budget'] = np.nan
    #         result_df[f'{signal}_risk_unit'] = np.nan
    #         result_df[f'{signal}_shares'] = np.nan
    #         result_df[f'{signal}_equity_risk_shares'] = np.nan
    #         result_df[f'{signal}_target_price'] = np.nan
    #         result_df[f'{signal}_partial_exit'] = np.nan

    #         if stop_loss_col is not None:
    #             # Calculate risk budget
    #             cache_key_budget = f"{cache_key_prefix}_risk_budget"
    #             if cache_key_budget in self._cache:
    #                 risk_budget = self._cache[cache_key_budget]
    #             else:
    #                 risk_budget = self._risk_budget(eqty, appetite, fx)
    #                 self._cache[cache_key_budget] = risk_budget
    #             result_df[f'{signal}_risk_budget'] = risk_budget

    #             # Calculate risk unit (R)
    #             cache_key_risk_unit = f"{cache_key_prefix}_risk_unit"
    #             if cache_key_risk_unit in self._cache:
    #                 risk_unit = self._cache[cache_key_risk_unit]
    #             else:
    #                 risk_unit = self._risk_unit(result_df[price_col], result_df[stop_loss_col])
    #                 self._cache[cache_key_risk_unit] = risk_unit
    #             result_df[f'{signal}_risk_unit'] = risk_unit

    #             # Calculate shares (using risk appetite)
    #             cache_key_shares = f"{cache_key_prefix}_shares"
    #             if cache_key_shares in self._cache:
    #                 shares = self._cache[cache_key_shares]
    #             else:
    #                 shares = self._shares_roundlot(risk_budget, fx, risk_unit, lot)
    #                 self._cache[cache_key_shares] = shares
    #             result_df[f'{signal}_shares'] = shares

    #             # Calculate equity risk shares (using fixed risk fraction)
    #             if use_equity_risk:
    #                 cache_key_eqty_shares = f"{cache_key_prefix}_equity_risk_shares"
    #                 if cache_key_eqty_shares in self._cache:
    #                     eqty_shares = self._cache[cache_key_eqty_shares]
    #                 else:
    #                     eqty_shares = self._equity_risk_shares(
    #                         result_df[price_col], result_df[stop_loss_col], eqty, risk, fx, lot, result_df[signal]
    #                     )
    #                     self._cache[cache_key_eqty_shares] = eqty_shares
    #                 result_df[f'{signal}_equity_risk_shares'] = eqty_shares

    #             # Calculate target price
    #             cache_key_target = f"{cache_key_prefix}_target_price"
    #             if cache_key_target in self._cache:
    #                 target_price = self._cache[cache_key_target]
    #             else:
    #                 target_price = self._target_price(result_df[price_col], result_df[stop_loss_col], r_multiplier)
    #                 self._cache[cache_key_target] = target_price
    #             result_df[f'{signal}_target_price'] = target_price

    #             # Calculate partial exit
    #             cache_key_partial = f"{cache_key_prefix}_partial_exit"
    #             if cache_key_partial in self._cache:
    #                 partial_exit = self._cache[cache_key_partial]
    #             else:
    #                 partial_exit = self._partial_exit(shares, r_multiplier)
    #                 self._cache[cache_key_partial] = partial_exit
    #             result_df[f'{signal}_partial_exit'] = partial_exit

    #         return result_df

    #     except Exception as e:
    #         raise ValueError(f"Error computing position sizing metrics: {str(e)}")

    # def _initialize_equity_and_shares(
    #         self,
    #         df: pd.DataFrame,
    #         starting_capital: float,
    #         strategies: dict
    #     ) -> pd.DataFrame:
    #     """Initialize equity curve and shares columns for all strategies"""
    #     for name, info in strategies.items():
    #         df[name] = float(starting_capital)
    #         df[f"shs_{info['short']}"] = 0.0
    #     return df
    
    # def _update_position_count(
    #         signal_prev: float,
    #         signal_curr: float,
    #         current_count: int
    #     ) -> tuple[int, bool]:
    #     """Determine if we entered or exited a position and update count"""
    #     entered = (signal_prev == 0) and (signal_curr != 0)
    #     exited  = (signal_curr == 0) and (signal_prev != 0)

    #     if entered:
    #         position_count += 1
    #     elif exited:
    #         position_count = max(0, position_count - 1)

    #     in_position = position_count > 0
    #     return current_count, in_position

    # def _get_direction(signal_value: float) -> float:
    #     """+1 for long, -1 for short, 0 otherwise"""
    #     return np.sign(signal_value) if signal_value != 0 else 0

    # def _calculate_equal_and_amortized_shares(
    #         self,
    #         df,
    #         idx_curr,
    #         equal_weight: float,
    #         amortized_root: float,
    #         pyramid_func,
    #         amortized_weight_func,
    #         position_count: int,
    #         fx: float,
    #         px: float,
    #         lot: float
    #     ) -> tuple[float, float]:
    #     """Calculate shares for equal-weight and amortized strategies"""
    #     current_equity_eql = df.at[idx_curr, "equal"]
    #     current_equity_amz = df.at[idx_curr, "amortized"]

    #     shs_eql = (current_equity_eql * equal_weight * fx // (px * lot)) * lot

    #     amortization_factor = pyramid_func(position_count, root=amortized_root)
    #     am_weight = amortized_weight_func(equal_weight, amortization_factor)
    #     shs_amz = (current_equity_amz * am_weight * fx // (px * lot)) * lot

    #     return shs_eql, shs_amz


    # def simulate_position_sizing_comparison(
    #     self,
    #     df: pd.DataFrame,
    #     signal_col: str,
    #     price_col: str = 'close',
    #     stop_loss_col: str = 'stop_loss',
    #     chg_col: str = 'chg1D_fx',
    #     starting_capital: float = 1_000_000.0,
    #     lot: float = 100.0,
    #     tolerance: float = -0.10,
    #     min_risk: float = -0.0025,
    #     max_risk: float = -0.0075,
    #     span: int = 5,
    #     equal_weight: float = 0.05,
    #     fx_col: str = None,
    #     output_prefix: str = ''
    # ) -> pd.DataFrame:
    #     """
    #     Simulate position sizing strategies + save risk_level used for each entry.
    #     """
    #     # ── Validation & copy ───────────────────────────────────────────────────────
    #     required = [signal_col, price_col, stop_loss_col, chg_col]
    #     if fx_col:
    #         required.append(fx_col)
    #     missing = [c for c in required if c not in df.columns]
    #     if missing:
    #         raise KeyError(f"Missing columns: {missing}")

    #     df_out = df.copy()

    #     # ── Strategy definitions ────────────────────────────────────────────────────
    #     strategies = {
    #         'constant': {'risk_type': 'fixed',   'risk_value': (min_risk + max_risk)/2},
    #         'concave':  {'risk_type': 'dynamic', 'shape': -1},
    #         'convex':   {'risk_type': 'dynamic', 'shape': +1},
    #         'equal':    {'risk_type': 'fixed',   'risk_value': equal_weight},
    #     }

    #     # ── Output columns ──────────────────────────────────────────────────────────
    #     pfx = f"{output_prefix}_" if output_prefix else ''
    #     equity_cols = {k: f"{pfx}{k}"             for k in strategies}
    #     shares_cols = {k: f"{pfx}shs_{k}"         for k in strategies}
    #     risk_cols   = {k: f"{pfx}risk_{k}"        for k in strategies}   # ← new!

    #     # Initialize
    #     for col in equity_cols.values():
    #         df_out[col] = float(starting_capital)
    #     for col in shares_cols.values():
    #         df_out[col] = 0.0
    #     for col in risk_cols.values():
    #         df_out[col] = np.nan               # will be filled on entries

    #     current_shares = {k: 0.0 for k in strategies}
    #     current_risk   = {k: np.nan for k in strategies}  # remember last used risk

    #     # ── Main loop ───────────────────────────────────────────────────────────────
    #     for i in range(1, len(df_out)):
    #         prev_idx = df_out.index[i-1]
    #         curr_idx = df_out.index[i]

    #         sig_prev = df_out.at[prev_idx, signal_col]
    #         sig_curr = df_out.at[curr_idx, signal_col]

    #         chg = df_out.at[curr_idx, chg_col]
    #         fx = df_out.at[curr_idx, fx_col] if fx_col else 1.0

    #         # ── Update equity & record current shares ───────────────────────────────
    #         for strat, eq_col in equity_cols.items():
    #             prev_eq = df_out.at[prev_idx, eq_col]
    #             df_out.at[curr_idx, eq_col] = prev_eq + chg * current_shares[strat]
    #             df_out.at[curr_idx, shares_cols[strat]] = current_shares[strat]

    #             # Forward-fill last known risk level
    #             df_out.at[curr_idx, risk_cols[strat]] = current_risk[strat]

    #         # ── Entry logic ─────────────────────────────────────────────────────────
    #         if sig_prev == 0 and sig_curr != 0:
    #             px = df_out.at[curr_idx, price_col]
    #             sl = df_out.at[curr_idx, stop_loss_col]

    #             if abs(px - sl) < 1e-8:
    #                 continue

    #             direction = np.sign(sig_curr)

    #             # ── Equal weight ────────────────────────────────────────────────────
    #             eq = df_out.at[curr_idx, equity_cols['equal']]
    #             shs_equal = direction * (eq * equal_weight * fx // (px * lot)) * lot
    #             current_shares['equal'] = shs_equal
    #             current_risk['equal']   = equal_weight   # store the % used

    #             # ── Risk-based strategies ───────────────────────────────────────────
    #             for strat, conf in strategies.items():
    #                 if strat == 'equal':
    #                     continue

    #                 eq_now = df_out.at[curr_idx, equity_cols[strat]]

    #                 if conf['risk_type'] == 'fixed':
    #                     risk_level = conf['risk_value']
    #                 else:
    #                     appetite_series = self.risk_appetite(
    #                         # df = df_out[equity_cols[strat]].iloc[:i+1],
    #                         eqty= equity_cols[strat],
    #                         tolerance = tolerance,
    #                         min_risk = min_risk,
    #                         max_risk = max_risk,
    #                         span = span,
    #                         shape = conf['shape']
    #                     )
    #                     risk_level = appetite_series.iloc[-1]

    #                 # Calculate shares
    #                 shares_series = self._equity_risk_shares(
    #                     price = pd.Series(px, index=[curr_idx]),
    #                     stop_loss = pd.Series(sl, index=[curr_idx]),
    #                     eqty = pd.Series(eq_now, index=[curr_idx]),
    #                     risk = risk_level,
    #                     fx = fx,
    #                     lot = lot
    #                 )
    #                 shs = shares_series.iloc[0]

    #                 current_shares[strat] = direction * shs
    #                 current_risk[strat]   = risk_level   # remember this risk level

    #                 # Store the risk level used for this entry
    #                 df_out.at[curr_idx, risk_cols[strat]] = risk_level

    #     return df_out

    # def simulate_position_sizing_comparison(
    #     self,
    #     df: pd.DataFrame,
    #     signal_col: str,
    #     price_col: str = 'close',
    #     stop_loss_col: str = 'stop_loss',
    #     chg_col: str = 'chg1D_fx',          # daily pnl per share in account currency
    #     starting_capital: float = 1_000_000.0,
    #     lot: float = 100.0,
    #     tolerance: float = -0.10,
    #     min_risk: float = -0.0025,
    #     max_risk: float = -0.0075,
    #     span: int = 5,
    #     equal_weight: float = 0.05,
    #     fx_col: str = None,                 # if None → fx = 1 everywhere
    #     output_prefix: str = ''             # optional prefix for output columns
    # ) -> pd.DataFrame:
    #     """
    #     Simulate 4 position sizing strategies and return equity curves + shares history.

    #     Strategies:
    #         constant    → fixed risk % (avg of min_risk & max_risk)
    #         concave     → risk appetite (shape=-1)
    #         convex      → risk appetite (shape=+1)
    #         equal       → fixed % of equity (equal_weight)

    #     Output columns added (with optional prefix):
    #         {prefix}constant, {prefix}concave, {prefix}convex, {prefix}equal
    #         {prefix}shs_constant, {prefix}shs_concave, {prefix}shs_convex, {prefix}shs_equal

    #     Important:
    #         Assumes df[chg_col] already contains daily P&L per share in account currency (fx-adjusted)
    #         Uses self.risk_appetite(...) and self._equity_risk_shares(...) if they exist
    #     """
    #     # ── Validation ──────────────────────────────────────────────────────────────
    #     required = [signal_col, price_col, stop_loss_col, chg_col]
    #     if fx_col:
    #         required.append(fx_col)

    #     missing = [c for c in required if c not in df.columns]
    #     if missing:
    #         raise KeyError(f"Missing required columns: {missing}")

    #     df_out = df.copy()  # we return a new copy by default

    #     # ── Strategy definitions ────────────────────────────────────────────────────
    #     strategies = {
    #         'constant': {'risk_type': 'fixed',   'risk_value': (min_risk + max_risk)/2},
    #         'concave':  {'risk_type': 'dynamic', 'shape': -1},
    #         'convex':   {'risk_type': 'dynamic', 'shape': +1},
    #         'equal':    {'risk_type': 'fixed',   'risk_value': equal_weight},
    #     }

    #     # ── Prepare output columns ──────────────────────────────────────────────────
    #     pfx = f"{output_prefix}_" if output_prefix else ''
    #     equity_cols = {k: f"{pfx}{k}"      for k in strategies}
    #     shares_cols = {k: f"{pfx}shs_{k}"  for k in strategies}
    #     # risk_cols   = {k: f"{pfx}risk_{k}" for k in strategies}   # ← new!
    #     # avgddr_cols = {k: f"{pfx}avg_ddr_{k}"       for k in ['constant', 'concave', 'convex']}  # ← new!

    #     for eq_col in equity_cols.values():
    #         df_out[eq_col] = float(starting_capital)

    #     for sh_col in shares_cols.values():
    #         df_out[sh_col] = 0.0

    #     # for col in risk_cols.values():
    #     #         df_out[col] = np.nan               # will be filled on entries
    #     # for col in avgddr_cols.values():
    #     #         df_out[col] = np.nan

    #     # Current shares (updated only on entry)
    #     current_shares = {k: 0.0 for k in strategies}
    #     # current_risk   = {k: np.nan for k in strategies}  # remember last used risk
    #     # ── Main simulation loop ────────────────────────────────────────────────────

    #     for i in range(1, len(df_out)):
    #         prev_idx = df_out.index[i-1]
    #         curr_idx = df_out.index[i]

    #         sig_prev = df_out.at[prev_idx, signal_col]
    #         sig_curr = df_out.at[curr_idx, signal_col]

    #         # Daily P&L per share
    #         chg = df_out.at[curr_idx, chg_col]
    #         fx = df_out.at[curr_idx, fx_col] if fx_col else 1.0

    #         # Update all equity curves
    #         for strat, eq_col in equity_cols.items():
    #             prev_eq = df_out.at[prev_idx, eq_col]
    #             df_out.at[curr_idx, eq_col] = prev_eq + chg * current_shares[strat]
    #             df_out.at[curr_idx, shares_cols[strat]] = current_shares[strat]
                # Forward-fill last known risk level
                # df_out.at[curr_idx, risk_cols[strat]] = current_risk[strat]


            # ── Entry logic ─────────────────────────────────────────────────────────
            # if sig_prev == 0 and sig_curr != 0:
            #     px = df_out.at[curr_idx, price_col]
            #     sl = df_out.at[curr_idx, stop_loss_col]

            #     if abs(px - sl) < 1e-8:
            #         continue  # avoid division by zero / invalid R

            #     direction = np.sign(sig_curr)  # +1 long, -1 short

            #     # ── Equal weight ────────────────────────────────────────────────────
            #     eq = df_out.at[curr_idx, equity_cols['equal']]
            #     shs_equal = direction * (eq * equal_weight * fx // (px * lot)) * lot
            #     current_shares['equal'] = shs_equal
            #     current_risk['equal']   = equal_weight   # store the % used

            #     # ── Risk-based strategies ───────────────────────────────────────────
            #     for strat, conf in strategies.items():
            #         if strat == 'equal':
            #             continue

            #         eq_now = df_out.at[curr_idx, equity_cols[strat]]

            #         if conf['risk_type'] == 'fixed':
            #             risk_level = conf['risk_value']
            #         else:
            #             # Recompute risk appetite using current equity curve
            #             appetite_series = self.risk_appetite(
            #                 eqty=df_out[equity_cols[strat]].iloc[:i+1],
            #                 tolerance=tolerance,
            #                 min_risk=min_risk,
            #                 max_risk=max_risk,
            #                 span=span,
            #                 shape=conf['shape']
            #             )
            #             risk_level = appetite_series.iloc[-1]

            #         # Calculate shares (assuming your existing method returns signed shares)
            #         shares = self._equity_risk_shares(
            #             price=pd.Series(px, index=[curr_idx]),
            #             stop_loss=pd.Series(sl, index=[curr_idx]),
            #             eqty=pd.Series(eq_now, index=[curr_idx]),
            #             risk=risk_level,
            #             fx=fx,
            #             lot=lot
            #         ).iloc[0]

            #         current_shares[strat] = direction * shares
            #         current_risk[strat]   = risk_level
            #         df_out.at[curr_idx, risk_cols[strat]] = risk_level

        # return df_out


    def compare_position_sizing(
            self, df: pd.DataFrame, 
            signal: str, 
            price_col: str = 'Close', 
            stop_loss_col: str = 'stop_loss', 
            fx_col: str = 'fx', 
            daily_change_col: str = 'tt_chg1D_fx', 
            starting_capital: float = 1000000, 
            lot: float = 100, 
            tolerance: float = -0.1, 
            min_risk: float = -0.0025, 
            max_risk: float = -0.0075, 
            span: int = 5, 
            equal_weight: float = 0.05, 
            amortized_root: float = 2, 
            inplace: bool = False
            ) -> pd.DataFrame:
        """
        Compares position sizing algorithms (constant, concave, convex, equal weight, amortized weight) by calculating
        equity curves based on daily price changes and entry signals for both long and short positions.
        Shares are positive for long positions (signal > 0) and negative for short positions (signal < 0).
        The amortized weight strategy adjusts the equal_weight using a pyramid factor based on the number of active positions.

        Args:
            df (pd.DataFrame): DataFrame containing signal, price, stop-loss, daily change, and fx columns.
            signal (str): Name of the signal column (e.g., 'tt', 'bo_5') with values like 1 (long), -1 (short), 0 (no position).
            price_col (str): Name of the price column (e.g., 'Close'). Defaults to 'Close'.
            stop_loss_col (str): Name of the stop-loss column (e.g., 'stop_loss'). Defaults to 'stop_loss'.
            fx_col (str): Name of the currency conversion column (e.g., 'fx'). Defaults to 'fx'.
            daily_change_col (str): Name of the daily price change column (e.g., 'tt_chg1D_fx'). Defaults to 'tt_chg1D_fx'.
            starting_capital (float): Initial capital for equity curves. Defaults to 1000000.
            lot (float): Lot size for rounding shares (e.g., 100 for stocks). Defaults to 100.
            tolerance (float): Drawdown tolerance for risk appetite (e.g., -0.1 for 10% drawdown). Defaults to -0.1.
            min_risk (float): Minimum risk level for risk appetite (e.g., -0.0025). Defaults to -0.0025.
            max_risk (float): Maximum risk level for risk appetite (e.g., -0.0075). Defaults to -0.0075.
            span (int): Window for exponential moving average in risk appetite. Defaults to 5.
            equal_weight (float): Fixed weight for equal weight and amortized weight strategies (e.g., 0.05 for 5%). Defaults to 0.05.
            amortized_root (float): Root for pyramid function in amortized weight strategy (e.g., 2 for square root). Defaults to 2.
            inplace (bool): If True, modify input DataFrame; else return a new one. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with additional columns:
                - 'constant': Equity curve for constant risk strategy.
                - 'concave': Equity curve for concave risk appetite strategy.
                - 'convex': Equity curve for convex risk appetite strategy.
                - 'equal_weight': Equity curve for equal weight strategy.
                - 'amortized': Equity curve for amortized weight strategy.
                - 'shs_fxd': Shares for constant risk strategy.
                - 'shs_ccv': Shares for concave risk appetite strategy.
                - 'shs_cvx': Shares for convex risk appetite strategy.
                - 'shs_eql': Shares for equal weight strategy.
                - 'shs_amz': Shares for amortized weight strategy.

        Raises:
            KeyError: If required columns are missing.
            ValueError: If DataFrame is empty, parameters are invalid, or columns contain non-numeric data.
        """
        try:
            # Validate inputs
            if df.empty:
                raise ValueError("Input DataFrame is empty.")
            required_cols = [signal, price_col, stop_loss_col, fx_col, daily_change_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"Missing columns: {missing_cols}")
            if not all(np.issubdtype(df[col].dtype, np.number) for col in [price_col, stop_loss_col, fx_col, daily_change_col, signal]):
                raise ValueError(f"Columns {price_col}, {stop_loss_col}, {fx_col}, {daily_change_col}, {signal} must contain numeric data.")
            if starting_capital <= 0:
                raise ValueError("starting_capital must be positive.")
            if lot <= 0:
                raise ValueError("lot must be positive.")
            if tolerance >= 0:
                raise ValueError("tolerance must be negative (e.g., -0.1 for 10% drawdown).")
            if min_risk > 0 or max_risk > 0:
                raise ValueError("min_risk and max_risk must be non-positive.")
            if min_risk < max_risk:
                raise ValueError("min_risk must be greater than or equal to max_risk.")
            if span < 1:
                raise ValueError("span must be positive.")
            if equal_weight <= 0:
                raise ValueError("equal_weight must be positive.")
            if amortized_root <= 0:
                raise ValueError("amortized_root must be positive.")

            # Create working DataFrame
            if inplace:
                    result_df = df
            else:
                result_df = df.copy()   # deep copy by default


            shs_fxd = shs_ccv = shs_cvx = shs_eql = 0
            result_df.loc[result_df.index[0],'constant'] = result_df.loc[result_df.index[0],'concave'] = starting_capital
            result_df.loc[result_df.index[0],'convex'] = result_df.loc[result_df.index[0],'equal_weight'] = starting_capital

            # Initialize columns
            result_df['ccv'] = 0.0
            result_df['cvx'] = 0.0
            result_df['shs_eql'] = 0.0
            result_df['shs_fxd'] = 0.0
            result_df['shs_ccv'] = 0.0
            result_df['shs_cvx'] = 0.0

            # Cache column references for faster access
            equal_weight_col = result_df['equal_weight']
            constant_col = result_df['constant']
            concave_col = result_df['concave']
            convex_col = result_df['convex']
            # tt_5010_chg1D_fx_col = result_df[daily_change_col]
            # tt_5010_col = result_df[signal]
            # close_col = result_df[price_col]


            for i in range(1, len(result_df)):
                # Cache previous row values to avoid repeated lookups
                prev_equal_weight = result_df.at[i-1, 'equal_weight']
                prev_constant = result_df.at[i-1, 'constant']
                prev_concave = result_df.at[i-1, 'concave']
                prev_convex = result_df.at[i-1, 'convex']
                curr_chg = result_df.at[i, daily_change_col]
                
                # Update equity columns using .at (faster than .loc for scalars)
                result_df.at[i, 'equal_weight'] = prev_equal_weight + curr_chg * shs_eql
                result_df.at[i, 'constant'] = prev_constant + curr_chg * shs_fxd
                result_df.at[i, 'concave'] = prev_concave + curr_chg * shs_ccv
                result_df.at[i, 'convex'] = prev_convex + curr_chg * shs_cvx
                
                # Calculate risk appetite values using iloc slicing (more efficient)
                ccv = self.get_risk_appetite(eqty=concave_col.iloc[:i], tolerance=tolerance, 
                                    mn=min, mx=mx, span=5, shape=-1)
                cvx = self.get_risk_appetiterisk_appetite(eqty=convex_col.iloc[:i], tolerance=tolerance, 
                                    mn=mn, mx=mx, span=5, shape=1)
                
                # Store risk appetite values
                ccv_val = ccv.iloc[-1]
                cvx_val = cvx.iloc[-1]
                result_df.at[i, 'ccv'] = ccv_val
                result_df.at[i, 'cvx'] = cvx_val

                # Check condition with cached values
                if (result_df.at[i-1, 'tt_5010'] == 0) and (result_df.at[i, 'tt_5010'] != 0):
                    px = result_df.at[i, 'close']
                    sl = px * 0.9
                    fx = 1
                    
                    # Calculate equal weight shares
                    shs_eql = (result_df.at[i, 'equal_weight'] * equal_weight * fx // (px * lot)) * lot
                    
                    if px != sl:
                        # Calculate risk-based shares
                        shs_fxd = eqty_risk_shares(px, sl, eqty=result_df.at[i, 'constant'],
                                                    risk=avg, fx=fx, lot=100)
                        shs_ccv = eqty_risk_shares(px, sl, eqty=result_df.at[i, 'concave'],
                                                    risk=ccv_val, fx=fx, lot=100)
                        shs_cvx = eqty_risk_shares(px, sl, eqty=result_df.at[i, 'convex'],
                                                    risk=cvx_val, fx=fx, lot=100)
                
                # Store share values
                result_df.at[i, 'shs_eql'] = shs_eql
                result_df.at[i, 'shs_fxd'] = shs_fxd
                result_df.at[i, 'shs_ccv'] = shs_ccv
                result_df.at[i, 'shs_cvx'] = shs_cvx

            return None if inplace else result_df

        except Exception as e:
            raise ValueError(f"Error computing position sizing comparison: {str(e)}")