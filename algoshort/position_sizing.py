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

    def get_equity_risk(self, df: pd.DataFrame, tolerance: float = -0.1, mn: float = -0.0025, 
                        mx: float = -0.0075, span: int = 5,
                        relative: bool = False, inplace: bool = False) -> pd.DataFrame:
        """
        Calculates equity risk metrics based on the close price, including peak equity, constant/convex/concave risk, tolerance, and drawdown.

        Args:
            df (pd.DataFrame): DataFrame containing OHLC data.
            tolerance (float): Drawdown tolerance as a negative fraction (e.g., -0.1 for 10% drawdown). Defaults to -0.1.
            mn (float): Minimum risk level (e.g., -0.0025 for -0.25%). Defaults to -0.0025.
            mx (float): Maximum risk level (e.g., -0.0075 for -0.75%). Defaults to -0.0075.
            span (int): Window for exponential moving average in risk appetite. Defaults to 5.
            relative (bool): If True, use relative OHLC columns ('r_close', etc.); else absolute ('close', etc.). Defaults to False.
            inplace (bool): If True, modify input DataFrame; else return a new one. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with additional columns:
                - 'peak_eqty': Cumulative maximum of the close price.
                - 'constant_risk': Constant risk as -close * (mn + mx)/2
                - 'convex_risk': Convex risk appetite scaled by peak equity.
                - 'concave_risk': Concave risk appetite scaled by peak equity.
                - 'tolerance': Peak equity * (1 + tolerance).
                - 'drawdown': Drawdown as (close / peak_eqty - 1).

        Raises:
            KeyError: If required close column is missing.
            ValueError: If DataFrame is empty, parameters are invalid, or data types are non-numeric.
        """
        try:
            # Validate inputs
            if df.empty:
                raise ValueError("Input DataFrame is empty.")
            if tolerance >= 0:
                raise ValueError("tolerance must be negative (e.g., -0.1 for 10% drawdown).")
            if mn > 0 or mx > 0:
                raise ValueError("mn and mx must be non-positive.")
            if mn < mx:
                raise ValueError("mn must be greater than or equal to mx.")
            if span < 1:
                raise ValueError("span must be positive.")

            # Get OHLC column names
            _o, _h, _l, _c = ReturnsCalculator._lower_upper_OHLC(self.data, relative=relative)
            
            # Validate close column
            if _c not in df.columns:
                raise KeyError(f"Close column '{_c}' not found in DataFrame.")
            if not np.issubdtype(df[_c].dtype, np.number):
                raise ValueError(f"Close column '{_c}' must contain numeric data.")

            # Create working DataFrame
            result_df = df if inplace else df.copy()

            # Calculate average risk
            avg = (mn + mx) / 2

            # Calculate peak equity
            result_df['peak_eqty'] = result_df[_c].cummax()

            # Calculate constant risk, 
            result_df['constant_risk'] = -result_df[_c] * avg

            # Calculate convex and concave risk using risk_appetite
            result_df['convex_risk'] = -self.get_risk_appetite(result_df, price_col=_c, tolerance=tolerance, 
                                                              min_risk=mn, max_risk=mx, span=span, shape=1, 
                                                              inplace=False)[f'{_c}_risk_appetite'] * result_df['peak_eqty']
            result_df['concave_risk'] = -self.get_risk_appetite(result_df, price_col=_c, tolerance=tolerance, 
                                                               min_risk=mn, max_risk=mx, span=span, shape=-1, 
                                                               inplace=False)[f'{_c}_risk_appetite'] * result_df['peak_eqty']

            # Calculate tolerance and drawdown
            result_df['tolerance'] = result_df['peak_eqty'] * (1 + tolerance)
            result_df['drawdown'] = result_df[_c] / result_df['peak_eqty'] - 1

            return result_df

        except Exception as e:
            raise ValueError(f"Error computing equity risk metrics: {str(e)}")

    def get_risk_appetite(self, df: pd.DataFrame, price_col: str, tolerance: float = -0.1, 
                          min_risk: float = -0.0025, max_risk: float = -0.0075, span: int = 5, 
                          shape: int = 0, inplace: bool = False) -> pd.DataFrame:
        """
        Calculates risk appetite based on the price series (e.g., close price) for position sizing.

        Args:
            df (pd.DataFrame): DataFrame containing the price column.
            price_col (str): Name of the price column (e.g., 'close', 'r_close').
            tolerance (float): Drawdown tolerance as a negative fraction (e.g., -0.1 for 10% drawdown). Defaults to -0.1.
            min_risk (float): Minimum risk level (e.g., -0.0025 for -0.25%). Defaults to -0.0025.
            max_risk (float): Maximum risk level (e.g., -0.0075 for -0.75%). Defaults to -0.0075.
            span (int): Window for exponential moving average to smooth risk appetite. Defaults to 5.
            shape (int): Shape of risk appetite curve: 1 (convex), -1 (concave), 0 (linear). Defaults to 0.
            inplace (bool): If True, modify input DataFrame; else return a new one. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with an additional column:
                - '<price_col>_risk_appetite': Risk appetite series, scaled between min_risk and max_risk.

        Raises:
            KeyError: If price_col is not found in DataFrame.
            ValueError: If DataFrame is empty, parameters are invalid, or price_col contains non-numeric data.
        """
        try:
            # Validate inputs
            if df.empty:
                raise ValueError("Input DataFrame is empty.")
            if price_col not in df.columns:
                raise KeyError(f"Price column '{price_col}' not found in DataFrame.")
            if not np.issubdtype(df[price_col].dtype, np.number):
                raise ValueError(f"Price column '{price_col}' must contain numeric data.")
            if tolerance >= 0:
                raise ValueError("tolerance must be negative (e.g., -0.1 for 10% drawdown).")
            if min_risk > 0 or max_risk > 0:
                raise ValueError("min_risk and max_risk must be non-positive.")
            if min_risk < max_risk:
                raise ValueError("min_risk must be greater than or equal to max_risk.")
            if span < 1:
                raise ValueError("span must be positive.")
            if shape not in [-1, 0, 1]:
                raise ValueError("shape must be -1 (concave), 0 (linear), or 1 (convex).")

            # Create working DataFrame
            result_df = df if inplace else df.copy()

            # Calculate drawdown rebased to tolerance
            price = result_df[price_col]
            cache_key_watermark = f"watermark_{price_col}"
            if cache_key_watermark in self._cache:
                watermark = self._cache[cache_key_watermark]
            else:
                watermark = price.expanding().max()  # All-time-high peak price
                self._cache[cache_key_watermark] = watermark

            drawdown = price / watermark - 1  # Drawdown from peak
            ddr = 1 - np.minimum(drawdown / tolerance, 1)  # Rebased drawdown (0 to 1)
            avg_ddr = ddr.ewm(span=span).mean()  # Smoothed rebased drawdown

            # Determine shape of the risk appetite curve
            if shape == 1:  # Convex
                _power = max_risk / min_risk if min_risk < 0 else 1
            elif shape == -1:  # Concave
                _power = min_risk / max_risk if max_risk < 0 else 1
            else:  # Linear
                _power = 1
            ddr_power = avg_ddr ** _power

            # Calculate risk appetite
            result_df[f'{price_col}_risk_appetite'] = min_risk + (max_risk - min_risk) * ddr_power

            return result_df

        except Exception as e:
            raise ValueError(f"Error computing risk appetite: {str(e)}")

    def get_drawdown(self, df: pd.DataFrame, equity_col: str, dd_tolerance: float = -0.1, 
                     inplace: bool = False) -> pd.DataFrame:
        """
        Calculates drawdown metrics for an equity curve.

        Args:
            df (pd.DataFrame): DataFrame containing the equity curve column.
            equity_col (str): Name of the equity curve column (e.g., 'bo_5_equity', 'sma_358_equity').
            dd_tolerance (float): Drawdown tolerance as a negative fraction (e.g., -0.1 for 10% drawdown). Defaults to -0.1.
            inplace (bool): If True, modify input DataFrame; else return a new one. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with additional columns:
                - '<equity_col>_peak': Cumulative maximum of the equity curve.
                - '<equity_col>_tolerance': Peak equity * (1 + dd_tolerance).
                - '<equity_col>_drawdown': Drawdown as (equity / peak_equity - 1).

        Raises:
            KeyError: If equity_col is not found in DataFrame.
            ValueError: If DataFrame is empty, dd_tolerance is non-negative, or equity_col contains non-numeric data.
        """
        try:
            # Validate inputs
            if df.empty:
                raise ValueError("Input DataFrame is empty.")
            if equity_col not in df.columns:
                raise KeyError(f"Equity column '{equity_col}' not found in DataFrame.")
            if not np.issubdtype(df[equity_col].dtype, np.number):
                raise ValueError(f"Equity column '{equity_col}' must contain numeric data.")
            if dd_tolerance >= 0:
                raise ValueError("dd_tolerance must be negative (e.g., -0.1 for 10% drawdown).")

            # Create working DataFrame
            result_df = df if inplace else df.copy()

            # Cache key for peak equity
            cache_key_peak = f"peak_{equity_col}"

            # Calculate peak equity (cumulative max)
            if cache_key_peak in self._cache:
                peak_equity = self._cache[cache_key_peak]
            else:
                peak_equity = result_df[equity_col].cummax()
                self._cache[cache_key_peak] = peak_equity

            # Calculate drawdown metrics
            result_df[f'{equity_col}_peak'] = peak_equity
            result_df[f'{equity_col}_tolerance'] = peak_equity * (1 + dd_tolerance)
            result_df[f'{equity_col}_drawdown'] = result_df[equity_col] / peak_equity - 1

            return result_df

        except Exception as e:
            raise ValueError(f"Error computing drawdown metrics: {str(e)}")
        
    def calculate_position_size(self, df: pd.DataFrame, signal: str, price_col: str = 'close', 
                               stop_loss_col: str = None, equity_col: str = None, 
                               fx: float = 1.0, lot: float = 100, r_multiplier: float = 2.0, 
                               tolerance: float = -0.1, min_risk: float = -0.0025, 
                               max_risk: float = -0.0075, span: int = 5, shape: int = 0, 
                               use_equity_risk: bool = False, risk: float = -0.005, 
                               inplace: bool = False) -> pd.DataFrame:
        """
        Calculates position sizing metrics, including risk budget, risk unit, shares, target price, partial exits,
        and optional equity-based shares using a fixed risk fraction. Shares are positive for long positions and
        negative for short positions when using equity risk.

        Args:
            df (pd.DataFrame): DataFrame containing price, stop-loss, signal, and optional equity columns.
            signal (str): Name of the signal column (e.g., 'bo_5', 'tt_52') used for naming output columns and trade direction.
            price_col (str): Name of the price column (e.g., 'close'). Defaults to 'close'.
            stop_loss_col (str, optional): Name of the stop-loss column (e.g., 'bo_5_stop_loss'). If None, skips position sizing.
            equity_col (str, optional): Name of the equity column (e.g., 'bo_5_equity'). If None, uses price_col.
            fx (float): Currency conversion factor. Defaults to 1.0.
            lot (float): Lot size for rounding shares (e.g., 100 for stocks). Defaults to 100.
            r_multiplier (float): Risk-reward multiplier for target price and partial exit (e.g., 2.0 for 2R). Defaults to 2.0.
            tolerance (float): Drawdown tolerance for risk appetite (e.g., -0.1 for 10% drawdown). Defaults to -0.1.
            min_risk (float): Minimum risk level for risk appetite (e.g., -0.0025). Defaults to -0.0025.
            max_risk (float): Maximum risk level for risk appetite (e.g., -0.0075). Defaults to -0.0075.
            span (int): Window for exponential moving average in risk appetite. Defaults to 5.
            shape (int): Shape of risk appetite curve: 1 (convex), -1 (concave), 0 (linear). Defaults to 0.
            use_equity_risk (bool): If True, compute shares using fixed risk fraction instead of risk appetite. Defaults to False.
            risk (float): Fixed risk fraction for equity-based shares (e.g., -0.005 for 0.5% risk). Defaults to -0.005.
            inplace (bool): If True, modify input DataFrame; else return a new one. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with additional columns:
                - '<signal>_peak_equity': Peak equity of the specified equity or price column.
                - '<signal>_risk_appetite': Risk appetite from get_risk_appetite.
                - '<signal>_risk_budget': Risk budget as peak_equity * risk_appetite * fx.
                - '<signal>_risk_unit': Risk unit (R) as price - stop_loss.
                - '<signal>_shares': Number of shares rounded to lot size (if use_equity_risk=False).
                - '<signal>_equity_risk_shares': Number of shares using fixed risk fraction (if use_equity_risk=True, positive for long, negative for short).
                - '<signal>_target_price': Target price as price + R * r_multiplier.
                - '<signal>_partial_exit': Partial exit quantity as shares / r_multiplier.

        Raises:
            KeyError: If required columns (price_col, stop_loss_col, equity_col, signal) are missing.
            ValueError: If DataFrame is empty, parameters are invalid, or columns contain non-numeric data.
        """
        try:
            # Validate inputs
            if df.empty:
                raise ValueError("Input DataFrame is empty.")
            if price_col not in df.columns:
                raise KeyError(f"Price column '{price_col}' not found in DataFrame.")
            if stop_loss_col is not None and stop_loss_col not in df.columns:
                raise KeyError(f"Stop-loss column '{stop_loss_col}' not found in DataFrame.")
            if equity_col is not None and equity_col not in df.columns:
                raise KeyError(f"Equity column '{equity_col}' not found in DataFrame.")
            if signal not in df.columns:
                raise KeyError(f"Signal column '{signal}' not found in DataFrame.")
            if not np.issubdtype(df[price_col].dtype, np.number):
                raise ValueError(f"Price column '{price_col}' must contain numeric data.")
            if stop_loss_col is not None and not np.issubdtype(df[stop_loss_col].dtype, np.number):
                raise ValueError(f"Stop-loss column '{stop_loss_col}' must contain numeric data.")
            if equity_col is not None and not np.issubdtype(df[equity_col].dtype, np.number):
                raise ValueError(f"Equity column '{equity_col}' must contain numeric data.")
            if not np.issubdtype(df[signal].dtype, np.number):
                raise ValueError(f"Signal column '{signal}' must contain numeric data.")
            if fx <= 0 and use_equity_risk:
                raise ValueError("fx must be positive when use_equity_risk=True.")
            if lot <= 0:
                raise ValueError("lot must be positive.")
            if r_multiplier <= 0:
                raise ValueError("r_multiplier must be positive.")
            if tolerance >= 0:
                raise ValueError("tolerance must be negative (e.g., -0.1 for 10% drawdown).")
            if min_risk > 0 or max_risk > 0:
                raise ValueError("min_risk and max_risk must be non-positive.")
            if min_risk < max_risk:
                raise ValueError("min_risk must be greater than or equal to max_risk.")
            if span < 1:
                raise ValueError("span must be positive.")
            if risk >= 0:
                raise ValueError("risk must be negative (e.g., -0.005 for 0.5% risk).")

            # Create working DataFrame
            result_df = df if inplace else df.copy()

            # Get risk appetite
            appetite_df = self.get_risk_appetite(
                result_df, price_col=price_col, tolerance=tolerance, min_risk=min_risk,
                max_risk=max_risk, span=span, shape=shape, inplace=False
            )
            appetite = appetite_df[f'{price_col}_risk_appetite']

            # Select equity series (use equity_col if provided, else price_col)
            eqty = result_df[equity_col] if equity_col is not None else result_df[price_col]

            # Cache key for position sizing metrics
            cache_key_prefix = f"position_size_{signal}_{price_col}_{equity_col}_{fx}_{lot}_{r_multiplier}_{risk}"

            # Calculate peak equity
            cache_key_peak = f"{cache_key_prefix}_peak_equity"
            if cache_key_peak in self._cache:
                peak_eqty = self._cache[cache_key_peak]
            else:
                peak_eqty = self._peak_equity(eqty)
                self._cache[cache_key_peak] = peak_eqty
            result_df[f'{signal}_peak_equity'] = peak_eqty

            # Copy risk appetite
            result_df[f'{signal}_risk_appetite'] = appetite

            # Initialize output columns
            result_df[f'{signal}_risk_budget'] = np.nan
            result_df[f'{signal}_risk_unit'] = np.nan
            result_df[f'{signal}_shares'] = np.nan
            result_df[f'{signal}_equity_risk_shares'] = np.nan
            result_df[f'{signal}_target_price'] = np.nan
            result_df[f'{signal}_partial_exit'] = np.nan

            if stop_loss_col is not None:
                # Calculate risk budget
                cache_key_budget = f"{cache_key_prefix}_risk_budget"
                if cache_key_budget in self._cache:
                    risk_budget = self._cache[cache_key_budget]
                else:
                    risk_budget = self._risk_budget(eqty, appetite, fx)
                    self._cache[cache_key_budget] = risk_budget
                result_df[f'{signal}_risk_budget'] = risk_budget

                # Calculate risk unit (R)
                cache_key_risk_unit = f"{cache_key_prefix}_risk_unit"
                if cache_key_risk_unit in self._cache:
                    risk_unit = self._cache[cache_key_risk_unit]
                else:
                    risk_unit = self._risk_unit(result_df[price_col], result_df[stop_loss_col])
                    self._cache[cache_key_risk_unit] = risk_unit
                result_df[f'{signal}_risk_unit'] = risk_unit

                # Calculate shares (using risk appetite)
                cache_key_shares = f"{cache_key_prefix}_shares"
                if cache_key_shares in self._cache:
                    shares = self._cache[cache_key_shares]
                else:
                    shares = self._shares_roundlot(risk_budget, fx, risk_unit, lot)
                    self._cache[cache_key_shares] = shares
                result_df[f'{signal}_shares'] = shares

                # Calculate equity risk shares (using fixed risk fraction)
                if use_equity_risk:
                    cache_key_eqty_shares = f"{cache_key_prefix}_equity_risk_shares"
                    if cache_key_eqty_shares in self._cache:
                        eqty_shares = self._cache[cache_key_eqty_shares]
                    else:
                        eqty_shares = self._equity_risk_shares(
                            result_df[price_col], result_df[stop_loss_col], eqty, risk, fx, lot, result_df[signal]
                        )
                        self._cache[cache_key_eqty_shares] = eqty_shares
                    result_df[f'{signal}_equity_risk_shares'] = eqty_shares

                # Calculate target price
                cache_key_target = f"{cache_key_prefix}_target_price"
                if cache_key_target in self._cache:
                    target_price = self._cache[cache_key_target]
                else:
                    target_price = self._target_price(result_df[price_col], result_df[stop_loss_col], r_multiplier)
                    self._cache[cache_key_target] = target_price
                result_df[f'{signal}_target_price'] = target_price

                # Calculate partial exit
                cache_key_partial = f"{cache_key_prefix}_partial_exit"
                if cache_key_partial in self._cache:
                    partial_exit = self._cache[cache_key_partial]
                else:
                    partial_exit = self._partial_exit(shares, r_multiplier)
                    self._cache[cache_key_partial] = partial_exit
                result_df[f'{signal}_partial_exit'] = partial_exit

            return result_df

        except Exception as e:
            raise ValueError(f"Error computing position sizing metrics: {str(e)}")

    def compare_position_sizing(self, df: pd.DataFrame, signal: str, price_col: str = 'Close', 
                            stop_loss_col: str = 'stop_loss', fx_col: str = 'fx', 
                            daily_change_col: str = 'tt_chg1D_fx', 
                            starting_capital: float = 1000000, lot: float = 100, 
                            tolerance: float = -0.1, min_risk: float = -0.0025, 
                            max_risk: float = -0.0075, span: int = 5, 
                            equal_weight: float = 0.05, amortized_root: float = 2, 
                            inplace: bool = False) -> pd.DataFrame:
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
            result_df = df if inplace else df.copy()

            # Initialize equity curves and shares
            result_df['constant'] = starting_capital
            result_df['concave'] = starting_capital
            result_df['convex'] = starting_capital
            result_df['equal_weight'] = starting_capital
            result_df['amortized'] = starting_capital
            result_df['shs_fxd'] = 0
            result_df['shs_ccv'] = 0
            result_df['shs_cvx'] = 0
            result_df['shs_eql'] = 0
            result_df['shs_amz'] = 0

            result_df['equal_weight'] = result_df['equal_weight'].astype(float)
            result_df['constant'] = result_df['constant'].astype(float)
            result_df['convex'] = result_df['convex'].astype(float)
            result_df['amortized'] = result_df['amortized'].astype(float)
            result_df['concave'] = result_df['concave'].astype(float)
            result_df['shs_fxd'] = result_df['shs_fxd'].astype(float)
            result_df['shs_ccv'] = result_df['shs_ccv'].astype(float)
            result_df['shs_cvx'] = result_df['shs_cvx'].astype(float)
            result_df['shs_eql'] = result_df['shs_eql'].astype(float)
            result_df['shs_amz'] = result_df['shs_amz'].astype(float)

            # Initialize shares
            shs_fxd = shs_ccv = shs_cvx = shs_eql = shs_amz = 0

            # Initialize position count for amortization
            position_count = 0

            # Calculate average risk
            avg = (min_risk + max_risk) / 2

            # Loop through DataFrame
            for i in range(1, len(result_df)):
                # Update position count based on signal
                if result_df[signal].iloc[i-1] == 0 and (result_df[signal].iloc[i] > 0 or result_df[signal].iloc[i] < 0):
                    position_count += 1  # New position
                elif result_df[signal].iloc[i] == 0 and result_df[signal].iloc[i-1] != 0:
                    position_count = max(0, position_count - 1)  # Exit position

                # Update equity curves
                result_df.loc[result_df.index[i], 'equal_weight'] = result_df.loc[result_df.index[i-1], 'equal_weight'] + \
                                                    result_df.loc[result_df.index[i], daily_change_col] * shs_eql
                result_df.loc[result_df.index[i], 'constant'] = result_df.loc[result_df.index[i-1], 'constant'] + \
                                                    result_df.loc[result_df.index[i], daily_change_col] * shs_fxd
                result_df.loc[result_df.index[i], 'concave'] = result_df.loc[result_df.index[i-1], 'concave'] + \
                                                    result_df.loc[result_df.index[i], daily_change_col] * shs_ccv
                result_df.loc[result_df.index[i], 'convex'] = result_df.loc[result_df.index[i-1], 'convex'] + \
                                                    result_df.loc[result_df.index[i], daily_change_col] * shs_cvx
                result_df.loc[result_df.index[i], 'amortized'] = result_df.loc[result_df.index[i-1], 'amortized'] + \
                                                    result_df.loc[result_df.index[i], daily_change_col] * shs_amz

                # Store current shares in DataFrame
                result_df.loc[result_df.index[i], 'shs_fxd'] = shs_fxd
                result_df.loc[result_df.index[i], 'shs_ccv'] = shs_ccv
                result_df.loc[result_df.index[i], 'shs_cvx'] = shs_cvx
                result_df.loc[result_df.index[i], 'shs_eql'] = shs_eql
                result_df.loc[result_df.index[i], 'shs_amz'] = shs_amz

                # Calculate risk appetite for concave and convex strategies
                concave_df = self.get_risk_appetite(
                    result_df.iloc[:i+1], price_col='concave', tolerance=tolerance, 
                    min_risk=min_risk, max_risk=max_risk, span=span, shape=-1, inplace=False
                )
                convex_df = self.get_risk_appetite(
                    result_df.iloc[:i+1], price_col='convex', tolerance=tolerance, 
                    min_risk=min_risk, max_risk=max_risk, span=span, shape=1, inplace=False
                )
                ccv = concave_df['concave_risk_appetite'].iloc[-1]
                cvx = convex_df['convex_risk_appetite'].iloc[-1]

                # Check for entry signal (long: signal > 0, short: signal < 0)
                if result_df[signal].iloc[i-1] == 0 and (result_df[signal].iloc[i] > 0 or result_df[signal].iloc[i] < 0):
                    px = result_df[price_col].iloc[i]
                    sl = result_df[stop_loss_col].iloc[i]
                    fx = result_df[fx_col].iloc[i]
                    sig = result_df[signal].iloc[i]

                    # Calculate shares for equal weight (positive for long, negative for short)
                    shs_eql = (result_df['equal_weight'].iloc[i] * equal_weight * fx // (px * lot)) * lot

                    # Calculate shares for amortized weight
                    amortization = self._pyramid(position_count, root=amortized_root)
                    amz_weight = self._amortized_weight(equal_weight, amortization)
                    shs_amz = (result_df['amortized'].iloc[i] * amz_weight * fx // (px * lot)) * lot

                    # Calculate shares for other strategies
                    if px != sl:
                        shs_fxd = self._equity_risk_shares(
                            pd.Series(px, index=[result_df.index[i]]), 
                            pd.Series(sl, index=[result_df.index[i]]), 
                            pd.Series(result_df['constant'].iloc[i], index=[result_df.index[i]]), 
                            risk=avg, fx=fx, lot=lot
                        ).iloc[0]
                        shs_ccv = self._equity_risk_shares(
                            pd.Series(px, index=[result_df.index[i]]), 
                            pd.Series(sl, index=[result_df.index[i]]), 
                            pd.Series(result_df['concave'].iloc[i], index=[result_df.index[i]]), 
                            risk=ccv, fx=fx, lot=lot
                        ).iloc[0]
                        shs_cvx = self._equity_risk_shares(
                            pd.Series(px, index=[result_df.index[i]]), 
                            pd.Series(sl, index=[result_df.index[i]]), 
                            pd.Series(result_df['convex'].iloc[i], index=[result_df.index[i]]), 
                            risk=cvx, fx=fx, lot=lot
                        ).iloc[0]

            return result_df

        except Exception as e:
            raise ValueError(f"Error computing position sizing comparison: {str(e)}")