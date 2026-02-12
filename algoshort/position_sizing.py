import pandas as pd
import numpy as np
from typing import List
from joblib import Parallel, delayed
import logging

logger = logging.getLogger(__name__)


class PositionSizing:
    """
    Portfolio calculator for managing multiple trading strategies with risk-based
    share allocations.
    """
    
    def __init__(
        self,
        tolerance: float,
        mn: float,
        mx: float,
        equal_weight: float,
        avg: float,
        lot: int,
        initial_capital: float = 100000
    ) -> None:
        """
        Initialize the portfolio calculator.

        Parameters:
        -----------
        tolerance : float
            Risk tolerance parameter (must be negative for drawdown)
        mn : float
            Minimum risk parameter (must be positive)
        mx : float
            Maximum risk parameter (must be positive and >= mn)
        equal_weight : float
            Equal weight allocation factor (must be in (0, 1])
        avg : float
            Average risk value for fixed strategy
        lot : int
            Lot size for share calculations (must be positive)
        initial_capital : float, default=100000
            Initial capital for each strategy (must be positive)

        Raises:
        -------
        ValueError
            If any parameter fails validation
        """
        # Validate tolerance (should be negative for drawdown)
        if tolerance >= 0:
            raise ValueError(f"tolerance must be negative, got {tolerance}")

        # Validate risk range
        if mn <= 0 or mx <= 0:
            raise ValueError(f"mn and mx must be positive: mn={mn}, mx={mx}")
        if mn > mx:
            raise ValueError(f"mn must be <= mx: mn={mn}, mx={mx}")

        # Validate equal_weight
        if not 0 < equal_weight <= 1:
            raise ValueError(f"equal_weight must be in (0, 1], got {equal_weight}")

        # Validate lot size
        if lot <= 0:
            raise ValueError(f"lot must be positive, got {lot}")

        # Validate capital
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")

        self.tolerance = tolerance
        self.mn = mn
        self.mx = mx
        self.equal_weight = equal_weight
        self.avg = avg
        self.lot = lot
        self.initial_capital = initial_capital

        logger.debug(f"PositionSizing initialized: tolerance={tolerance}, "
                    f"risk_range=({mn}, {mx}), lot={lot}")

    def eqty_risk_shares(
        self,
        px: float,
        sl: float,
        eqty: float,
        risk: float,
        fx: float,
        lot: int
    ) -> int:
        """
        Calculate position size based on risk parameters.

        Args:
            px: Current entry price (must be positive)
            sl: Stop loss price
            eqty: Current account equity
            risk: Risk percentage (e.g., 0.02 for 2%)
            fx: Foreign exchange rate multiplier
            lot: Minimum lot size (must be positive)

        Returns:
            Number of shares rounded to lot size increments
        """
        # Validate inputs
        if px <= 0:
            logger.warning(f"Invalid price: {px}, returning 0 shares")
            return 0
        if lot <= 0:
            logger.warning(f"Invalid lot size: {lot}, returning 0 shares")
            return 0
        if eqty < 0:
            logger.warning(f"Negative equity: {eqty}, returning 0 shares")
            return 0

        # Use absolute risk per share to handle both long and short positions
        risk_per_share = abs(sl - px)

        # Guard against stop loss too close to price (near-zero risk)
        if risk_per_share < 1e-10:
            logger.warning(f"Stop loss ({sl}) too close to price ({px})")
            return 0

        # Calculate budget
        budget = eqty * risk * (fx if fx > 0 else 1)

        # Calculate shares rounded to lot size
        num_lots = int(budget / (risk_per_share * lot))
        shares = num_lots * lot

        return max(0, shares)
    
    def risk_appetite(
        self,
        eqty: pd.Series,
        tolerance: float,
        mn: float,
        mx: float,
        span: int,
        shape: int
    ) -> pd.Series:
        """
        Calculate risk appetite based on equity curve and drawdown.

        Parameters:
        -----------
        eqty : pd.Series or array-like
            Equity curve series
        tolerance : float
            Tolerance for drawdown (<0, must be non-zero)
        mn : float
            Minimum risk (must be positive)
        mx : float
            Maximum risk (must be positive)
        span : int
            Exponential moving average span to smooth the risk_appetite
        shape : int
            Convex (>45 deg diagonal) = 1, concave (<diagonal) = -1,
            else: simple risk_appetite

        Returns:
        --------
        pd.Series
            Risk appetite values
        """
        # Validate inputs to prevent division by zero
        if tolerance == 0:
            logger.warning("tolerance is 0, using default -0.1")
            tolerance = -0.1
        if mn <= 0:
            logger.warning(f"mn must be positive, got {mn}, using 0.01")
            mn = 0.01
        if mx <= 0:
            logger.warning(f"mx must be positive, got {mx}, using 0.01")
            mx = 0.01

        # Drawdown rebased
        eqty = pd.Series(eqty)
        watermark = eqty.expanding().max()  # all-time-high peak equity

        # Guard against zero watermark (would cause division by zero)
        watermark = watermark.replace(0, np.nan).ffill().fillna(1)

        drawdown = eqty / watermark - 1  # drawdown from peak
        ddr = 1 - np.minimum(drawdown / tolerance, 1)  # drawdown rebased to tolerance from 0 to 1
        avg_ddr = ddr.ewm(span=span).mean()  # span rebased drawdown

        # Shape of the curve - with safe division
        if shape == 1:
            _power = mx / mn  # convex
        elif shape == -1:
            _power = mn / mx  # concave
        else:
            _power = 1  # raw, straight line
        ddr_power = avg_ddr ** _power  # ddr

        # mn + adjusted delta
        risk_appetite = mn + (mx - mn) * ddr_power

        return risk_appetite
    
    def _get_column_names(self, prefix):
        """
        Generate clear, consistent column names.
        Example output for prefix='rbo_50__rbo_50':
            equity:  rbo_50__rbo_50_equity_equal
                    rbo_50__rbo_50_equity_constant
                    rbo_50__rbo_50_equity_concave
                    rbo_50__rbo_50_equity_convex
            shares:  rbo_50__rbo_50_shares_equal
                    ...
        """
        base = prefix
        return {
            'strategies': [
                f'{base}_equity_equal',
                f'{base}_equity_constant',
                f'{base}_equity_concave',
                f'{base}_equity_convex',
            ],
            'share_cols': [
                f'{base}_shares_equal',
                f'{base}_shares_constant',
                f'{base}_shares_concave',
                f'{base}_shares_convex',
            ],
            'risk_configs': [
                {'col': f'{base}_equity_concave', 'shape': -1, 'store': f'{base}_risk_concave'},
                {'col': f'{base}_equity_convex',  'shape':  1, 'store': f'{base}_risk_convex'},
            ],
            'ccv': f'{base}_risk_concave',
            'cvx': f'{base}_risk_convex'
        }

    def _initialize_columns(self, df, cols):
        """
        Initialize required columns in the DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to initialize
        cols : dict
            Dictionary of column names from _get_column_names
        """
        # Initialize equity strategy columns with initial capital
        for strategy in cols['strategies']:
            if strategy not in df.columns:
                df[strategy] = 0.0
                df.at[0, strategy] = self.initial_capital
        
        # Initialize output columns
        for col in [cols['ccv'], cols['cvx']] + cols['share_cols']:
            df[col] = 0.0
    
    def _update_equity_columns(self, df, i, shares_dict, curr_chg, cols):
        """
        Update all equity columns for the current iteration.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to update
        i : int
            Current row index
        shares_dict : dict
            Dictionary mapping strategy names to share values
        curr_chg : float
            Current period change value
        cols : dict
            Dictionary of column names from _get_column_names
        """
        for strategy, share_col in zip(cols['strategies'], cols['share_cols']):
            prev_value = df.at[i-1, strategy]
            shares = shares_dict[share_col]
            df.at[i, strategy] = prev_value + curr_chg * shares
    
    def _calculate_risk_appetites(self, df, i, cols):
        """
        Calculate and store risk appetite values.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing equity data
        i : int
            Current row index
        cols : dict
            Dictionary of column names from _get_column_names
            
        Returns:
        --------
        dict
            Dictionary with 'ccv' and 'cvx' risk values
        """
        risk_values = {}
        
        for config in cols['risk_configs']:
            risk_series = self.risk_appetite(
                eqty=df[config['col']].iloc[:i],
                tolerance=self.tolerance,
                mn=self.mn,
                mx=self.mx,
                span=5,
                shape=config['shape']
            )
            risk_val = risk_series.iloc[-1]
            df.at[i, config['store']] = risk_val
            risk_values[config['store']] = risk_val
        
        return risk_values
    
    def _recalculate_shares(self, df, i, risk_values, sl, signal, close, cols):
        """
        Recalculate shares when entry condition is met.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing trading data
        i : int
            Current row index
        risk_values : dict
            Dictionary with 'ccv' and 'cvx' risk values
        sl : str
            Column name for stop loss
        signal : str
            Column name for entry signal
        close : str
            Column name for close price
        cols : dict
            Dictionary of column names from _get_column_names
            
        Returns:
        --------
        dict or None
            Dictionary with updated share values, or None if no recalculation needed
        """
        if df.at[i, signal] == df.at[i-1, signal]:
            return None

        px = df.at[i, close]
        sl_price = df.at[i, sl]

        # Use tolerance comparison instead of exact float equality
        if abs(px - sl_price) < 1e-10:
            return None
        
        fx = 1
        
        # Calculate equal weight shares
        shs_eql = (df.at[i, cols['strategies'][0]] * self.equal_weight * fx // 
                   (px * self.lot)) * self.lot
        
        # Calculate risk-based shares
        share_configs = [
            {'strategy': cols['strategies'][1], 'risk': self.avg,     'key': cols['share_cols'][1]},
            {'strategy': cols['strategies'][2], 'risk': risk_values[cols['ccv']], 'key': cols['share_cols'][2]},
            {'strategy': cols['strategies'][3], 'risk': risk_values[cols['cvx']], 'key': cols['share_cols'][3]}
        ]
        
        shares = {cols['share_cols'][0]: shs_eql}
        
        for config in share_configs:
            shares[config['key']] = self.eqty_risk_shares(
                px, sl_price,
                eqty=df.at[i, config['strategy']],
                risk=config['risk'],
                fx=fx,
                lot=self.lot
            )
        
        return shares
    
    def _store_shares(self, df, i, shares_dict, cols):
        """
        Store current share values in the DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to update
        i : int
            Current row index
        shares_dict : dict
            Dictionary of current share values
        cols : dict
            Dictionary of column names from _get_column_names
        """
        for share_col in cols['share_cols']:
            df.at[i, share_col] = shares_dict[share_col]
    
    def calculate_shares(self, df, signal, shs_eql=0, shs_fxd=0, shs_ccv=0, shs_cvx=0, 
                        daily_chg='chg1D_fx', sl='stop_loss', close='close'):
        """
        Calculate and update portfolio equity values and share allocations.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing trading data
        signal : str
            Column name for entry signal (also used as prefix for output columns)
        shs_eql : float, default=0
            Initial equal weight shares
        shs_fxd : float, default=0
            Initial fixed risk shares
        shs_ccv : float, default=0
            Initial concave risk shares
        shs_cvx : float, default=0
            Initial convex risk shares
        daily_chg : str, default='chg1D_fx'
            Column name for daily change
        sl : str, default='stop_loss'
            Column name for stop loss
        close : str, default='close'
            Column name for close price
            
        Returns:
        --------
        pd.DataFrame
            Updated DataFrame with calculated values
        """
        # Generate column names with signal prefix
        cols = self._get_column_names(signal)
        
        # Initialize columns
        self._initialize_columns(df, cols)
        
        # Track current shares with prefixed keys
        shares_dict = {
            cols['share_cols'][0]: shs_eql,
            cols['share_cols'][1]: shs_fxd,
            cols['share_cols'][2]: shs_ccv,
            cols['share_cols'][3]: shs_cvx
        }

        for i in range(1, len(df)):
            # Get current change value
            curr_chg = df.at[i, daily_chg]
            
            # Update equity columns
            self._update_equity_columns(df, i, shares_dict, curr_chg, cols)
            
            # Calculate and store risk appetite values
            risk_values = self._calculate_risk_appetites(df, i, cols)
            
            # Recalculate shares if entry condition is met
            new_shares = self._recalculate_shares(df, i, risk_values, sl, signal, close, cols)
            if new_shares:
                shares_dict.update(new_shares)
            
            # Store current share values
            self._store_shares(df, i, shares_dict, cols)
        
        return df
    
    def calculate_shares_for_signal(
            self,
            df: pd.DataFrame,
            signal: str,
            daily_chg: str = "chg1D_fx",
            sl: str = "stop_loss",
            close: str = "close",
            **kwargs
        ) -> pd.DataFrame:
            """
            Thin wrapper that works on a **copy** and returns it.
            Prevents in-place modification issues in parallel runs.
            """
            df_copy = df.copy()
            return self.calculate_shares(
                df=df_copy,
                signal=signal,
                daily_chg=daily_chg,
                sl=sl,
                close=close,
                **kwargs
            )

def get_signal_column_names(
    signal: str,
    chg_suffix: str = "_chg1D_fx",
    sl_suffix: str = "_stop_loss",
    close_col: str = "close"
) -> dict:
    """
    Central place to define how column names are constructed for each signal.

    Returns dictionary with resolved column names.
    """
    if not signal or not isinstance(signal, str):
        raise ValueError("Signal must be a non-empty string")

    return {
        "signal": signal,
        "daily_chg": f"{signal}{chg_suffix}",
        "sl": f"{signal}{sl_suffix}",
        "close": close_col,
    }

def run_position_sizing_parallel(
    sizer: PositionSizing,
    df: pd.DataFrame,
    signals: List[str],
    chg_suffix: str = "_chg1D_fx",
    sl_suffix: str = "_stop_loss",
    close_col: str = "close",
    n_jobs: int = -1,
    verbose: int = 10,
) -> pd.DataFrame:
    """
    Run position sizing for multiple signals in parallel using consistent naming.

    Args:
        signals: List of signal column names (e.g. ['rsi2', 'ma_cross'])
        chg_suffix / sl_suffix: Allow customization of naming convention
        n_jobs: -1 = use all cores

    Returns:
        DataFrame with all new equity/shares/risk columns added
    """
    if not signals:
        raise ValueError("No signals provided")

    # Validate all signals exist
    missing_signals = [s for s in signals if s not in df.columns]
    if missing_signals:
        raise ValueError(f"Missing signal columns in DataFrame: {missing_signals}")

    # Prepare tasks with resolved column names
    tasks = []
    for signal in signals:
        cols = get_signal_column_names(
            signal=signal,
            chg_suffix=chg_suffix,
            sl_suffix=sl_suffix,
            close_col=close_col
        )

        # Early column existence check (fail fast)
        required_cols = [cols["signal"], cols["daily_chg"], cols["sl"], cols["close"]]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            logger.error(
                f"Skipping signal '{signal}': missing columns {missing}\n"
                f"Expected: {required_cols}"
            )
            continue

        tasks.append(cols)

    if not tasks:
        raise RuntimeError("No valid signal configurations found — check column names")

    logger.info(f"Processing {len(tasks)} signals in parallel (n_jobs={n_jobs})")

    # Run in parallel
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(sizer.calculate_shares_for_signal)(
            df=df,
            signal=task["signal"],
            daily_chg=task["daily_chg"],
            sl=task["sl"],
            close=task["close"],
        )
        for task in tasks
    )

    # Merge results safely
    result_df = df.copy()

    for processed_df in results:
        # Only add new columns (avoid overwriting original data)
        new_columns = [c for c in processed_df.columns if c not in result_df.columns]
        if new_columns:
            result_df[new_columns] = processed_df[new_columns]

    logger.info(f"Completed — added columns for {len(results)} signals")

    return result_df