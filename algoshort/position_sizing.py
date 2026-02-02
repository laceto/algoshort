import pandas as pd
import numpy as np


class PositionSizing:
    """
    Portfolio calculator for managing multiple trading strategies with risk-based
    share allocations.
    """
    
    def __init__(self, tolerance, mn, mx, equal_weight, avg, lot, initial_capital=100000):
        """
        Initialize the portfolio calculator.
        
        Parameters:
        -----------
        tolerance : float
            Risk tolerance parameter
        mn : float
            Minimum risk parameter
        mx : float
            Maximum risk parameter
        equal_weight : float
            Equal weight allocation factor
        avg : float
            Average risk value for fixed strategy
        lot : int
            Lot size for share calculations
        initial_capital : float, default=100000
            Initial capital for each strategy
        """
        self.tolerance = tolerance
        self.mn = mn
        self.mx = mx
        self.equal_weight = equal_weight
        self.avg = avg
        self.lot = lot
        self.initial_capital = initial_capital

    def eqty_risk_shares(self, px, sl, eqty, risk, fx, lot):
        r = sl - px
        if fx > 0:
            budget = eqty * risk * fx
        else:
            budget = eqty * risk
        shares = round(budget // (r * lot) * lot, 0)
        return shares
    
    def risk_appetite(self, eqty, tolerance, mn, mx, span, shape):
        """
        Calculate risk appetite based on equity curve and drawdown.
        
        Parameters:
        -----------
        eqty : pd.Series or array-like
            Equity curve series
        tolerance : float
            Tolerance for drawdown (<0)
        mn : float
            Minimum risk
        mx : float
            Maximum risk
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
        # Drawdown rebased
        eqty = pd.Series(eqty)
        watermark = eqty.expanding().max()  # all-time-high peak equity
        drawdown = eqty / watermark - 1  # drawdown from peak
        ddr = 1 - np.minimum(drawdown / tolerance, 1)  # drawdown rebased to tolerance from 0 to 1
        avg_ddr = ddr.ewm(span=span).mean()  # span rebased drawdown
        
        # Shape of the curve
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
        Generate prefixed column names for strategies and outputs.
        
        Parameters:
        -----------
        prefix : str
            Prefix to add to column names (typically the signal name)
            
        Returns:
        --------
        dict
            Dictionary containing all column name mappings
        """
        return {
            'strategies': [f'{prefix}_equal_weight', f'{prefix}_constant', 
                          f'{prefix}_concave', f'{prefix}_convex'],
            'share_cols': [f'{prefix}_shs_eql', f'{prefix}_shs_fxd', 
                          f'{prefix}_shs_ccv', f'{prefix}_shs_cvx'],
            'risk_configs': [
                {'col': f'{prefix}_concave', 'shape': -1, 'store': f'{prefix}_ccv'},
                {'col': f'{prefix}_convex', 'shape': 1, 'store': f'{prefix}_cvx'}
            ],
            'ccv': f'{prefix}_ccv',
            'cvx': f'{prefix}_cvx'
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
        # if not ((df.at[i-1, signal] == 0) and (df.at[i, signal] != 0)):
        #     return None
        if df.at[i, signal] == df.at[i-1, signal]: return None
        
        px = df.at[i, close]
        sl_price = df.at[i, sl]
        
        if px == sl_price:
            return None
        
        fx = 1
        
        # Calculate equal weight shares
        shs_eql = (df.at[i, cols['strategies'][0]] * self.equal_weight * fx // 
                   (px * self.lot)) * self.lot
        
        # Calculate risk-based shares
        share_configs = [
            {'strategy': cols['strategies'][1], 'risk': self.avg, 'key': cols['share_cols'][1]},
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
                lot=100
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