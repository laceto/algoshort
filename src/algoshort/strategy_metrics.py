import pandas as pd
import numpy as np
from typing import Optional
from algoshort.returns import ReturnsCalculator

class StrategyMetrics:
    """
    A class to calculate risk metrics for trading strategies, including profit ratio for trend-following strategies
    and tail ratio for mean-reversion strategies, based on log returns and cumulative returns.

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
        self._cache: dict[str, pd.Series] = {}  # Cache for calculations like profits and losses

    def _expectancy(self, win_rate: pd.Series, avg_win: pd.Series, avg_loss: pd.Series) -> pd.Series:
        """
        Calculates trading edge (expectancy) as (win_rate * avg_win) + (1 - win_rate) * avg_loss.

        Args:
            win_rate (pd.Series): Rolling win rate.
            avg_win (pd.Series): Average win amount.
            avg_loss (pd.Series): Average loss amount (negative).

        Returns:
            pd.Series: Trading edge series.
        """
        return win_rate * avg_win + (1 - win_rate) * avg_loss

    def _geometric_expectancy(self, win_rate: pd.Series, avg_win: pd.Series, avg_loss: pd.Series) -> pd.Series:
        """
        Calculates geometric expectancy as (1 + avg_win) ** win_rate * (1 + avg_loss) ** (1 - win_rate) - 1.

        Args:
            win_rate (pd.Series): Rolling win rate.
            avg_win (pd.Series): Average win amount.
            avg_loss (pd.Series): Average loss amount (negative).

        Returns:
            pd.Series: Geometric expectancy series.
        """
        return (1 + avg_win) ** win_rate * (1 + avg_loss) ** (1 - win_rate) - 1

    def _kelly(self, win_rate: pd.Series, avg_win: pd.Series, avg_loss: pd.Series) -> pd.Series:
        """
        Calculates Kelly criterion as win_rate / |avg_loss| - (1 - win_rate) / avg_win.

        Args:
            win_rate (pd.Series): Rolling win rate.
            avg_win (pd.Series): Average win amount.
            avg_loss (pd.Series): Average loss amount (negative).

        Returns:
            pd.Series: Kelly criterion series.
        """
        return win_rate / np.abs(avg_loss) - (1 - win_rate) / avg_win
    
    def _rolling_profits(self, returns: pd.Series, window: int) -> pd.Series:
        """
        Calculates the rolling sum of positive returns.

        Args:
            returns (pd.Series): Series of log returns.
            window (int): Rolling window size.

        Returns:
            pd.Series: Rolling sum of positive returns, forward-filled.
        """
        profit_roll = returns.copy()
        profit_roll[profit_roll < 0] = 0
        return profit_roll.rolling(window, min_periods=window).sum().ffill()

    def _rolling_losses(self, returns: pd.Series, window: int) -> pd.Series:
        """
        Calculates the rolling sum of negative returns.

        Args:
            returns (pd.Series): Series of log returns.
            window (int): Rolling window size.

        Returns:
            pd.Series: Rolling sum of negative returns, forward-filled.
        """
        loss_roll = returns.copy()
        loss_roll[loss_roll > 0] = 0
        return loss_roll.rolling(window, min_periods=window).sum().ffill()

    def _expanding_profits(self, returns: pd.Series) -> pd.Series:
        """
        Calculates the expanding sum of positive returns.

        Args:
            returns (pd.Series): Series of log returns.

        Returns:
            pd.Series: Expanding sum of positive returns, forward-filled.
        """
        profit_roll = returns.copy()
        profit_roll[profit_roll < 0] = 0
        return profit_roll.expanding().sum().ffill()

    def _expanding_losses(self, returns: pd.Series) -> pd.Series:
        """
        Calculates the expanding sum of negative returns.

        Args:
            returns (pd.Series): Series of log returns.

        Returns:
            pd.Series: Expanding sum of negative returns, forward-filled.
        """
        loss_roll = returns.copy()
        loss_roll[loss_roll > 0] = 0
        return loss_roll.expanding().sum().ffill()

    def _profit_ratio(self, profits: pd.Series, losses: pd.Series) -> pd.Series:
        """
        Calculates the profit ratio as profits / |losses|.

        Args:
            profits (pd.Series): Series of summed profits.
            losses (pd.Series): Series of summed losses (negative).

        Returns:
            pd.Series: Profit ratio, forward-filled.
        """
        return profits.ffill() / np.abs(losses.ffill())

    def _rolling_tail_ratio(self, cumul_returns: pd.Series, window: int, percentile: float, limit: float) -> pd.Series:
        """
        Calculates the rolling tail ratio as right_tail / |left_tail|, capped between [-limit, limit].

        Args:
            cumul_returns (pd.Series): Series of cumulative log returns.
            window (int): Rolling window size.
            percentile (float): Percentile for tail calculation (e.g., 0.05 for 5th percentile).
            limit (float): Maximum/minimum value for the ratio.

        Returns:
            pd.Series: Rolling tail ratio, capped and forward-filled.
        """
        np.seterr(all='ignore')
        left_tail = np.abs(cumul_returns.rolling(window, min_periods=window).quantile(percentile))
        right_tail = cumul_returns.rolling(window, min_periods=window).quantile(1 - percentile)
        tail = np.maximum(np.minimum(right_tail / left_tail, limit), -limit)
        return tail.ffill()

    def _expanding_tail_ratio(self, cumul_returns: pd.Series, percentile: float, limit: float) -> pd.Series:
        """
        Calculates the expanding tail ratio as right_tail / |left_tail|, capped between [-limit, limit].

        Args:
            cumul_returns (pd.Series): Series of cumulative log returns.
            percentile (float): Percentile for tail calculation (e.g., 0.05 for 5th percentile).
            limit (float): Maximum/minimum value for the ratio.

        Returns:
            pd.Series: Expanding tail ratio, capped and forward-filled.
        """
        np.seterr(all='ignore')
        left_tail = np.abs(cumul_returns.expanding().quantile(percentile))
        right_tail = cumul_returns.expanding().quantile(1 - percentile)
        tail = np.maximum(np.minimum(right_tail / left_tail, limit), -limit)
        return tail.ffill()
        
    def get_expectancies(self, df: pd.DataFrame, signal: str, window: int = 100, 
                        inplace: bool = False) -> pd.DataFrame:
        """
        Calculates rolling expectancy metrics for a trading strategy based on log returns.

        Args:
            df (pd.DataFrame): DataFrame containing the log returns column ('<signal>_log_returns').
            signal (str): Name of the signal column (e.g., 'bo_5', 'tt_52', 'sma_358') used for naming output columns.
            window (int): Lookback window for rolling calculations (e.g., 100 days). Defaults to 100.
            inplace (bool): If True, modify input DataFrame; else return a new one. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with additional columns:
                - '<signal>_trading_edge': Rolling expectancy as (win_rate * avg_win) + (1 - win_rate) * avg_loss.
                - '<signal>_geometric_expectancy': Rolling geometric expectancy as (1 + avg_win) ** win_rate * (1 + avg_loss) ** (1 - win_rate) - 1.
                - '<signal>_kelly': Rolling Kelly criterion as win_rate / |avg_loss| - (1 - win_rate) / avg_win.

        Raises:
            KeyError: If '<signal>_log_returns' not found in DataFrame.
            ValueError: If DataFrame is empty, window is invalid, or log returns column contains non-numeric data.
        """
        try:
            # Validate inputs
            if df.empty:
                raise ValueError("Input DataFrame is empty.")
            if window < 1:
                raise ValueError("window must be positive.")
            if len(df) < window:
                raise ValueError(f"DataFrame length ({len(df)}) must be >= window ({window}).")
            log_returns_col = f'{signal}_log_returns'
            if log_returns_col not in df.columns:
                raise KeyError(f"Log returns column '{log_returns_col}' not found in DataFrame.")
            if not np.issubdtype(df[log_returns_col].dtype, np.number):
                raise ValueError(f"Log returns column '{log_returns_col}' must contain numeric data.")

            # Create working DataFrame
            result_df = df if inplace else df.copy()

            # Cache key for expectancy metrics
            cache_key_prefix = f"expectancy_{signal}_{window}"

            # Separate profits from losses
            tt_log_returns = result_df[[log_returns_col]].copy()
            loss_roll = tt_log_returns.copy()
            loss_roll[loss_roll > 0] = np.nan
            win_roll = tt_log_returns.copy()
            win_roll[win_roll < 0] = np.nan

            # Calculate rolling metrics
            cache_key_win_rate = f"{cache_key_prefix}_win_rate"
            cache_key_loss_rate = f"{cache_key_prefix}_loss_rate"
            cache_key_avg_win = f"{cache_key_prefix}_avg_win"
            cache_key_avg_loss = f"{cache_key_prefix}_avg_loss"

            if cache_key_win_rate in self._cache:
                win_rate = self._cache[cache_key_win_rate]
            else:
                win_rate = win_roll.rolling(window, min_periods=window).count() / window
                self._cache[cache_key_win_rate] = win_rate

            if cache_key_loss_rate in self._cache:
                loss_rate = self._cache[cache_key_loss_rate]
            else:
                loss_rate = loss_roll.rolling(window, min_periods=window).count() / window
                self._cache[cache_key_loss_rate] = loss_rate

            if cache_key_avg_win in self._cache:
                avg_win = self._cache[cache_key_avg_win]
            else:
                avg_win = win_roll.fillna(0).rolling(window, min_periods=window).mean()
                self._cache[cache_key_avg_win] = avg_win

            if cache_key_avg_loss in self._cache:
                avg_loss = self._cache[cache_key_avg_loss]
            else:
                avg_loss = loss_roll.fillna(0).rolling(window, min_periods=window).mean()
                self._cache[cache_key_avg_loss] = avg_loss

            # Calculate expectancy metrics
            result_df[f'{signal}_trading_edge'] = self._expectancy(win_rate[log_returns_col], 
                                                                 avg_win[log_returns_col], 
                                                                 avg_loss[log_returns_col]).ffill()
            result_df[f'{signal}_geometric_expectancy'] = self._geometric_expectancy(win_rate[log_returns_col], 
                                                                                   avg_win[log_returns_col], 
                                                                                   avg_loss[log_returns_col]).ffill()
            result_df[f'{signal}_kelly'] = self._kelly(win_rate[log_returns_col], 
                                                      avg_win[log_returns_col], 
                                                      avg_loss[log_returns_col]).ffill()

            return result_df

        except Exception as e:
            raise ValueError(f"Error computing expectancy metrics: {str(e)}")
        
    def get_risk_metrics(self, df: pd.DataFrame, signal: str, window: int = 252, 
                         percentile: float = 0.05, limit: float = 5, 
                         inplace: bool = False) -> pd.DataFrame:
        """
        Calculates risk metrics for trend-following and mean-reversion strategies based on log returns and cumulative returns.

        Args:
            df (pd.DataFrame): DataFrame containing '<signal>_log_returns' and '<signal>_cumul' columns.
            signal (str): Name of the signal column (e.g., 'bo_5', 'tt_52', 'sma_358') used for naming output columns.
            window (int): Lookback window for rolling calculations (e.g., 252 for yearly). Defaults to 252.
            percentile (float): Percentile for tail ratio calculation (e.g., 0.05 for 5th/95th percentiles). Defaults to 0.05.
            limit (float): Maximum/minimum value for tail ratio. Defaults to 5.
            inplace (bool): If True, modify input DataFrame; else return a new one. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with additional columns:
                - '<signal>_pr_roll': Rolling profit ratio as rolling_profits / |rolling_losses|.
                - '<signal>_pr': Expanding profit ratio as expanding_profits / |expanding_losses|.
                - '<signal>_tr_roll': Rolling tail ratio as right_tail / |left_tail|, capped at [-limit, limit].
                - '<signal>_tr': Expanding tail ratio as right_tail / |left_tail|, capped at [-limit, limit].

        Raises:
            KeyError: If '<signal>_log_returns' or '<signal>_cumul' not found in DataFrame.
            ValueError: If DataFrame is empty, window/percentile/limit are invalid, or columns contain non-numeric data.
        """
        try:
            # Validate inputs
            if df.empty:
                raise ValueError("Input DataFrame is empty.")
            if window < 1:
                raise ValueError("window must be positive.")
            if len(df) < window:
                raise ValueError(f"DataFrame length ({len(df)}) must be >= window ({window}).")
            if not 0 < percentile < 0.5:
                raise ValueError("percentile must be between 0 and 0.5 (e.g., 0.05 for 5th/95th percentiles).")
            if limit <= 0:
                raise ValueError("limit must be positive.")
            log_returns_col = f'{signal}_log_returns'
            cumul_returns_col = f'{signal}_cumul'
            if log_returns_col not in df.columns:
                raise KeyError(f"Log returns column '{log_returns_col}' not found in DataFrame.")
            if cumul_returns_col not in df.columns:
                raise KeyError(f"Cumulative returns column '{cumul_returns_col}' not found in DataFrame.")
            if not np.issubdtype(df[log_returns_col].dtype, np.number) or \
               not np.issubdtype(df[cumul_returns_col].dtype, np.number):
                raise ValueError(f"Columns '{log_returns_col}' and '{cumul_returns_col}' must contain numeric data.")

            # Create working DataFrame
            result_df = df if inplace else df.copy()

            # Cache key for risk metrics
            cache_key_prefix = f"risk_metrics_{signal}_{window}_{percentile}_{limit}"

            # Calculate profit ratio metrics
            cache_key_roll_profits = f"{cache_key_prefix}_roll_profits"
            cache_key_roll_losses = f"{cache_key_prefix}_roll_losses"
            cache_key_exp_profits = f"{cache_key_prefix}_exp_profits"
            cache_key_exp_losses = f"{cache_key_prefix}_exp_losses"

            if cache_key_roll_profits in self._cache:
                roll_profits = self._cache[cache_key_roll_profits]
            else:
                roll_profits = self._rolling_profits(result_df[log_returns_col], window)
                self._cache[cache_key_roll_profits] = roll_profits

            if cache_key_roll_losses in self._cache:
                roll_losses = self._cache[cache_key_roll_losses]
            else:
                roll_losses = self._rolling_losses(result_df[log_returns_col], window)
                self._cache[cache_key_roll_losses] = roll_losses

            if cache_key_exp_profits in self._cache:
                exp_profits = self._cache[cache_key_exp_profits]
            else:
                exp_profits = self._expanding_profits(result_df[log_returns_col])
                self._cache[cache_key_exp_profits] = exp_profits

            if cache_key_exp_losses in self._cache:
                exp_losses = self._cache[cache_key_exp_losses]
            else:
                exp_losses = self._expanding_losses(result_df[log_returns_col])
                self._cache[cache_key_exp_losses] = exp_losses

            result_df[f'{signal}_pr_roll'] = self._profit_ratio(roll_profits, roll_losses)
            result_df[f'{signal}_pr'] = self._profit_ratio(exp_profits, exp_losses)

            # Calculate tail ratio metrics
            cache_key_roll_tr = f"{cache_key_prefix}_roll_tr"
            cache_key_exp_tr = f"{cache_key_prefix}_exp_tr"

            if cache_key_roll_tr in self._cache:
                roll_tr = self._cache[cache_key_roll_tr]
            else:
                roll_tr = self._rolling_tail_ratio(result_df[cumul_returns_col], window, percentile, limit)
                self._cache[cache_key_roll_tr] = roll_tr

            if cache_key_exp_tr in self._cache:
                exp_tr = self._cache[cache_key_exp_tr]
            else:
                exp_tr = self._expanding_tail_ratio(result_df[cumul_returns_col], percentile, limit)
                self._cache[cache_key_exp_tr] = exp_tr

            result_df[f'{signal}_tr_roll'] = roll_tr
            result_df[f'{signal}_tr'] = exp_tr

            return result_df

        except Exception as e:
            raise ValueError(f"Error computing risk metrics: {str(e)}")