import pandas as pd
import numpy as np
import itertools
from algoshort.stop_loss import StopLossCalculator

class StrategyOptimizer:
    def __init__(self, data: pd.DataFrame, calculator: StopLossCalculator, equity_func):
        """
        Args:
            data: The full OHLC DataFrame.
            calculator: An instance of StopLossCalculator.
            equity_func: Your custom function that returns a 1-row, 4-col DataFrame.
        """
        self.data = data
        self.calc = calculator
        self.equity_func = equity_func
        self.optimization_results = pd.DataFrame()
        self.best_params = {}

   
    def run_grid_search(self, is_data, signal, windows, multipliers, price_col = 'close'):
        """Performs a standard grid search on a specific data segment."""
        results = []
        self.calc.data = is_data
        
        for w, m in itertools.product(windows, multipliers):
            temp_df = self.calc.atr_stop_loss(signal=signal, window=w, multiplier=m, price_col=price_col)
            row = self.equity_func(temp_df, signal, price_col = price_col)
            row.update({'window': w, 'multiplier': m})
            results.append(row)
            
        return pd.DataFrame(results)

    def rolling_walk_forward(self, signal, close_col, windows, multipliers, n_segments=4):
        """Executes a rolling WFA and returns OOS metrics and parameter stability."""
        segment_size = len(self.data) // (n_segments + 1)
        print(f"Debug: Data Length: {len(self.data)}, Segments: {n_segments}")
        oos_results = []
        param_history = []

        for i in range(n_segments):
            # Define Splits
            is_data = self.data.iloc[i * segment_size : (i + 1) * segment_size]
            oos_data = self.data.iloc[(i + 1) * segment_size : (i + 2) * segment_size]

            # In-Sample Optimization
            is_df = self.run_grid_search(is_data, signal, windows, multipliers, price_col = close_col)
            best_row = is_df.sort_values('convex', ascending=False).iloc[0]
            
            w_best, m_best = int(best_row['window']), best_row['multiplier']
            param_history.append({'segment': i+1, 'window': w_best, 'multiplier': m_best})

            # Out-of-Sample Validation
            self.calc.data = oos_data
            final_oos = self.calc.atr_stop_loss(signal, window=w_best, multiplier=m_best, price_col=close_col)
            oos_metrics = self.equity_func(final_oos, signal, close_col)
            oos_metrics['segment'] = i + 1
            # oos_metrics['w_best'] = w_best
            # oos_metrics['m_best'] = m_best
            oos_results.append(oos_metrics)

            # # Update the placeholder every loop
            # self.final_best_params = {
            #     'window': w_best,
            #     'multiplier': m_best,
            #     'segment_index': i
            # }

        # Calculate Stability
        history_df = pd.DataFrame(param_history)
        stability = {
            'window_cv': history_df['window'].std() / history_df['window'].mean(),
            'multiplier_cv': history_df['multiplier'].std() / history_df['multiplier'].mean()
        }
        
        # return pd.DataFrame(oos_results)
        return pd.DataFrame(oos_results), stability, param_history

    def sensitivity_analysis(self, signal, best_w, best_m, variance=0.2):
        """Tests the 'plateau' around the optimal parameters."""
        w_range = [int(best_w * r) for r in [1-variance, 1, 1+variance]]
        m_range = [round(best_m * r, 2) for r in [1-variance, 1, 1+variance]]
        
        # Test on full data
        self.calc.data = self.data
        results = self.run_grid_search(self.data, signal, w_range, m_range)
        
        peak_equity = results[(results['window']==best_w) & (results['multiplier']==best_m)]['convex'].iloc[0]
        avg_equity = results['convex'].mean()
        
        return (avg_equity / peak_equity) * 100, results