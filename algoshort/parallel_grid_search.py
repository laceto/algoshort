"""
Parallel Grid Search Implementation for SignalGridSearch Class

This module provides two implementations of parallel grid search:
1. Using Python's multiprocessing.Pool
2. Using joblib.Parallel (recommended for better memory management)

Add these methods to your SignalGridSearch class.
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def _process_single_combination(combo, df, direction_col, allow_flips, 
                                 require_regime_alignment, verbose):
    """
    Worker function to process a single signal combination.
    Must be defined at module level for pickling compatibility.
    
    Parameters:
    -----------
    combo : dict
        Combination dictionary with 'entry', 'exit', 'name' keys
    df : pd.DataFrame
        Input DataFrame with all signals
    direction_col : str
        Direction filter column
    allow_flips : bool
        Allow position flips
    require_regime_alignment : bool
        Require regime alignment
    verbose : bool
        Print detailed logs
        
    Returns:
    --------
    dict
        Result dictionary with statistics and metadata
    """
    try:
        # Import here to avoid issues with multiprocessing
        from HybridSignalCombiner import HybridSignalCombiner
        
        # Create a working copy
        df_test = df.copy()
        
        # Initialize combiner
        combiner = HybridSignalCombiner(
            direction_col=direction_col,
            entry_col=combo['entry'],
            exit_col=combo['exit'],
            verbose=verbose
        )
        
        # Combine signals
        output_col = combo['name']
        df_test = combiner.combine_signals(
            df_test,
            output_col=output_col,
            allow_flips=allow_flips,
            require_regime_alignment=require_regime_alignment
        )
        
        # Add metadata
        df_test = combiner.add_signal_metadata(df_test, output_col)
        
        # Get trade summary
        summary = combiner.get_trade_summary(df_test, output_col)
        
        # Return result with combined signal column
        result = {
            'combination_name': combo['name'],
            'entry_signal': combo['entry'],
            'exit_signal': combo['exit'],
            'direction_signal': direction_col,
            'output_column': output_col,
            
            # Trade statistics
            'total_trades': summary['total_entries'],
            'long_trades': summary['entry_long_count'],
            'short_trades': summary['entry_short_count'],
            'long_to_short_flips': summary['flip_long_to_short_count'],
            'short_to_long_flips': summary['flip_short_to_long_count'],
            
            # Position distribution
            'long_bars': summary['long_bars'],
            'short_bars': summary['short_bars'],
            'flat_bars': summary['flat_bars'],
            'long_pct': summary['long_pct'],
            'short_pct': summary['short_pct'],
            'flat_pct': summary['flat_pct'],
            
            # Average holding periods
            'avg_bars_per_long_trade': summary['avg_bars_per_long_trade'],
            'avg_bars_per_short_trade': summary['avg_bars_per_short_trade'],
            
            # Include the combined signal column for later addition to main df
            'combined_signal': df_test[output_col].copy(),
            
            'success': True,
            'error': None
        }
        
        return result
        
    except Exception as e:
        # Return error result
        return {
            'combination_name': combo['name'],
            'entry_signal': combo['entry'],
            'exit_signal': combo['exit'],
            'direction_signal': direction_col,
            'output_column': combo['name'],
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'long_to_short_flips': 0,
            'short_to_long_flips': 0,
            'long_bars': 0,
            'short_bars': 0,
            'flat_bars': 0,
            'long_pct': 0,
            'short_pct': 0,
            'flat_pct': 0,
            'avg_bars_per_long_trade': 0,
            'avg_bars_per_short_trade': 0,
            'combined_signal': None,
            'success': False,
            'error': str(e)
        }


# Add this method to your SignalGridSearch class
def run_grid_search_parallel(self,
                             allow_flips=True,
                             require_regime_alignment=True,
                             verbose=False,
                             n_jobs=-1,
                             backend='multiprocessing',
                             batch_size=None):
    """
    Run HybridSignalCombiner on all signal combinations in parallel.
    
    This method parallelizes the grid search across multiple CPU cores,
    significantly reducing computation time for large parameter spaces.
    Each signal combination is processed independently, making the problem
    embarrassingly parallel.
    
    Parameters:
    -----------
    allow_flips : bool, default=True
        Allow direct position flips (long to short or vice versa)
    require_regime_alignment : bool, default=True
        Require entries to align with direction signal
    verbose : bool, default=False
        Print progress for each combination (not recommended with parallel execution)
    n_jobs : int, default=-1
        Number of parallel jobs to run:
        - -1: Use all available CPU cores
        - 1: Sequential execution (same as original run_grid_search)
        - n: Use n CPU cores
    backend : str, default='multiprocessing'
        Parallelization backend:
        - 'multiprocessing': Use Python's multiprocessing.Pool
        - 'joblib': Use joblib.Parallel (better error handling, requires joblib)
    batch_size : int, default=None
        Number of combinations to process per batch (helps with memory management)
        - None: Automatically determined
        - n: Process n combinations at a time
        
    Returns:
    --------
    pd.DataFrame
        Results with combined signals for each combination
        
    Performance Notes:
    ------------------
    - Expected speedup: ~(n_cores - 1)x on CPU-bound tasks
    - Memory usage: Scales with n_jobs (each worker holds a copy of df)
    - For large DataFrames, consider reducing n_jobs or using batch_size
    - Progress tracking is approximate with multiprocessing
    
    Examples:
    ---------
    # Use all cores
    results = searcher.run_grid_search_parallel(n_jobs=-1)
    
    # Use 4 cores
    results = searcher.run_grid_search_parallel(n_jobs=4)
    
    # Use joblib backend with batching
    results = searcher.run_grid_search_parallel(
        n_jobs=-1,
        backend='joblib',
        batch_size=10
    )
    """
    # Generate grid
    grid = self.generate_grid()
    
    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        raise ValueError(f"n_jobs must be -1 or positive integer, got {n_jobs}")
    
    # If n_jobs=1, fall back to sequential
    if n_jobs == 1:
        print("n_jobs=1: Using sequential execution")
        return self.run_grid_search(
            allow_flips=allow_flips,
            require_regime_alignment=require_regime_alignment,
            verbose=verbose
        )
    
    # Print header
    print(f"\n{'='*70}")
    print(f"RUNNING PARALLEL GRID SEARCH")
    print(f"{'='*70}")
    print(f"Direction column: {self.direction_col}")
    print(f"Allow flips: {allow_flips}")
    print(f"Require regime alignment: {require_regime_alignment}")
    print(f"Backend: {backend}")
    print(f"Parallel jobs: {n_jobs} cores")
    print(f"\nProcessing {len(grid)} combinations in parallel...")
    
    # Execute based on backend
    if backend == 'multiprocessing':
        results = self._run_multiprocessing(
            grid, allow_flips, require_regime_alignment, 
            verbose, n_jobs, batch_size
        )
    elif backend == 'joblib':
        results = self._run_joblib(
            grid, allow_flips, require_regime_alignment,
            verbose, n_jobs, batch_size
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'multiprocessing' or 'joblib'")
    
    # Add combined signals to main dataframe
    print("\nAdding combined signal columns to main dataframe...")
    for result in results:
        if result['success'] and result['combined_signal'] is not None:
            self.df[result['output_column']] = result['combined_signal']
    
    # Remove combined_signal from results (no longer needed)
    for result in results:
        if 'combined_signal' in result:
            del result['combined_signal']
    
    # Convert to DataFrame
    self.results = pd.DataFrame(results)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Successful combinations: {self.results['success'].sum()}")
    print(f"Failed combinations: {(~self.results['success']).sum()}")
    
    return self.results


def _run_multiprocessing(self, grid, allow_flips, require_regime_alignment,
                         verbose, n_jobs, batch_size):
    """
    Execute grid search using multiprocessing.Pool.
    
    Parameters:
    -----------
    grid : list
        List of combination dictionaries
    allow_flips : bool
        Allow position flips
    require_regime_alignment : bool
        Require regime alignment
    verbose : bool
        Print detailed logs
    n_jobs : int
        Number of parallel jobs
    batch_size : int or None
        Batch size for processing
        
    Returns:
    --------
    list
        List of result dictionaries
    """
    # Create partial function with fixed parameters
    worker_func = partial(
        _process_single_combination,
        df=self.df,
        direction_col=self.direction_col,
        allow_flips=allow_flips,
        require_regime_alignment=require_regime_alignment,
        verbose=verbose
    )
    
    # Determine batch size
    if batch_size is None:
        batch_size = max(1, len(grid) // n_jobs)
    
    # Process with multiprocessing
    with Pool(processes=n_jobs) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(worker_func, grid, chunksize=batch_size),
            total=len(grid),
            desc="Testing combinations"
        ))
    
    return results


def _run_joblib(self, grid, allow_flips, require_regime_alignment,
                verbose, n_jobs, batch_size):
    """
    Execute grid search using joblib.Parallel.
    
    Requires: pip install joblib
    
    Parameters:
    -----------
    grid : list
        List of combination dictionaries
    allow_flips : bool
        Allow position flips
    require_regime_alignment : bool
        Require regime alignment
    verbose : bool
        Print detailed logs
    n_jobs : int
        Number of parallel jobs
    batch_size : int or None
        Batch size for processing
        
    Returns:
    --------
    list
        List of result dictionaries
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError(
            "joblib backend requires joblib. Install with: pip install joblib"
        )
    
    # Determine batch size
    if batch_size is None:
        batch_size = max(1, len(grid) // n_jobs)
    
    # Process with joblib
    results = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=10)(
        delayed(_process_single_combination)(
            combo=combo,
            df=self.df,
            direction_col=self.direction_col,
            allow_flips=allow_flips,
            require_regime_alignment=require_regime_alignment,
            verbose=verbose
        )
        for combo in tqdm(grid, desc="Testing combinations")
    )
    
    return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    """
    Example usage of parallel grid search.
    
    Note: The 'if __name__ == "__main__"' guard is REQUIRED on Windows
    when using multiprocessing to prevent infinite process spawning.
    """