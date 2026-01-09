import pandas as pd
from algoshort.regime_bo import RegimeBO
from algoshort.regime_ma import TripleMACrossoverRegime
from algoshort.regime_fc import RegimeFC
from algoshort.utils import load_config, extract_signal_name
from algoshort.returns import ReturnsCalculator
from algoshort.strategy_metrics import StrategyMetrics
import warnings

def calculate_trading_edge(
        stock_data: pd.DataFrame,
        signal_columns,
        config_path: str = './config.json'
) -> pd.DataFrame:
    """
    Calculate risk metrics for each signal using StrategyMetrics.
    
    Args:
        stock_data: DataFrame with OHLC and signal columns.
        symbol: Ticker symbol (e.g., 'AAPL').
        config_path: Path to JSON config file.
    
    Returns:
        tuple: (DataFrame with risk metric columns, list of signal column names).
    
    Raises:
        ValueError: If required columns or signals are missing.
    """
    
    # Load config
    config = load_config(config_path)
    
    # Get signal names
    signal_names = signal_columns
    
    # Validate signal columns
    missing_signals = [name for name in signal_names if name not in stock_data.columns]
    if missing_signals:
        raise ValueError(f"Signal columns not found in DataFrame: {missing_signals}")
    
    # Initialize ReturnsCalculator
    strategy_metrics = StrategyMetrics(stock_data)
    
# 1. Create a list to store all the new metric DataFrames
    new_metrics_frames = []
    
    # Calculate metrics for each signal
    try:
        for signal in signal_names:
            # We call the functions with inplace=False to get a fresh DF of JUST the new columns
            # Note: Ensure your get_expectancies/get_risk_metrics return 
            # the combined DF or just the new columns based on your implementation.
            
            # Here we assume the refactored version that creates new columns:
            metrics_df = strategy_metrics.get_expectancies(
                df=stock_data,
                signal=signal,
                window=100,
                inplace=False
            )
            
            # To avoid duplicating the original stock_data columns in every loop, 
            # we extract ONLY the new columns added in this iteration.
            new_cols = metrics_df.iloc[:, len(stock_data.columns):]
            new_metrics_frames.append(new_cols)

    except Exception as e:
        raise ValueError(f"Error calculating risk metrics for signal {signal}: {str(e)}")
    
    # 2. Final Tip: Concatenate all new columns at once and force a copy to defragment
    if new_metrics_frames:
        # Join the original data with all new metric columns in one go
        stock_data = pd.concat([stock_data] + new_metrics_frames, axis=1)
        
        # Defragment memory and clean up the block manager
        stock_data = stock_data.copy()
    
    return stock_data

def calculate_risk_metrics(
        stock_data: pd.DataFrame,
        signal_columns: list,
        config_path: str = './config.json'
) -> pd.DataFrame:
    """
    Calculate risk metrics for each signal using a batch-concatenation 
    approach to avoid DataFrame fragmentation.
    """
    # Load config
    config = load_config(config_path)
    
    # Get signal names
    signal_names = signal_columns
    
    # Validate signal columns
    missing_signals = [name for name in signal_names if name not in stock_data.columns]
    if missing_signals:
        raise ValueError(f"Signal columns not found in DataFrame: {missing_signals}")
    
    # Initialize StrategyMetrics
    strategy_metrics = StrategyMetrics(stock_data)
    
    # 1. Store the original column count to isolate new columns later
    original_col_count = len(stock_data.columns)
    new_metrics_frames = []
    
    # Calculate risk metrics for each signal
    try:
        for signal in signal_names:
            # Generate metrics (inplace=False returns a new DataFrame)
            metrics_df = strategy_metrics.get_risk_metrics(
                df=stock_data,
                signal=signal,
                window=config['metrics']['risk_window'],
                percentile=config['metrics']['percentile'],
                limit=config['metrics']['limit'],
                inplace=False
            )
            
            # Isolate only the newly created columns
            new_cols = metrics_df.iloc[:, original_col_count:]
            new_metrics_frames.append(new_cols)

    except Exception as e:
        raise ValueError(f"Error calculating risk metrics for signal {signal}: {str(e)}")
    
    # 2. Final Tip: Batch concat and defragment
    if new_metrics_frames:
        # Concatenate original data with all results in one operation
        stock_data = pd.concat([stock_data] + new_metrics_frames, axis=1)
        
        # Consolidate memory blocks
        stock_data = stock_data.copy()
    
    return stock_data

def calculate_return(
        stock_data: pd.DataFrame,
        signal_columns,
        config_path: str = './config.json'
) -> tuple[pd.DataFrame, list]:
    """
    Calculate returns for each signal using ReturnsCalculator.
    
    Args:
        stock_data: DataFrame with OHLC and signal columns.
        symbol: Ticker symbol (e.g., 'AAPL').
        config_path: Path to JSON config file.
    
    Returns:
        tuple: (DataFrame with return columns, list of signal column names).
    
    Raises:
        ValueError: If required columns or signals are missing.
    """
    
    # Load config
    config = load_config(config_path)
    
    # Get signal names
    signal_names = signal_columns
    
    # Validate signal columns
    missing_signals = [name for name in signal_names if name not in stock_data.columns]
    if missing_signals:
        raise ValueError(f"Signal columns not found in DataFrame: {missing_signals}")
    
    # Initialize ReturnsCalculator
    returns_calc = ReturnsCalculator(ohlc_stock=stock_data)
    
    # Calculate returns for each signal
    for signal in signal_names:
        stock_data = returns_calc.get_returns(
            df=stock_data,
            signal=signal,
            relative=config['returns']['relative'],
            inplace=True
        )
    
    return stock_data

def generate_signals(
        df: pd.DataFrame, 
        config_path='./config.json',
        relative: bool = True
        ) -> tuple[pd.DataFrame, list]:
    """
    Generates signals for breakout, Turtle Trader, and MA crossover regimes.
    
    Args:
        df: DataFrame with OHLC data (e.g., AAPL_Open, SPY_Close).
        symbol: Ticker symbol (e.g., 'AAPL').
        benchmark: Benchmark ticker (e.g., 'SPY').
        config_path: Path to JSON config file.
    
    Returns:
        tuple: (DataFrame with signal columns, list of signal column names).
    
    Raises:
        ValueError: If input DataFrame is missing required columns or config is invalid.
    """
    regime_bo = RegimeBO(ohlc_stock=df)
    regime_ma = TripleMACrossoverRegime(ohlc_stock=df)

    config = load_config(config_path)

    regime_bo.compute_regime(regime_type='breakout', window=config['regimes']['breakout']['bo_window'],
                             relative=relative, inplace=True)
    regime_bo.compute_regime(regime_type='turtle', fast_window=config['regimes']['turtle']['fast_window'],
                             window=config['regimes']['turtle']['slow_window'],
                             relative=relative, inplace=True)

    for ma_type in config['regimes']['ma_crossover']['ma_type']:
        regime_ma.compute_ma_regime(
            ma_type=ma_type,
            short_window=config['regimes']['ma_crossover']['short_window'],
            medium_window=config['regimes']['ma_crossover']['medium_window'],
            long_window=config['regimes']['ma_crossover']['long_window'],
            relative=relative,
            inplace=True
        )

    regime_fc = RegimeFC(df=df)
    df = regime_fc.compute_regime(
        relative = relative,
        lvl = config['regimes']['floor_ceiling']['lvl'],
        vlty_n = config['regimes']['floor_ceiling']['vlty_n'],
        threshold = config['regimes']['floor_ceiling']['threshold'],
        dgt = config['regimes']['floor_ceiling']['dgt'],
        d_vol = config['regimes']['floor_ceiling']['d_vol'],
        dist_pct = config['regimes']['floor_ceiling']['dist_pct'],
        retrace_pct = config['regimes']['floor_ceiling']['retrace_pct'],
        r_vol = config['regimes']['floor_ceiling']['r_vol']
    )
    
    # Get signal column names
    signal_names = extract_signal_name(config_path)
    
    # Verify signal columns exist
    missing_signals = [name for name in signal_names if name not in df.columns]
    if missing_signals:
        warnings.warn(f"Signal columns not generated: {missing_signals}", UserWarning)
    
    # Select signal columns dynamically
    signal_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in ['rbo_', 'bo_', 'rtt_', 'tt_', 'rsma_', 'sma_', 'rema_', 'ema_', 'rg', 'rrg'])]
    
    signal_columns = [
        item for item in signal_columns
        if not any(keyword in item for keyword in ['short', 'medium', 'long'])
    ]

    return df, signal_columns