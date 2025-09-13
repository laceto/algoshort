import pandas as pd
from algoshort.regime_bo import RegimeBO
from algoshort.regime_ma import TripleMACrossoverRegime
from algoshort.utils import load_config, extract_signal_name
from algoshort.returns import ReturnsCalculator
from algoshort.strategy_metrics import StrategyMetrics

def calculate_relative_prices(df, symbol: str, benchmark: str) -> pd.DataFrame:
    
    return df.calculate_relative_prices(symbol=symbol, benchmark_symbol=benchmark)


def calculate_metrics(
        stock_data: pd.DataFrame,
        config_path: str = 'config.json'
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
    signal_names = extract_signal_name(config_path)
    
    # Validate signal columns
    missing_signals = [name for name in signal_names if name not in stock_data.columns]
    if missing_signals:
        raise ValueError(f"Signal columns not found in DataFrame: {missing_signals}")
    
    # Initialize ReturnsCalculator
    strategy_metrics = StrategyMetrics(stock_data)
    
    # Calculate risk metrics for each signal
    try:
        for signal in signal_names:
            stock_data = strategy_metrics.get_risk_metrics(
                df=stock_data,
                signal=signal,
                window=config['metrics']['risk_window'],
                percentile=config['metrics']['percentile'],
                limit=config['metrics']['limit'],
                inplace=True
            )
    except Exception as e:
        raise ValueError(f"Error calculating risk metrics for signal {signal}: {str(e)}")
    
    return stock_data


def calculate_return(
        stock_data: pd.DataFrame,
        config_path: str = 'config.json'
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
    signal_names = extract_signal_name(config_path)
    
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
        config_path='config.json'
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
                             relative=config['regimes']['breakout']['relative'], inplace=True)
    regime_bo.compute_regime(regime_type='turtle', fast_window=config['regimes']['turtle']['fast_window'],
                             window=config['regimes']['turtle']['slow_window'],
                             relative=config['regimes']['turtle']['relative'], inplace=True)

    for ma_type in config['regimes']['ma_crossover']['ma_type']:
        regime_ma.compute_ma_regime(
            ma_type=ma_type,
            short_window=config['regimes']['ma_crossover']['short_window'],
            medium_window=config['regimes']['ma_crossover']['medium_window'],
            long_window=config['regimes']['ma_crossover']['long_window'],
            relative=config['regimes']['ma_crossover']['relative'],
            inplace=True
        )
    
    # Get signal column names
    signal_names = extract_signal_name(config_path)
    
    # Verify signal columns exist
    missing_signals = [name for name in signal_names if name not in df.columns]
    if missing_signals:
        raise ValueError(f"Signal columns not generated: {missing_signals}")
    
    # Select signal columns dynamically
    signal_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in ['rbo_', 'bo_', 'rtt_', 'tt_', 'rsma_', 'sma_', 'rema_', 'ema_'])]
    
    return df, signal_columns