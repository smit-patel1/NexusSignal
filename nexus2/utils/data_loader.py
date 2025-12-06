"""Data loading utilities for NexusSignal 2.0."""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_raw_prices(
    ticker: str,
    timeframe: str = "1h",
    data_dir: str = "data/raw/prices"
) -> pd.DataFrame:
    """
    Load raw OHLCV data for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        timeframe: Data timeframe ('1h' or 'daily')
        data_dir: Directory containing raw price data
    
    Returns:
        DataFrame with OHLCV columns and datetime index
    
    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    path = Path(data_dir) / f"{ticker}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data found: {path}")
    
    df = pd.read_parquet(path)
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
    
    # Sort by time
    df = df.sort_index()
    
    return df


def load_processed_features(
    ticker: str,
    data_dir: str = "data/processed/features"
) -> pd.DataFrame:
    """
    Load processed features for a ticker (legacy v1 features if they exist).
    
    Note: These are legacy features from the old system. For NexusSignal 2.0,
    you should rebuild features using the new FeatureBuilder2_0 class.
    
    Args:
        ticker: Stock ticker symbol
        data_dir: Directory containing processed features
    
    Returns:
        DataFrame with features and datetime index
    
    Raises:
        FileNotFoundError: If the features file doesn't exist
    """
    path = Path(data_dir) / f"{ticker}_1h.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No features found: {path}")
    
    df = pd.read_parquet(path)
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
    
    # Sort by time
    df = df.sort_index()
    
    return df


def save_features(
    df: pd.DataFrame,
    ticker: str,
    output_dir: str = "data/interim",
    filename_suffix: str = "features_v2"
) -> Path:
    """
    Save processed features to parquet.
    
    Args:
        df: DataFrame to save
        ticker: Stock ticker symbol
        output_dir: Output directory
        filename_suffix: Suffix for filename (default: 'features_v2')
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / f"{ticker}_{filename_suffix}.parquet"
    df.to_parquet(filepath, compression='snappy')
    
    return filepath

