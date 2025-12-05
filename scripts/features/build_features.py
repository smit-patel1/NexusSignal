"""
Feature builder for NexusSignal.
Loads raw price data, computes features, and saves processed datasets.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.features.indicators import add_all_indicators
from scripts.features.utils import (
    ensure_datetime_index,
    add_lagged_returns,
    add_forward_returns,
    add_volatility,
    add_rolling_stats,
    add_time_features,
    merge_timeframes,
    clean_dataframe,
)
from scripts.utils.parquet_utils import read_parquet, write_parquet


# Data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_PRICES_DIR = DATA_DIR / "raw" / "prices"
PROCESSED_FEATURES_DIR = DATA_DIR / "processed" / "features"

# Ensure output directory exists
PROCESSED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw daily and hourly price data for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Tuple of (daily_df, hourly_df)
    """
    daily_path = RAW_PRICES_DIR / f"{ticker}_daily.parquet"
    hourly_path = RAW_PRICES_DIR / f"{ticker}_1h.parquet"

    daily_df = read_parquet(daily_path)
    hourly_df = read_parquet(hourly_path)

    if daily_df is None:
        raise FileNotFoundError(f"Daily data not found for {ticker} at {daily_path}")

    if hourly_df is None:
        raise FileNotFoundError(f"Hourly data not found for {ticker} at {hourly_path}")

    # Ensure datetime index
    daily_df = ensure_datetime_index(daily_df)
    hourly_df = ensure_datetime_index(hourly_df)

    return daily_df, hourly_df


def build_features_for_ticker(ticker: str) -> pd.DataFrame:
    """
    Build complete feature set for a single ticker.

    This function:
    1. Loads daily and hourly raw data
    2. Computes technical indicators
    3. Adds returns, volatility, and rolling stats
    4. Merges multi-timeframe features
    5. Adds target variables
    6. Cleans and saves the result

    Args:
        ticker: Stock ticker symbol

    Returns:
        DataFrame with all features
    """
    print(f"\n{'=' * 60}")
    print(f"Building features for {ticker}")
    print(f"{'=' * 60}")

    # Step 1: Load raw data
    print("Loading raw data...")
    daily_df, hourly_df = load_raw_data(ticker)
    print(f"  Daily: {len(daily_df)} rows")
    print(f"  Hourly: {len(hourly_df)} rows")

    # Step 2: Build features for daily timeframe
    print("Computing daily features...")
    daily_df = add_all_indicators(daily_df)
    daily_df = add_volatility(daily_df, windows=[5, 10, 20])
    daily_df = add_rolling_stats(daily_df, windows=[5, 10, 20])

    # Step 3: Build features for hourly timeframe
    print("Computing hourly features...")
    hourly_df = add_all_indicators(hourly_df)
    hourly_df = add_lagged_returns(hourly_df, horizons=[1, 2, 4, 24])
    hourly_df = add_volatility(hourly_df, windows=[24, 72, 168])
    hourly_df = add_rolling_stats(hourly_df, windows=[24, 72, 168])

    # Step 4: Add forward returns (targets)
    print("Adding target variables...")
    hourly_df = add_forward_returns(hourly_df, horizons=[1, 4, 24])

    # Step 5: Merge daily features into hourly
    print("Merging multi-timeframe features...")
    daily_features_to_merge = [
        'sma_50', 'sma_200', 'ema_50', 'ema_200',
        'rsi_14', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_width',
        'volatility_20', 'rolling_mean_20'
    ]

    # Filter to only existing columns
    daily_features_to_merge = [
        col for col in daily_features_to_merge if col in daily_df.columns
    ]

    hourly_df = merge_timeframes(hourly_df, daily_df, daily_features_to_merge)

    # Step 6: Add time-based features
    print("Adding time features...")
    hourly_df = add_time_features(hourly_df)

    # Step 7: Clean the dataframe
    print("Cleaning data...")
    initial_rows = len(hourly_df)
    hourly_df = clean_dataframe(hourly_df)
    final_rows = len(hourly_df)
    print(f"  Removed {initial_rows - final_rows} rows with NaN/Inf")

    # Step 8: Save to parquet
    output_path = PROCESSED_FEATURES_DIR / f"{ticker}_1h.parquet"
    print(f"Saving to {output_path}...")
    write_parquet(hourly_df, output_path)

    print(f"[SUCCESS] Features built for {ticker}")
    print(f"  Output: {len(hourly_df)} rows, {len(hourly_df.columns)} columns")
    print(f"  File: {output_path}")

    return hourly_df


def get_feature_summary(ticker: str) -> dict:
    """
    Get summary statistics for a ticker's feature set.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with summary info
    """
    output_path = PROCESSED_FEATURES_DIR / f"{ticker}_1h.parquet"

    if not output_path.exists():
        return {"error": "Features not built yet"}

    df = read_parquet(output_path)

    if df is None:
        return {"error": "Failed to read features"}

    # Count feature types
    indicator_cols = [col for col in df.columns if any(
        x in col for x in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'stoch']
    )]

    return_cols = [col for col in df.columns if 'return' in col]
    vol_cols = [col for col in df.columns if 'vol' in col]
    rolling_cols = [col for col in df.columns if 'rolling' in col]
    target_cols = [col for col in df.columns if 'target' in col]

    return {
        "ticker": ticker,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "date_range": f"{df.index.min()} to {df.index.max()}",
        "indicators": len(indicator_cols),
        "returns": len(return_cols),
        "volatility": len(vol_cols),
        "rolling_stats": len(rolling_cols),
        "targets": len(target_cols),
    }
