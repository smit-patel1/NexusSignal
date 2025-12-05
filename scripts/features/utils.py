"""
Utility functions for feature engineering.
Handles data preparation, transformations, and cleaning.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has a datetime index.

    Args:
        df: Input dataframe

    Returns:
        DataFrame with datetime index
    """
    result = df.copy()

    if 'timestamp' in result.columns:
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        result = result.set_index('timestamp')
    elif not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)

    # Sort by index
    result = result.sort_index()

    # Remove duplicates
    result = result[~result.index.duplicated(keep='last')]

    return result


def add_lagged_returns(
    df: pd.DataFrame,
    horizons: List[int] = [1, 2, 4, 24]
) -> pd.DataFrame:
    """
    Add lagged returns (past returns) to the dataframe.

    Args:
        df: DataFrame with 'close' column
        horizons: List of lag periods

    Returns:
        DataFrame with lagged return columns
    """
    result = df.copy()

    for horizon in horizons:
        result[f'return_lag_{horizon}'] = result['close'].pct_change(horizon).shift(1)

    return result


def add_forward_returns(
    df: pd.DataFrame,
    horizons: List[int] = [1, 4, 24]
) -> pd.DataFrame:
    """
    Add forward returns (future returns) to the dataframe.
    These are typically used as targets for prediction.

    Args:
        df: DataFrame with 'close' column
        horizons: List of forward periods

    Returns:
        DataFrame with forward return columns
    """
    result = df.copy()

    for horizon in horizons:
        result[f'target_{horizon}h_return'] = result['close'].pct_change(horizon).shift(-horizon)

    return result


def add_volatility(
    df: pd.DataFrame,
    windows: List[int] = [24, 72, 168]
) -> pd.DataFrame:
    """
    Add rolling volatility features.

    Args:
        df: DataFrame with 'close' column
        windows: List of rolling window sizes

    Returns:
        DataFrame with volatility columns
    """
    result = df.copy()

    # Calculate returns
    returns = result['close'].pct_change()

    for window in windows:
        # Standard deviation of returns
        result[f'volatility_{window}'] = returns.rolling(
            window=window, min_periods=window
        ).std()

        # Realized volatility (sqrt of sum of squared returns)
        result[f'realized_vol_{window}'] = np.sqrt(
            (returns ** 2).rolling(window=window, min_periods=window).sum()
        )

    return result


def add_rolling_stats(
    df: pd.DataFrame,
    windows: List[int] = [24, 72, 168]
) -> pd.DataFrame:
    """
    Add rolling statistical features.

    Args:
        df: DataFrame with 'close' column
        windows: List of rolling window sizes

    Returns:
        DataFrame with rolling stats columns
    """
    result = df.copy()

    for window in windows:
        # Rolling mean
        result[f'rolling_mean_{window}'] = result['close'].rolling(
            window=window, min_periods=window
        ).mean()

        # Rolling std
        result[f'rolling_std_{window}'] = result['close'].rolling(
            window=window, min_periods=window
        ).std()

        # Rolling min/max
        result[f'rolling_min_{window}'] = result['close'].rolling(
            window=window, min_periods=window
        ).min()

        result[f'rolling_max_{window}'] = result['close'].rolling(
            window=window, min_periods=window
        ).max()

        # Distance from rolling max (drawdown indicator)
        result[f'pct_from_max_{window}'] = (
            result['close'] - result[f'rolling_max_{window}']
        ) / result[f'rolling_max_{window}']

    return result


def zscore(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize columns using z-score standardization.

    Args:
        df: Input dataframe
        columns: List of columns to normalize (None = all numeric columns)

    Returns:
        DataFrame with normalized columns
    """
    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in result.columns:
            mean = result[col].mean()
            std = result[col].std()

            if std > 0:
                result[col] = (result[col] - mean) / std

    return result


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe by handling NaNs, infs, and duplicates.

    Args:
        df: Input dataframe

    Returns:
        Cleaned dataframe
    """
    result = df.copy()

    # Replace infinities with NaN
    result = result.replace([np.inf, -np.inf], np.nan)

    # Remove rows where ALL values are NaN
    result = result.dropna(how='all')

    # For remaining NaN values, forward fill then backward fill
    result = result.ffill().bfill()

    # If any NaNs remain (e.g., at the start), drop those rows
    result = result.dropna()

    # Remove duplicate index values
    result = result[~result.index.duplicated(keep='last')]

    return result


def merge_timeframes(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    daily_features: List[str]
) -> pd.DataFrame:
    """
    Merge daily features into hourly dataframe.
    Daily features are forward-filled to match hourly timestamps.

    Args:
        hourly_df: Hourly dataframe (target)
        daily_df: Daily dataframe (source)
        daily_features: List of columns to merge from daily

    Returns:
        Hourly dataframe with daily features
    """
    result = hourly_df.copy()

    # Ensure both have datetime index
    if not isinstance(result.index, pd.DatetimeIndex):
        result = ensure_datetime_index(result)
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        daily_df = ensure_datetime_index(daily_df)

    # Normalize timezones - convert both to UTC then remove timezone
    # This ensures compatibility between timezone-aware and timezone-naive data
    if result.index.tz is not None:
        result.index = result.index.tz_convert('UTC').tz_localize(None)

    if daily_df.index.tz is not None:
        daily_df = daily_df.copy()  # Avoid modifying original
        daily_df.index = daily_df.index.tz_convert('UTC').tz_localize(None)

    # Extract daily features
    daily_subset = daily_df[daily_features].copy()

    # Add suffix to avoid column name conflicts
    daily_subset.columns = [f'{col}_daily' for col in daily_subset.columns]

    # Merge using asof (forward fill)
    result = pd.merge_asof(
        result.reset_index(),
        daily_subset.reset_index(),
        left_on='timestamp' if 'timestamp' in result.columns else result.index.name,
        right_on='timestamp' if 'timestamp' in daily_subset.columns else daily_subset.index.name,
        direction='backward'
    )

    # Restore index
    if 'timestamp' in result.columns:
        result = result.set_index('timestamp')

    return result


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features from the datetime index.

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with time features
    """
    result = df.copy()

    if isinstance(result.index, pd.DatetimeIndex):
        result['hour'] = result.index.hour
        result['day_of_week'] = result.index.dayofweek
        result['day_of_month'] = result.index.day
        result['month'] = result.index.month
        result['quarter'] = result.index.quarter

        # Cyclical encoding for hour
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)

        # Cyclical encoding for day of week
        result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)

    return result
