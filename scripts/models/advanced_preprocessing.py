"""
Advanced preprocessing utilities for return prediction.

Includes:
- Log return transformations
- Outlier detection and clipping
- Rolling normalization
- Stationarity transformations
- Data quality filtering
- Target preprocessing and denoising
"""

from typing import Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler


def compute_log_returns(
    prices: pd.Series, periods: int = 1, clip_std: float = 5.0
) -> pd.Series:
    """
    Compute log returns with outlier clipping.

    Args:
        prices: Price series
        periods: Number of periods for return calculation
        clip_std: Number of standard deviations for clipping

    Returns:
        Clipped log returns
    """
    log_returns = np.log(prices / prices.shift(periods))

    # Clip extreme outliers
    mean = log_returns.mean()
    std = log_returns.std()
    lower = mean - clip_std * std
    upper = mean + clip_std * std

    return log_returns.clip(lower, upper)


def rolling_normalize(
    series: pd.Series, window: int = 60, method: str = "zscore"
) -> pd.Series:
    """
    Apply rolling normalization to a series.

    Args:
        series: Input series
        window: Rolling window size
        method: 'zscore' or 'minmax'

    Returns:
        Normalized series
    """
    if method == "zscore":
        rolling_mean = series.rolling(window, min_periods=1).mean()
        rolling_std = series.rolling(window, min_periods=1).std()
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1e-8)
        return (series - rolling_mean) / rolling_std

    elif method == "minmax":
        rolling_min = series.rolling(window, min_periods=1).min()
        rolling_max = series.rolling(window, min_periods=1).max()
        # Avoid division by zero
        denominator = rolling_max - rolling_min
        denominator = denominator.replace(0, 1e-8)
        return (series - rolling_min) / denominator

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def make_stationary(df: pd.DataFrame, diff_cols: list = None) -> pd.DataFrame:
    """
    Apply differencing to make features stationary.

    Args:
        df: Input DataFrame
        diff_cols: Columns to difference (if None, difference all numeric columns)

    Returns:
        DataFrame with differenced features
    """
    df_stationary = df.copy()

    if diff_cols is None:
        diff_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in diff_cols:
        if col in df_stationary.columns:
            df_stationary[f"{col}_diff"] = df_stationary[col].diff()

    return df_stationary


def clip_outliers_iqr(
    series: pd.Series, iqr_multiplier: float = 3.0
) -> pd.Series:
    """
    Clip outliers using IQR method.

    Args:
        series: Input series
        iqr_multiplier: Multiplier for IQR (default 3.0 = extreme outliers)

    Returns:
        Series with outliers clipped
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    return series.clip(lower, upper)


def filter_bad_data(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    min_volume_percentile: float = 1.0,
    max_price_jump_pct: float = 20.0,
) -> pd.DataFrame:
    """
    Filter out bad data points based on quality checks.

    Args:
        df: Input DataFrame with OHLCV data
        price_col: Name of price column
        volume_col: Name of volume column
        min_volume_percentile: Minimum volume percentile to keep
        max_price_jump_pct: Maximum allowed price jump percentage

    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()

    # Remove zero/negative prices
    df_filtered = df_filtered[df_filtered[price_col] > 0]

    # Remove extremely low volume periods
    if volume_col in df_filtered.columns:
        volume_threshold = df_filtered[volume_col].quantile(
            min_volume_percentile / 100
        )
        df_filtered = df_filtered[df_filtered[volume_col] >= volume_threshold]

    # Remove extreme price jumps (likely bad ticks)
    price_returns = df_filtered[price_col].pct_change().abs()
    max_jump = max_price_jump_pct / 100
    df_filtered = df_filtered[price_returns <= max_jump]

    # Remove NaN rows
    df_filtered = df_filtered.dropna(subset=[price_col])

    return df_filtered


def preprocess_targets(
    targets: pd.Series,
    method: str = "log",
    normalize: bool = True,
    denoise: bool = False,
    denoise_window: int = 5,
) -> Tuple[pd.Series, Optional[dict]]:
    """
    Preprocess target returns for better learning.

    Args:
        targets: Raw target returns
        method: 'log', 'arctanh', or 'winsorize'
        normalize: Whether to normalize targets
        denoise: Whether to apply Savitzky-Golay filtering
        denoise_window: Window size for denoising (must be odd)

    Returns:
        Preprocessed targets and transformation parameters
    """
    processed = targets.copy()
    transform_params = {"method": method}

    # Apply transformation
    if method == "log":
        # Log transform: sign(x) * log(1 + |x|)
        processed = np.sign(processed) * np.log1p(np.abs(processed))
    elif method == "arctanh":
        # Inverse hyperbolic tangent (bounded transformation)
        # Clip to avoid overflow
        processed = processed.clip(-0.99, 0.99)
        processed = np.arctanh(processed)
    elif method == "winsorize":
        # Winsorize to 1st and 99th percentiles
        lower = processed.quantile(0.01)
        upper = processed.quantile(0.99)
        processed = processed.clip(lower, upper)
        transform_params["lower"] = lower
        transform_params["upper"] = upper

    # Normalize
    if normalize:
        mean = processed.mean()
        std = processed.std()
        if std > 0:
            processed = (processed - mean) / std
            transform_params["mean"] = mean
            transform_params["std"] = std

    # Denoise
    if denoise and len(processed) > denoise_window:
        # Ensure window is odd
        if denoise_window % 2 == 0:
            denoise_window += 1

        # Apply Savitzky-Golay filter
        try:
            valid_indices = ~processed.isna()
            if valid_indices.sum() > denoise_window:
                processed_values = processed.values.copy()
                valid_values = processed_values[valid_indices]

                # Apply filter
                smoothed = savgol_filter(
                    valid_values, denoise_window, polyorder=2, mode="nearest"
                )

                # Put back into array
                processed_values[valid_indices] = smoothed
                processed = pd.Series(processed_values, index=processed.index)

                transform_params["denoised"] = True
        except Exception as e:
            warnings.warn(f"Denoising failed: {e}")

    return processed, transform_params


def inverse_preprocess_targets(
    processed: np.ndarray, transform_params: dict
) -> np.ndarray:
    """
    Inverse transform preprocessed targets back to original scale.

    Args:
        processed: Preprocessed target values
        transform_params: Parameters from preprocess_targets

    Returns:
        Targets in original scale
    """
    result = processed.copy()

    # Inverse normalize
    if "mean" in transform_params and "std" in transform_params:
        result = result * transform_params["std"] + transform_params["mean"]

    # Inverse transformation
    method = transform_params["method"]
    if method == "log":
        result = np.sign(result) * np.expm1(np.abs(result))
    elif method == "arctanh":
        result = np.tanh(result)
    elif method == "winsorize":
        # No inverse needed for winsorization
        pass

    return result


def apply_kalman_filter(series: pd.Series, process_variance: float = 1e-5) -> pd.Series:
    """
    Apply simple Kalman filter for denoising time series.

    Args:
        series: Input time series
        process_variance: Process noise variance (lower = smoother)

    Returns:
        Filtered series
    """
    # Simple 1D Kalman filter
    n = len(series)
    filtered = np.zeros(n)

    # Initialize
    x_est = series.iloc[0]  # Initial state estimate
    p_est = 1.0  # Initial error covariance

    # Estimate measurement variance from data
    measurement_variance = series.var() * 0.1

    for i in range(n):
        # Prediction
        x_pred = x_est
        p_pred = p_est + process_variance

        # Update
        if not np.isnan(series.iloc[i]):
            # Kalman gain
            k = p_pred / (p_pred + measurement_variance)

            # Update estimate
            x_est = x_pred + k * (series.iloc[i] - x_pred)

            # Update error covariance
            p_est = (1 - k) * p_pred
        else:
            x_est = x_pred
            p_est = p_pred

        filtered[i] = x_est

    return pd.Series(filtered, index=series.index)


def robust_scale_features(df: pd.DataFrame, exclude_cols: list = None) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    Apply RobustScaler to features (resistant to outliers).

    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude from scaling

    Returns:
        Scaled DataFrame and fitted scaler
    """
    if exclude_cols is None:
        exclude_cols = []

    # Identify numeric columns to scale
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Fit scaler
    scaler = RobustScaler()
    df_scaled = df.copy()

    if scale_cols:
        df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df_scaled, scaler


def check_stationarity(series: pd.Series, max_pvalue: float = 0.05) -> bool:
    """
    Check if series is stationary using Augmented Dickey-Fuller test.

    Args:
        series: Time series to test
        max_pvalue: Maximum p-value for stationarity

    Returns:
        True if stationary, False otherwise
    """
    try:
        from statsmodels.tsa.stattools import adfuller

        # Remove NaNs
        series_clean = series.dropna()

        if len(series_clean) < 10:
            return False

        # Run ADF test
        result = adfuller(series_clean, autolag="AIC")
        p_value = result[1]

        return p_value < max_pvalue

    except Exception as e:
        warnings.warn(f"Stationarity test failed: {e}")
        return False
