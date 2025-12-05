"""
Advanced alpha feature engineering for return prediction.

Includes:
- Rolling volatility, skewness, kurtosis
- Technical indicators (RSI, MACD, ADX)
- Market beta and correlation features
- Time-of-day and calendar features
- Market regime detection
- Cross-asset features
"""

from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd


def add_rolling_moments(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: list = [10, 20, 60],
) -> pd.DataFrame:
    """
    Add rolling statistical moments (volatility, skewness, kurtosis).

    Args:
        df: Input DataFrame
        price_col: Price column name
        windows: Rolling window sizes

    Returns:
        DataFrame with moment features
    """
    df_out = df.copy()

    # Compute returns first
    returns = df_out[price_col].pct_change()

    for window in windows:
        # Volatility (realized volatility)
        df_out[f"volatility_{window}"] = returns.rolling(window).std() * np.sqrt(window)

        # Skewness
        df_out[f"skew_{window}"] = returns.rolling(window).skew()

        # Kurtosis
        df_out[f"kurtosis_{window}"] = returns.rolling(window).kurt()

    return df_out


def add_rsi(
    df: pd.DataFrame,
    price_col: str = "close",
    periods: list = [14, 28],
) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI).

    Args:
        df: Input DataFrame
        price_col: Price column name
        periods: RSI periods

    Returns:
        DataFrame with RSI features
    """
    df_out = df.copy()

    delta = df_out[price_col].diff()

    for period in periods:
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        df_out[f"rsi_{period}"] = rsi

    return df_out


def add_macd(
    df: pd.DataFrame,
    price_col: str = "close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Add MACD (Moving Average Convergence Divergence).

    Args:
        df: Input DataFrame
        price_col: Price column name
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        DataFrame with MACD features
    """
    df_out = df.copy()

    ema_fast = df_out[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df_out[price_col].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    df_out["macd"] = macd_line
    df_out["macd_signal"] = signal_line
    df_out["macd_histogram"] = macd_histogram

    return df_out


def add_adx(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    period: int = 14,
) -> pd.DataFrame:
    """
    Add Average Directional Index (ADX) for trend strength.

    Args:
        df: Input DataFrame
        high_col: High price column
        low_col: Low price column
        close_col: Close price column
        period: ADX period

    Returns:
        DataFrame with ADX features
    """
    df_out = df.copy()

    # Calculate True Range
    high_low = df_out[high_col] - df_out[low_col]
    high_close = np.abs(df_out[high_col] - df_out[close_col].shift())
    low_close = np.abs(df_out[low_col] - df_out[close_col].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate Directional Movement
    high_diff = df_out[high_col].diff()
    low_diff = -df_out[low_col].diff()

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    # Smooth with Wilder's smoothing
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df_out.index).ewm(
        alpha=1 / period, adjust=False
    ).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df_out.index).ewm(
        alpha=1 / period, adjust=False
    ).mean() / atr

    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    df_out["adx"] = adx
    df_out["plus_di"] = plus_di
    df_out["minus_di"] = minus_di

    return df_out


def add_rolling_beta(
    df: pd.DataFrame,
    price_col: str = "close",
    market_returns: Optional[pd.Series] = None,
    windows: list = [20, 60],
) -> pd.DataFrame:
    """
    Add rolling beta vs market (SPY).

    Args:
        df: Input DataFrame
        price_col: Price column name
        market_returns: Market return series (aligned with df)
        windows: Rolling window sizes

    Returns:
        DataFrame with beta features
    """
    df_out = df.copy()

    if market_returns is None:
        warnings.warn("No market returns provided, skipping beta calculation")
        return df_out

    # Compute asset returns
    asset_returns = df_out[price_col].pct_change()

    # Align market returns with asset
    market_returns_aligned = market_returns.reindex(df_out.index, method="ffill")

    for window in windows:
        # Rolling covariance and variance
        cov = asset_returns.rolling(window).cov(market_returns_aligned)
        var = market_returns_aligned.rolling(window).var()

        beta = cov / var.replace(0, 1e-10)
        df_out[f"beta_{window}"] = beta

        # Also add rolling correlation
        corr = asset_returns.rolling(window).corr(market_returns_aligned)
        df_out[f"corr_market_{window}"] = corr

    return df_out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-of-day and calendar features.

    Args:
        df: Input DataFrame with DatetimeIndex

    Returns:
        DataFrame with time features
    """
    df_out = df.copy()

    if not isinstance(df_out.index, pd.DatetimeIndex):
        warnings.warn("Index is not DatetimeIndex, skipping time features")
        return df_out

    # Hour of day (cyclical encoding)
    hour = df_out.index.hour
    df_out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df_out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Day of week (cyclical encoding)
    day_of_week = df_out.index.dayofweek
    df_out["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df_out["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    # Day of month (cyclical encoding)
    day_of_month = df_out.index.day
    df_out["dom_sin"] = np.sin(2 * np.pi * day_of_month / 31)
    df_out["dom_cos"] = np.cos(2 * np.pi * day_of_month / 31)

    # Month of year (cyclical encoding)
    month = df_out.index.month
    df_out["month_sin"] = np.sin(2 * np.pi * month / 12)
    df_out["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Binary features
    df_out["is_market_open"] = (
        (hour >= 9) & (hour < 16) & (day_of_week < 5)
    ).astype(int)
    df_out["is_morning"] = ((hour >= 9) & (hour < 12)).astype(int)
    df_out["is_afternoon"] = ((hour >= 12) & (hour < 16)).astype(int)
    df_out["is_month_start"] = (day_of_month <= 5).astype(int)
    df_out["is_month_end"] = (day_of_month >= 26).astype(int)

    return df_out


def detect_market_regime(
    df: pd.DataFrame,
    price_col: str = "close",
    volatility_window: int = 20,
    trend_window: int = 50,
) -> pd.DataFrame:
    """
    Detect market regime (trending vs mean-reverting, high vs low vol).

    Args:
        df: Input DataFrame
        price_col: Price column name
        volatility_window: Window for volatility calculation
        trend_window: Window for trend detection

    Returns:
        DataFrame with regime features
    """
    df_out = df.copy()

    returns = df_out[price_col].pct_change()

    # Volatility regime
    rolling_vol = returns.rolling(volatility_window).std()
    vol_median = rolling_vol.median()
    df_out["high_vol_regime"] = (rolling_vol > vol_median).astype(int)

    # Trend regime (using ADX-like logic)
    sma_short = df_out[price_col].rolling(20).mean()
    sma_long = df_out[price_col].rolling(trend_window).mean()

    df_out["trending_regime"] = (
        (sma_short > sma_long).astype(int) * 2 - 1
    )  # +1 uptrend, -1 downtrend

    # Mean reversion indicator (Hurst exponent approximation)
    # Simplified: high autocorrelation = trending, low = mean-reverting
    autocorr = returns.rolling(volatility_window).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    df_out["mean_reversion_strength"] = -autocorr  # Negative autocorr = mean reversion

    # Volume regime
    if "volume" in df_out.columns:
        rolling_vol_median = df_out["volume"].rolling(volatility_window).median()
        df_out["high_volume_regime"] = (
            df_out["volume"] > rolling_vol_median
        ).astype(int)

    return df_out


def add_orderbook_features(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Add microstructure and orderbook-inspired features.

    Args:
        df: Input DataFrame
        high_col: High price column
        low_col: Low price column
        close_col: Close price column
        volume_col: Volume column

    Returns:
        DataFrame with microstructure features
    """
    df_out = df.copy()

    # Spread (high - low) normalized by close
    df_out["spread_pct"] = (
        (df_out[high_col] - df_out[low_col]) / df_out[close_col] * 100
    )

    # Amihud illiquidity measure
    if volume_col in df_out.columns:
        returns_abs = df_out[close_col].pct_change().abs()
        df_out["illiquidity"] = returns_abs / (
            df_out[volume_col].replace(0, 1e-10) + 1e-10
        )

        # Volume-weighted price momentum
        vwap = (df_out[close_col] * df_out[volume_col]).rolling(20).sum() / df_out[
            volume_col
        ].rolling(20).sum()
        df_out["vwap_distance"] = (df_out[close_col] - vwap) / vwap

    # Close position within bar (where did price close relative to high/low?)
    bar_range = df_out[high_col] - df_out[low_col]
    df_out["close_position"] = (df_out[close_col] - df_out[low_col]) / bar_range.replace(
        0, 1e-10
    )

    return df_out


def add_lagged_features(
    df: pd.DataFrame, feature_cols: list, lags: list = [1, 2, 3, 5]
) -> pd.DataFrame:
    """
    Add lagged versions of features for temporal context.

    Args:
        df: Input DataFrame
        feature_cols: Columns to lag
        lags: Lag periods

    Returns:
        DataFrame with lagged features
    """
    df_out = df.copy()

    for col in feature_cols:
        if col in df_out.columns:
            for lag in lags:
                df_out[f"{col}_lag{lag}"] = df_out[col].shift(lag)

    return df_out


def add_interaction_features(
    df: pd.DataFrame,
    feature_pairs: list = None,
) -> pd.DataFrame:
    """
    Add interaction features (products, ratios).

    Args:
        df: Input DataFrame
        feature_pairs: List of (col1, col2) tuples to create interactions

    Returns:
        DataFrame with interaction features
    """
    df_out = df.copy()

    if feature_pairs is None:
        # Default interactions
        feature_pairs = [
            ("volatility_20", "volume"),
            ("rsi_14", "macd"),
            ("adx", "volatility_20"),
        ]

    for col1, col2 in feature_pairs:
        if col1 in df_out.columns and col2 in df_out.columns:
            # Product
            df_out[f"{col1}_x_{col2}"] = df_out[col1] * df_out[col2]

            # Ratio (avoid division by zero)
            df_out[f"{col1}_div_{col2}"] = df_out[col1] / df_out[col2].replace(
                0, 1e-10
            )

    return df_out


def build_alpha_features(
    df: pd.DataFrame,
    market_returns: Optional[pd.Series] = None,
    include_interactions: bool = True,
) -> pd.DataFrame:
    """
    Build comprehensive alpha feature set.

    Args:
        df: Input DataFrame with OHLCV data
        market_returns: Market returns for beta calculation
        include_interactions: Whether to add interaction features

    Returns:
        DataFrame with all alpha features
    """
    df_alpha = df.copy()

    # Rolling moments
    df_alpha = add_rolling_moments(df_alpha)

    # Technical indicators
    df_alpha = add_rsi(df_alpha)
    df_alpha = add_macd(df_alpha)

    if all(col in df_alpha.columns for col in ["high", "low", "close"]):
        df_alpha = add_adx(df_alpha)

    # Market beta
    if market_returns is not None:
        df_alpha = add_rolling_beta(df_alpha, market_returns=market_returns)

    # Time features
    df_alpha = add_time_features(df_alpha)

    # Market regime
    df_alpha = detect_market_regime(df_alpha)

    # Microstructure
    if all(col in df_alpha.columns for col in ["high", "low", "close"]):
        df_alpha = add_orderbook_features(df_alpha)

    # Lagged features (lag key indicators)
    key_features = ["close", "volume", "volatility_20", "rsi_14"]
    available_features = [f for f in key_features if f in df_alpha.columns]
    df_alpha = add_lagged_features(df_alpha, available_features)

    # Interactions
    if include_interactions:
        df_alpha = add_interaction_features(df_alpha)

    return df_alpha
