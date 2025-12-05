"""
Technical indicators for financial data.
All functions are vectorized using pandas/NumPy for production performance.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average.

    Args:
        series: Price series (typically close price)
        window: Rolling window size

    Returns:
        SMA values
    """
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        series: Price series (typically close price)
        window: Rolling window size

    Returns:
        EMA values
    """
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Args:
        series: Price series (typically close price)
        window: RSI period (default: 14)

    Returns:
        RSI values (0-100)
    """
    # Calculate price changes
    delta = series.diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Calculate average gain and loss
    avg_gain = gain.ewm(span=window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(span=window, adjust=False, min_periods=window).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence.

    Args:
        series: Price series (typically close price)
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    # Calculate MACD line
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow

    # Calculate signal line
    signal_line = ema(macd_line, signal)

    # Calculate histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Args:
        series: Price series (typically close price)
        window: Rolling window size (default: 20)
        num_std: Number of standard deviations (default: 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle_band = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()

    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)

    return upper_band, middle_band, lower_band


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    Average True Range.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: ATR period (default: 14)

    Returns:
        ATR values
    """
    # Calculate True Range components
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()

    # True Range is the maximum of the three
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR is the moving average of True Range
    atr_values = true_range.ewm(span=window, adjust=False, min_periods=window).mean()

    return atr_values


def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (%K and %D).

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Lookback period (default: 14)
        smooth_k: %K smoothing period (default: 3)
        smooth_d: %D smoothing period (default: 3)

    Returns:
        Tuple of (%K, %D)
    """
    # Calculate lowest low and highest high
    lowest_low = low.rolling(window=window, min_periods=window).min()
    highest_high = high.rolling(window=window, min_periods=window).max()

    # Calculate %K (raw stochastic)
    k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # Smooth %K
    k = k_raw.rolling(window=smooth_k, min_periods=smooth_k).mean()

    # Calculate %D (signal line)
    d = k.rolling(window=smooth_d, min_periods=smooth_d).mean()

    return k, d


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a dataframe.

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)

    Returns:
        DataFrame with added indicator columns
    """
    result = df.copy()

    # Moving Averages
    for window in [10, 20, 50, 200]:
        result[f'sma_{window}'] = sma(result['close'], window)
        result[f'ema_{window}'] = ema(result['close'], window)

    # RSI
    result['rsi_14'] = rsi(result['close'], 14)

    # MACD
    macd_line, signal_line, histogram = macd(result['close'])
    result['macd'] = macd_line
    result['macd_signal'] = signal_line
    result['macd_hist'] = histogram

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = bollinger_bands(result['close'])
    result['bb_upper'] = bb_upper
    result['bb_mid'] = bb_mid
    result['bb_lower'] = bb_lower
    result['bb_width'] = (bb_upper - bb_lower) / bb_mid

    # ATR
    result['atr_14'] = atr(result['high'], result['low'], result['close'])

    # Stochastic
    stoch_k, stoch_d = stochastic_oscillator(
        result['high'], result['low'], result['close']
    )
    result['stoch_k'] = stoch_k
    result['stoch_d'] = stoch_d

    # Price-based features
    result['price_range'] = result['high'] - result['low']
    result['price_change'] = result['close'] - result['open']
    result['price_change_pct'] = result['price_change'] / result['open']

    # Volume features
    result['volume_sma_20'] = sma(result['volume'], 20)
    result['volume_ratio'] = result['volume'] / result['volume_sma_20']

    return result
