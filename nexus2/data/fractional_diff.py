"""
Fractional Differencing for Stationarity

Implements Fixed-Width Window Fractional Differencing (FFD) from
"Advances in Financial Machine Learning" by Lopez de Prado.

Key concepts:
- Traditional differencing (d=1) removes too much memory/signal
- No differencing (d=0) is non-stationary
- Fractional differencing (0 < d < 1) balances stationarity vs memory

FFD uses a fixed window of weights, making it computationally tractable
and preventing information leakage from future observations.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.special import gamma


def get_weights_ffd(d: float, threshold: float = 1e-5, max_lag: int = 100) -> np.ndarray:
    """
    Compute weights for Fixed-Width Window Fractional Differencing.
    
    The fractional differencing operator is:
    (1 - B)^d = sum_{k=0}^{inf} w_k * B^k
    
    where w_k = (-1)^k * C(d, k) and C(d, k) is the binomial coefficient.
    
    Using the recursive formula:
    w_k = -w_{k-1} * (d - k + 1) / k
    
    Args:
        d: Differencing order (0 < d < 1 for fractional)
        threshold: Minimum absolute weight to include
        max_lag: Maximum number of lags (prevents infinite series)
    
    Returns:
        Array of weights [w_0, w_1, ..., w_k] where |w_k| >= threshold
        
    Mathematical derivation:
        w_0 = 1
        w_k = w_{k-1} * (k - 1 - d) / k  [recursive formula]
        
    The weights decay geometrically, allowing truncation at threshold.
    """
    weights = [1.0]
    k = 1
    
    while k < max_lag:
        # Recursive weight computation
        w_k = -weights[-1] * (d - k + 1) / k
        
        if abs(w_k) < threshold:
            break
            
        weights.append(w_k)
        k += 1
    
    return np.array(weights[::-1])  # Reverse for convolution


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
    max_lag: int = 100
) -> pd.Series:
    """
    Apply Fixed-Width Window Fractional Differencing to a series.
    
    FFD formula:
    X_t^d = sum_{k=0}^{K} w_k * X_{t-k}
    
    where K is determined by the weight threshold.
    
    Args:
        series: Input time series (prices or features)
        d: Fractional differencing order
        threshold: Minimum weight threshold
        max_lag: Maximum lag for weight computation
    
    Returns:
        Fractionally differenced series (shorter by len(weights)-1)
        
    Note:
        - d=0: No differencing (original series)
        - d=1: First difference (returns)
        - 0<d<1: Fractional differencing (balanced)
        - d>1: Over-differencing (rarely used)
    """
    weights = get_weights_ffd(d, threshold, max_lag)
    width = len(weights)
    
    # Apply convolution (dot product of weights with lagged values)
    result = {}
    
    for i in range(width - 1, len(series)):
        # Get window of values
        window = series.iloc[i - width + 1:i + 1].values
        
        if len(window) == width and not np.any(np.isnan(window)):
            # Compute weighted sum
            result[series.index[i]] = np.dot(weights, window)
    
    return pd.Series(result, dtype=float)


def adf_test(series: pd.Series, max_lag: Optional[int] = None) -> Tuple[float, float]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Args:
        series: Time series to test
        max_lag: Maximum lag for ADF (None = automatic)
    
    Returns:
        Tuple of (ADF statistic, p-value)
        
    Interpretation:
        - p-value < 0.05: Reject null hypothesis → series is stationary
        - p-value >= 0.05: Cannot reject → series may be non-stationary
    """
    series_clean = series.dropna()
    
    if len(series_clean) < 20:
        return np.nan, 1.0
    
    try:
        result = adfuller(series_clean, maxlag=max_lag, autolag='AIC')
        return result[0], result[1]
    except Exception:
        return np.nan, 1.0


def find_optimal_d(
    series: pd.Series,
    d_min: float = 0.0,
    d_max: float = 1.0,
    d_step: float = 0.01,
    adf_threshold: float = 0.05,
    weight_threshold: float = 1e-5,
    max_lag: int = 100
) -> Dict[str, float]:
    """
    Find minimum d that achieves stationarity while preserving maximum memory.
    
    Strategy:
    1. Start from d_min and increase by d_step
    2. At each d, apply FFD and run ADF test
    3. Return first d where ADF p-value < threshold
    
    Args:
        series: Input time series
        d_min: Minimum d to search
        d_max: Maximum d to search  
        d_step: Step size for d search
        adf_threshold: ADF p-value threshold for stationarity
        weight_threshold: Minimum weight for FFD
        max_lag: Maximum lag for FFD weights
    
    Returns:
        Dictionary with:
        - 'optimal_d': Minimum d achieving stationarity
        - 'adf_stat': ADF statistic at optimal d
        - 'adf_pvalue': ADF p-value at optimal d
        - 'is_stationary': Whether stationarity was achieved
        - 'search_results': DataFrame with all tested d values
    """
    results = []
    optimal_d = None
    optimal_stat = None
    optimal_pvalue = None
    
    d_values = np.arange(d_min, d_max + d_step, d_step)
    
    for d in d_values:
        if d == 0:
            diff_series = series
        else:
            diff_series = frac_diff_ffd(series, d, weight_threshold, max_lag)
        
        if len(diff_series) < 50:
            continue
            
        adf_stat, adf_pvalue = adf_test(diff_series)
        
        results.append({
            'd': d,
            'adf_stat': adf_stat,
            'adf_pvalue': adf_pvalue,
            'n_samples': len(diff_series),
            'is_stationary': adf_pvalue < adf_threshold if not np.isnan(adf_pvalue) else False
        })
        
        # First d achieving stationarity
        if optimal_d is None and adf_pvalue < adf_threshold:
            optimal_d = d
            optimal_stat = adf_stat
            optimal_pvalue = adf_pvalue
    
    return {
        'optimal_d': optimal_d if optimal_d is not None else d_max,
        'adf_stat': optimal_stat,
        'adf_pvalue': optimal_pvalue,
        'is_stationary': optimal_d is not None,
        'search_results': pd.DataFrame(results)
    }


def apply_fractional_diff(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    d_values: Optional[Dict[str, float]] = None,
    auto_find_d: bool = True,
    config: Optional[dict] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Apply fractional differencing to multiple columns.
    
    Args:
        df: Input DataFrame with time series columns
        columns: Columns to difference (None = all numeric)
        d_values: Pre-specified d values per column
        auto_find_d: Whether to auto-find optimal d
        config: Configuration dict (d_min, d_max, etc.)
    
    Returns:
        Tuple of (differenced DataFrame, dict of d values used)
        
    Example:
        >>> df_diff, d_used = apply_fractional_diff(df, columns=['close', 'volume'])
        >>> print(d_used)
        {'close': 0.35, 'volume': 0.42}
    """
    if config is None:
        config = {
            'd_min': 0.0,
            'd_max': 1.0,
            'd_step': 0.05,
            'adf_threshold': 0.05,
            'weight_threshold': 1e-5,
            'max_lag': 100
        }
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if d_values is None:
        d_values = {}
    
    result_df = pd.DataFrame(index=df.index)
    final_d_values = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        series = df[col].dropna()
        
        # Get d value
        if col in d_values:
            d = d_values[col]
        elif auto_find_d:
            search_result = find_optimal_d(
                series,
                d_min=config['d_min'],
                d_max=config['d_max'],
                d_step=config['d_step'],
                adf_threshold=config['adf_threshold'],
                weight_threshold=config['weight_threshold'],
                max_lag=config['max_lag']
            )
            d = search_result['optimal_d']
        else:
            d = 0.5  # Default fractional order
        
        final_d_values[col] = d
        
        # Apply differencing
        if d > 0:
            diff_series = frac_diff_ffd(
                series, d,
                threshold=config['weight_threshold'],
                max_lag=config['max_lag']
            )
            result_df[f"{col}_ffd"] = diff_series
        else:
            result_df[f"{col}_ffd"] = series
    
    return result_df, final_d_values


def compute_memory_correlation(
    original: pd.Series,
    differenced: pd.Series
) -> float:
    """
    Compute correlation between original and differenced series.
    
    Higher correlation = more memory preserved.
    
    Args:
        original: Original time series
        differenced: Fractionally differenced series
    
    Returns:
        Pearson correlation coefficient
    """
    # Align series
    common_idx = original.index.intersection(differenced.index)
    
    if len(common_idx) < 10:
        return np.nan
    
    orig = original.loc[common_idx]
    diff = differenced.loc[common_idx]
    
    return np.corrcoef(orig, diff)[0, 1]

