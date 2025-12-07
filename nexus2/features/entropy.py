"""
Entropy Features for Financial Time Series

Entropy measures quantify the randomness/complexity of a time series.
Changes in entropy can indicate regime shifts, increased uncertainty,
or structural breaks.

Key measures:
- Approximate Entropy (ApEn): Regularity/predictability measure
- Sample Entropy (SampEn): Improved ApEn without self-matching
- Permutation Entropy: Complexity based on order patterns

Reference: Pincus (1991), Richman & Moorman (2000)
"""

from typing import Optional
import math
import numpy as np
import pandas as pd
from numba import jit


@jit(nopython=True)
def _phi_apen(x: np.ndarray, m: int, r: float) -> float:
    """
    Compute phi(m) for approximate entropy.
    
    Args:
        x: Time series
        m: Embedding dimension
        r: Tolerance (threshold)
    
    Returns:
        phi(m) value
    """
    n = len(x)
    count = 0
    
    for i in range(n - m + 1):
        for j in range(n - m + 1):
            # Check if templates match within tolerance
            match = True
            for k in range(m):
                if abs(x[i + k] - x[j + k]) > r:
                    match = False
                    break
            if match:
                count += 1
    
    if count == 0:
        return 0.0
    
    return np.log(count / (n - m + 1))


def approximate_entropy(
    x: np.ndarray,
    m: int = 2,
    r: Optional[float] = None
) -> float:
    """
    Compute Approximate Entropy (ApEn).
    
    ApEn measures the likelihood that patterns that are close for m
    observations remain close for m+1 observations. Lower ApEn means
    more regularity/predictability.
    
    Formula:
        ApEn(m, r) = phi(m) - phi(m+1)
        
    where phi(m) = (1/(N-m+1)) * sum(log(C_i^m(r)))
    and C_i^m(r) = fraction of m-length patterns within distance r
    
    Args:
        x: Time series (1D array)
        m: Embedding dimension (pattern length)
        r: Tolerance (default: 0.2 * std(x))
    
    Returns:
        Approximate entropy value
        
    Interpretation:
    - Low ApEn (close to 0): Regular, predictable series
    - High ApEn (close to 2): Random, unpredictable series
    
    Note: ApEn is biased because it counts self-matches.
    """
    if len(x) < m + 1:
        return np.nan
    
    if r is None:
        r = 0.2 * np.std(x)
    
    phi_m = _phi_apen(x, m, r)
    phi_m1 = _phi_apen(x, m + 1, r)
    
    return phi_m - phi_m1


@jit(nopython=True)
def _count_matches(x: np.ndarray, m: int, r: float, exclude_self: bool) -> float:
    """
    Count matching templates for sample entropy.
    
    Args:
        x: Time series
        m: Embedding dimension
        r: Tolerance
        exclude_self: Whether to exclude self-matches
    
    Returns:
        Average match count
    """
    n = len(x)
    count = 0
    total = 0
    
    for i in range(n - m):
        for j in range(n - m):
            if exclude_self and i == j:
                continue
            
            # Check if templates match within tolerance
            match = True
            for k in range(m):
                if abs(x[i + k] - x[j + k]) > r:
                    match = False
                    break
            if match:
                count += 1
            total += 1
    
    if total == 0:
        return 0.0
    
    return count / total


def sample_entropy(
    x: np.ndarray,
    m: int = 2,
    r: Optional[float] = None
) -> float:
    """
    Compute Sample Entropy (SampEn).
    
    SampEn is an improved version of ApEn that:
    - Does not count self-matches (removes bias)
    - Is less dependent on data length
    - Has better consistency
    
    Formula:
        SampEn(m, r) = -log(A / B)
        
    where A = # of matching (m+1)-templates
          B = # of matching m-templates
          (excluding self-matches)
    
    Args:
        x: Time series (1D array)
        m: Embedding dimension
        r: Tolerance (default: 0.2 * std(x))
    
    Returns:
        Sample entropy value
        
    Interpretation:
    - Low SampEn (< 0.5): Highly regular/predictable
    - Medium SampEn (0.5-1.5): Moderate complexity
    - High SampEn (> 1.5): High randomness
    """
    if len(x) < m + 2:
        return np.nan
    
    if r is None:
        r = 0.2 * np.std(x)
    
    B = _count_matches(x, m, r, exclude_self=True)
    A = _count_matches(x, m + 1, r, exclude_self=True)
    
    if B == 0 or A == 0:
        return np.nan
    
    return -np.log(A / B)


def permutation_entropy(
    x: np.ndarray,
    order: int = 3,
    normalize: bool = True
) -> float:
    """
    Compute Permutation Entropy.
    
    Permutation entropy measures complexity based on the relative
    ordering of values rather than their magnitudes. More robust
    to outliers and noise.
    
    Algorithm:
    1. Create overlapping windows of length `order`
    2. Convert each window to a permutation pattern
    3. Compute Shannon entropy of pattern distribution
    
    Args:
        x: Time series
        order: Order (pattern length)
        normalize: Whether to normalize by log(order!)
    
    Returns:
        Permutation entropy [0, 1] if normalized
        
    Example with order=3:
        Window [3.2, 1.1, 4.5] → Pattern [1, 0, 2] (ranks)
        Different patterns indicate different dynamics
    """
    n = len(x)
    
    if n < order:
        return np.nan
    
    # Count permutation patterns
    pattern_counts = {}
    
    for i in range(n - order + 1):
        window = x[i:i + order]
        # Get permutation pattern (argsort gives ranks)
        pattern = tuple(np.argsort(window))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    # Convert to probabilities
    total = sum(pattern_counts.values())
    probs = np.array(list(pattern_counts.values())) / total
    
    # Shannon entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    if normalize:
        # Maximum entropy = log(order!)
        max_entropy = np.log(math.factorial(order))
        entropy = entropy / max_entropy
    
    return entropy


def rolling_entropy(
    series: pd.Series,
    window: int,
    method: str = 'sample',
    m: int = 2,
    r: Optional[float] = None
) -> pd.Series:
    """
    Compute rolling entropy over a time series.
    
    Args:
        series: Input time series
        window: Rolling window size
        method: 'approximate', 'sample', or 'permutation'
        m: Embedding dimension (for ApEn/SampEn)
        r: Tolerance (for ApEn/SampEn)
    
    Returns:
        Rolling entropy series
    """
    if r is None:
        r = 0.2 * series.std()
    
    def compute_entropy(x):
        x_arr = x.values
        if len(x_arr) < window:
            return np.nan
        
        if method == 'approximate':
            return approximate_entropy(x_arr, m, r)
        elif method == 'sample':
            return sample_entropy(x_arr, m, r)
        elif method == 'permutation':
            return permutation_entropy(x_arr, order=m + 1)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return series.rolling(window).apply(compute_entropy, raw=False)


def build_entropy_features(
    df: pd.DataFrame,
    price_col: str = 'close',
    volume_col: str = 'volume',
    windows: list = [20, 50, 100],
    m: int = 2
) -> pd.DataFrame:
    """
    Build complete set of entropy features.
    
    Args:
        df: DataFrame with price data
        price_col: Price column name
        volume_col: Volume column name
        windows: Rolling windows
        m: Embedding dimension
    
    Returns:
        DataFrame with entropy features
    """
    features = pd.DataFrame(index=df.index)
    
    # Price returns for entropy calculation
    returns = df[price_col].pct_change()
    
    for w in windows:
        # Sample entropy of returns
        features[f'sampen_returns_{w}'] = rolling_entropy(
            returns, window=w, method='sample', m=m
        )
        
        # Permutation entropy of returns
        features[f'perm_entropy_{w}'] = rolling_entropy(
            returns, window=w, method='permutation', m=m
        )
        
        # Entropy of price levels
        features[f'sampen_price_{w}'] = rolling_entropy(
            df[price_col], window=w, method='sample', m=m
        )
    
    # Volume entropy
    if volume_col in df.columns:
        log_vol = np.log1p(df[volume_col])
        for w in windows:
            features[f'sampen_volume_{w}'] = rolling_entropy(
                log_vol, window=w, method='sample', m=m
            )
    
    # Entropy change (regime shift indicator)
    for w in windows:
        ent = features[f'sampen_returns_{w}']
        features[f'entropy_change_{w}'] = ent.diff(w // 4)
        features[f'entropy_zscore_{w}'] = (ent - ent.rolling(w * 2).mean()) / ent.rolling(w * 2).std()
    
    return features

