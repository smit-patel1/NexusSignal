"""
Event-Based Sampling for Financial Time Series

Implements various sampling methods from Lopez de Prado's work:
- CUSUM filter for event detection
- Volatility estimators (Parkinson, Garman-Klass, Yang-Zhang)
- Dollar/Volume bars (conceptual framework)

Key insight: Fixed time intervals (hourly bars) oversample quiet periods
and undersample volatile periods. Event-based sampling addresses this.
"""

from typing import Tuple, Optional, List, Literal
import numpy as np
import pandas as pd
from numba import jit


def get_daily_vol(
    close: pd.Series,
    span: int = 20,
    method: Literal["std", "parkinson", "garman_klass", "yang_zhang"] = "yang_zhang",
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    open_: Optional[pd.Series] = None
) -> pd.Series:
    """
    Compute daily volatility using various estimators.
    
    Methods:
    - std: Simple standard deviation of returns
    - parkinson: Uses high-low range (more efficient than std)
    - garman_klass: Uses OHLC data (even more efficient)
    - yang_zhang: Handles overnight jumps (most robust)
    
    Args:
        close: Close prices
        span: EWM span for smoothing
        method: Volatility estimation method
        high: High prices (required for parkinson, garman_klass, yang_zhang)
        low: Low prices (required for parkinson, garman_klass, yang_zhang)
        open_: Open prices (required for garman_klass, yang_zhang)
    
    Returns:
        Volatility series (annualized if daily data)
        
    Mathematical formulas:
    
    1. Standard deviation:
       σ = std(r_t) where r_t = ln(C_t / C_{t-1})
       
    2. Parkinson (1980):
       σ = sqrt(1/(4*ln(2)) * E[(ln(H/L))^2])
       
    3. Garman-Klass (1980):
       σ = sqrt(0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2)
       
    4. Yang-Zhang (2000):
       σ = sqrt(σ_overnight^2 + k*σ_open^2 + (1-k)*σ_garman_klass^2)
       where k = 0.34 / (1.34 + (n+1)/(n-1))
    """
    # Log returns for std method
    log_returns = np.log(close / close.shift(1))
    
    if method == "std":
        # Simple historical volatility
        vol = log_returns.ewm(span=span).std()
        
    elif method == "parkinson":
        if high is None or low is None:
            raise ValueError("Parkinson method requires high and low prices")
        
        # Parkinson volatility
        log_hl = np.log(high / low)
        parkinson_var = log_hl ** 2 / (4 * np.log(2))
        vol = np.sqrt(parkinson_var.ewm(span=span).mean())
        
    elif method == "garman_klass":
        if high is None or low is None or open_ is None:
            raise ValueError("Garman-Klass method requires OHLC prices")
        
        # Garman-Klass volatility
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        
        gk_var = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        vol = np.sqrt(gk_var.ewm(span=span).mean())
        
    elif method == "yang_zhang":
        if high is None or low is None or open_ is None:
            raise ValueError("Yang-Zhang method requires OHLC prices")
        
        # Yang-Zhang volatility (handles overnight gaps)
        n = span
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        
        # Overnight volatility (close-to-open)
        log_oc = np.log(open_ / close.shift(1))
        var_overnight = log_oc.ewm(span=span).var()
        
        # Open-to-close volatility
        log_co = np.log(close / open_)
        var_open = log_co.ewm(span=span).var()
        
        # Rogers-Satchell volatility (within day)
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)
        
        var_rs = (log_ho * log_hc + log_lo * log_lc).ewm(span=span).mean()
        
        vol = np.sqrt(var_overnight + k * var_open + (1 - k) * var_rs)
        
    else:
        raise ValueError(f"Unknown volatility method: {method}")
    
    return vol


def cusum_filter(
    raw_prices: pd.Series,
    threshold: float,
    use_log_returns: bool = True
) -> pd.DatetimeIndex:
    """
    CUSUM Filter for event detection.
    
    Identifies significant price movements based on cumulative deviations
    from expected returns. Events occur when cumulative deviation exceeds
    a threshold.
    
    The CUSUM filter samples observations according to information arrival
    rate (more samples during volatile periods, fewer during quiet periods).
    
    Args:
        raw_prices: Price series
        threshold: Event threshold (in units of returns or volatility)
        use_log_returns: Whether to use log returns (recommended)
    
    Returns:
        DatetimeIndex of event timestamps
        
    Algorithm:
        1. Compute returns: r_t = ln(P_t / P_{t-1})
        2. Initialize S+ = S- = 0
        3. For each t:
           - S+ = max(0, S+ + r_t - E[r])
           - S- = min(0, S- + r_t - E[r])
           - If S+ > h or S- < -h:
             * Record event at t
             * Reset S+ = S- = 0
    
    Example:
        >>> events = cusum_filter(prices, threshold=0.02)
        >>> # Events where cumulative return deviation > 2%
    """
    if use_log_returns:
        returns = np.log(raw_prices / raw_prices.shift(1)).dropna()
    else:
        returns = raw_prices.pct_change().dropna()
    
    # Expected return (rolling mean)
    expected = returns.rolling(window=20, min_periods=1).mean()
    
    events = []
    s_pos = 0.0
    s_neg = 0.0
    
    for t, r in returns.items():
        e = expected.loc[t] if t in expected.index else 0.0
        
        # Update CUSUM
        s_pos = max(0, s_pos + r - e)
        s_neg = min(0, s_neg + r - e)
        
        # Check for event
        if s_pos > threshold:
            events.append(t)
            s_pos = 0.0
        elif s_neg < -threshold:
            events.append(t)
            s_neg = 0.0
    
    return pd.DatetimeIndex(events)


def get_events(
    close: pd.Series,
    vol: pd.Series,
    threshold_multiplier: float = 2.0,
    min_events_pct: float = 0.1
) -> pd.DatetimeIndex:
    """
    Generate event timestamps using volatility-adjusted CUSUM filter.
    
    The threshold adapts to local volatility, ensuring we sample
    proportionally to information arrival rate.
    
    Args:
        close: Close prices
        vol: Volatility series (from get_daily_vol)
        threshold_multiplier: Multiplier for volatility threshold
        min_events_pct: Minimum events as percentage of data
    
    Returns:
        DatetimeIndex of event timestamps
        
    Note:
        If too few events are generated, the threshold is automatically
        reduced to ensure sufficient samples for training.
    """
    # Initial threshold based on volatility
    threshold = vol.mean() * threshold_multiplier
    
    events = cusum_filter(close, threshold)
    
    # Ensure minimum event count
    min_events = int(len(close) * min_events_pct)
    
    while len(events) < min_events and threshold_multiplier > 0.5:
        threshold_multiplier *= 0.8
        threshold = vol.mean() * threshold_multiplier
        events = cusum_filter(close, threshold)
    
    return events


def get_vertical_barriers(
    events: pd.DatetimeIndex,
    close: pd.Series,
    num_bars: int
) -> pd.Series:
    """
    Create vertical barriers (time expiry) for Triple Barrier Method.
    
    A vertical barrier represents the maximum holding period.
    If neither profit-taking nor stop-loss barriers are hit before
    the vertical barrier, the position is closed.
    
    Args:
        events: Event timestamps (trade entry times)
        close: Close prices for the entire series
        num_bars: Number of bars for vertical barrier
    
    Returns:
        Series mapping event times to vertical barrier times
    """
    # Get all available timestamps
    all_timestamps = close.index
    
    vertical_barriers = {}
    
    for event_time in events:
        # Find index of event time
        try:
            event_idx = all_timestamps.get_loc(event_time)
            
            # Vertical barrier is num_bars ahead
            barrier_idx = min(event_idx + num_bars, len(all_timestamps) - 1)
            vertical_barriers[event_time] = all_timestamps[barrier_idx]
            
        except KeyError:
            continue
    
    return pd.Series(vertical_barriers)


@jit(nopython=True)
def _compute_vol_bars_numba(
    close: np.ndarray,
    volume: np.ndarray,
    timestamps: np.ndarray,
    target_vol: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated volume bar computation.
    
    Volume bars sample the market after a fixed amount of volume
    has traded, rather than fixed time intervals.
    
    Args:
        close: Close prices array
        volume: Volume array
        timestamps: Timestamp array (as int64 nanoseconds)
        target_vol: Target volume per bar
    
    Returns:
        Tuple of (bar_timestamps, bar_closes)
    """
    n = len(close)
    bar_timestamps = []
    bar_closes = []
    
    cum_vol = 0.0
    
    for i in range(n):
        cum_vol += volume[i]
        
        if cum_vol >= target_vol:
            bar_timestamps.append(timestamps[i])
            bar_closes.append(close[i])
            cum_vol = 0.0
    
    return np.array(bar_timestamps), np.array(bar_closes)


def create_volume_bars(
    df: pd.DataFrame,
    target_volume: Optional[float] = None,
    num_bars: Optional[int] = None
) -> pd.DataFrame:
    """
    Create volume bars from tick/minute data.
    
    Volume bars sample after a fixed amount of volume has traded.
    This ensures bars contain roughly equal information content.
    
    Args:
        df: DataFrame with 'close', 'volume' columns and datetime index
        target_volume: Target volume per bar (auto-computed if None)
        num_bars: Target number of bars (used to compute target_volume)
    
    Returns:
        DataFrame of volume bars
        
    Advantages over time bars:
    - Information-synchronous sampling
    - More stable statistical properties
    - Better for high-frequency strategies
    """
    if target_volume is None:
        if num_bars is not None:
            target_volume = df['volume'].sum() / num_bars
        else:
            # Default: create ~same number of bars as time bars
            target_volume = df['volume'].mean() * 10
    
    # Use numba for speed
    timestamps = df.index.values.astype(np.int64)
    close = df['close'].values
    volume = df['volume'].values
    
    bar_ts, bar_close = _compute_vol_bars_numba(close, volume, timestamps, target_volume)
    
    # Convert back to DataFrame
    result = pd.DataFrame({
        'close': bar_close
    }, index=pd.to_datetime(bar_ts))
    
    return result

