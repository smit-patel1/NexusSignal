"""
Microstructure Features for Financial ML

Market microstructure features capture information about
trading activity, liquidity, and order flow that simple
price-based indicators miss.

Key features implemented:
- VPIN: Volume-Synchronized Probability of Informed Trading
- Kyle's Lambda: Price impact coefficient
- Roll Spread: Implicit bid-ask spread
- Amihud Illiquidity: Price impact per dollar volume
- Order Flow Imbalance: Buy vs sell pressure

Reference: Easley et al., "VPIN and the Flash Crash" (2012)
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from numba import jit


def classify_volume_bulk(
    close: pd.Series,
    volume: pd.Series,
    method: str = "tick"
) -> Tuple[pd.Series, pd.Series]:
    """
    Classify volume into buy/sell initiated trades.
    
    Methods:
    - tick: Sign of price change (+ = buy, - = sell)
    - bulk: BVC method (bulk volume classification)
    
    Args:
        close: Close prices
        volume: Trade volume
        method: Classification method
    
    Returns:
        Tuple of (buy_volume, sell_volume)
        
    Tick rule logic:
        If price_t > price_{t-1}: volume is buy-initiated
        If price_t < price_{t-1}: volume is sell-initiated
        If price_t == price_{t-1}: use previous classification
    """
    price_change = close.diff()
    
    if method == "tick":
        # Tick rule: sign of price change
        sign = np.sign(price_change)
        
        # Forward fill zeros (unchanged prices use previous sign)
        sign = sign.replace(0, np.nan).ffill().fillna(0)
        
        buy_vol = volume.where(sign > 0, 0)
        sell_vol = volume.where(sign < 0, 0)
        
    elif method == "bulk":
        # Bulk Volume Classification (BVC)
        # Split volume based on normalized price change
        norm_change = price_change / close.rolling(20).std()
        buy_pct = norm_change.clip(0, 1)  # More sophisticated: use CDF
        
        buy_vol = volume * (0.5 + 0.5 * np.tanh(norm_change))
        sell_vol = volume - buy_vol
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return buy_vol, sell_vol


@jit(nopython=True)
def _compute_vpin_numba(
    buy_vol: np.ndarray,
    sell_vol: np.ndarray,
    volume: np.ndarray,
    bucket_size: float,
    n_buckets: int
) -> np.ndarray:
    """
    Numba-accelerated VPIN computation.
    
    VPIN = |V_buy - V_sell| / V_total over volume buckets
    
    Args:
        buy_vol: Buy volume array
        sell_vol: Sell volume array
        volume: Total volume array
        bucket_size: Target volume per bucket
        n_buckets: Number of buckets for VPIN window
    
    Returns:
        VPIN values
    """
    n = len(volume)
    vpin = np.full(n, np.nan)
    
    # Initialize bucket tracking
    cum_vol = 0.0
    cum_buy = 0.0
    cum_sell = 0.0
    bucket_imbalances = []
    
    for i in range(n):
        cum_vol += volume[i]
        cum_buy += buy_vol[i]
        cum_sell += sell_vol[i]
        
        # Check if bucket is complete
        while cum_vol >= bucket_size:
            # Proportionally split this bar
            pct = bucket_size / cum_vol
            
            # Record bucket imbalance
            bucket_buy = cum_buy * pct
            bucket_sell = cum_sell * pct
            imbalance = abs(bucket_buy - bucket_sell) / bucket_size
            bucket_imbalances.append(imbalance)
            
            # Carry over remainder
            cum_vol -= bucket_size
            cum_buy = cum_buy * (1 - pct)
            cum_sell = cum_sell * (1 - pct)
            
            # Compute VPIN if enough buckets
            if len(bucket_imbalances) >= n_buckets:
                vpin[i] = np.mean(np.array(bucket_imbalances[-n_buckets:]))
    
    return vpin


def compute_vpin(
    close: pd.Series,
    volume: pd.Series,
    n_buckets: int = 50,
    bucket_volume_pct: float = 0.02
) -> pd.Series:
    """
    Compute Volume-Synchronized Probability of Informed Trading (VPIN).
    
    VPIN measures the probability that a trade is informed (has private
    information) vs uninformed (liquidity-motivated). High VPIN indicates
    potential adverse selection risk.
    
    Args:
        close: Close prices
        volume: Trade volume
        n_buckets: Number of buckets for averaging
        bucket_volume_pct: Each bucket as % of avg daily volume
    
    Returns:
        VPIN series [0, 1] where higher = more informed trading
        
    Interpretation:
    - VPIN < 0.3: Normal market conditions
    - VPIN 0.3-0.5: Elevated informed trading
    - VPIN > 0.5: High probability of informed trading (potential toxic flow)
    
    Note: VPIN spiked before the 2010 Flash Crash, showing predictive value.
    """
    # Classify volume
    buy_vol, sell_vol = classify_volume_bulk(close, volume)
    
    # Compute bucket size (based on average volume)
    avg_daily_vol = volume.rolling(20).mean()
    bucket_size = (avg_daily_vol * bucket_volume_pct).median()
    
    if bucket_size <= 0 or np.isnan(bucket_size):
        bucket_size = volume.mean() * bucket_volume_pct
    
    # Compute VPIN using numba
    vpin_values = _compute_vpin_numba(
        buy_vol.values.astype(float),
        sell_vol.values.astype(float),
        volume.values.astype(float),
        float(bucket_size),
        n_buckets
    )
    
    return pd.Series(vpin_values, index=close.index, name='vpin')


def compute_kyle_lambda(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Compute Kyle's Lambda (price impact coefficient).
    
    Kyle's Lambda measures how much price moves per unit of order flow.
    Higher lambda = lower liquidity (prices move more per trade).
    
    Formula:
        Δp_t = λ * signed_volume_t + ε
        
    where λ is estimated via rolling regression.
    
    Args:
        close: Close prices
        volume: Trade volume
        window: Rolling window for regression
    
    Returns:
        Lambda series (price impact per unit volume)
        
    Interpretation:
    - Low λ: Deep liquidity, trades don't move price much
    - High λ: Thin liquidity, trades have large price impact
    
    Reference: Kyle (1985), "Continuous Auctions and Insider Trading"
    """
    # Price changes
    delta_price = close.diff()
    
    # Signed volume (using tick rule)
    sign = np.sign(delta_price).replace(0, np.nan).ffill().fillna(0)
    signed_volume = sign * volume
    
    # Rolling OLS: Δp = λ * signed_vol + ε
    # λ = Cov(Δp, signed_vol) / Var(signed_vol)
    cov_term = delta_price.rolling(window).cov(signed_volume)
    var_term = signed_volume.rolling(window).var()
    
    kyle_lambda = cov_term / var_term.replace(0, np.nan)
    
    return kyle_lambda.rename('kyle_lambda')


def compute_roll_spread(
    close: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Compute Roll's implicit bid-ask spread measure.
    
    Roll (1984) showed that bid-ask bounce causes negative autocorrelation
    in price changes. The spread can be estimated from this autocorrelation.
    
    Formula:
        spread = 2 * sqrt(-Cov(Δp_t, Δp_{t-1}))
        
    if Cov < 0 (otherwise undefined)
    
    Args:
        close: Close prices
        window: Rolling window
    
    Returns:
        Estimated spread as fraction of price
        
    Interpretation:
    - Low spread: Tight market, low trading costs
    - High spread: Wide market, high trading costs
    
    Note: Only valid when autocov is negative. NaN otherwise.
    """
    delta_price = close.diff()
    
    # Rolling autocovariance
    autocov = delta_price.rolling(window).apply(
        lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan,
        raw=False
    )
    
    # Roll measure (only valid for negative autocov)
    spread = np.where(autocov < 0, 2 * np.sqrt(-autocov), np.nan)
    
    # Normalize by price
    spread_pct = spread / close.values
    
    return pd.Series(spread_pct, index=close.index, name='roll_spread')


def compute_amihud(
    close: pd.Series,
    volume: pd.Series,
    dollar_volume: Optional[pd.Series] = None,
    window: int = 20
) -> pd.Series:
    """
    Compute Amihud Illiquidity measure.
    
    Amihud (2002) illiquidity measures the price impact per dollar traded:
    
        ILLIQ = |r_t| / (price_t * volume_t)
        
    Higher ILLIQ = less liquid (price moves more per dollar traded).
    
    Args:
        close: Close prices
        volume: Trade volume
        dollar_volume: Pre-computed dollar volume (optional)
        window: Rolling window for averaging
    
    Returns:
        Amihud illiquidity series
        
    Interpretation:
    - Low ILLIQ: Liquid market, can trade size without impact
    - High ILLIQ: Illiquid market, trades cause price impact
    
    Note: Multiply by 10^6 for interpretability.
    """
    returns = close.pct_change().abs()
    
    if dollar_volume is None:
        dollar_volume = close * volume
    
    # Avoid division by zero
    dollar_volume = dollar_volume.replace(0, np.nan)
    
    illiq = returns / dollar_volume
    
    # Rolling average
    amihud = illiq.rolling(window).mean() * 1e6  # Scale for readability
    
    return amihud.rename('amihud')


def compute_order_flow_imbalance(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Compute Order Flow Imbalance (OFI).
    
    OFI measures the excess buying vs selling pressure.
    
        OFI = (Buy_volume - Sell_volume) / Total_volume
        
    Ranges from -1 (all selling) to +1 (all buying).
    
    Args:
        close: Close prices
        volume: Trade volume
        window: Rolling window
    
    Returns:
        Order flow imbalance series [-1, 1]
    """
    buy_vol, sell_vol = classify_volume_bulk(close, volume)
    
    total = buy_vol + sell_vol
    total = total.replace(0, np.nan)
    
    ofi = (buy_vol - sell_vol) / total
    
    # Rolling smoothed
    return ofi.rolling(window).mean().rename('ofi')


def build_microstructure_features(
    df: pd.DataFrame,
    close_col: str = 'close',
    volume_col: str = 'volume',
    windows: list = [10, 20, 50]
) -> pd.DataFrame:
    """
    Build complete set of microstructure features.
    
    Args:
        df: DataFrame with OHLCV data
        close_col: Close price column name
        volume_col: Volume column name
        windows: Windows for rolling computations
    
    Returns:
        DataFrame with microstructure features
    """
    close = df[close_col]
    volume = df[volume_col]
    
    features = pd.DataFrame(index=df.index)
    
    # VPIN
    features['vpin'] = compute_vpin(close, volume)
    
    # Kyle's Lambda at multiple windows
    for w in windows:
        features[f'kyle_lambda_{w}'] = compute_kyle_lambda(close, volume, window=w)
    
    # Roll spread
    for w in windows:
        features[f'roll_spread_{w}'] = compute_roll_spread(close, window=w)
    
    # Amihud illiquidity
    for w in windows:
        features[f'amihud_{w}'] = compute_amihud(close, volume, window=w)
    
    # Order flow imbalance
    for w in windows:
        features[f'ofi_{w}'] = compute_order_flow_imbalance(close, volume, window=w)
    
    # Volume classification
    buy_vol, sell_vol = classify_volume_bulk(close, volume)
    features['buy_vol_pct'] = buy_vol / volume.replace(0, np.nan)
    features['sell_vol_pct'] = sell_vol / volume.replace(0, np.nan)
    
    # Imbalance volatility
    ofi = compute_order_flow_imbalance(close, volume, window=1)
    for w in windows:
        features[f'ofi_vol_{w}'] = ofi.rolling(w).std()
    
    return features

