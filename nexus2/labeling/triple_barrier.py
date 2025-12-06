"""
Triple Barrier Method (TBM) for Label Generation

Implements the labeling framework from Lopez de Prado's
"Advances in Financial Machine Learning".

Key concept: Instead of predicting raw returns over a fixed horizon,
TBM defines labels based on which barrier is hit first:
- Upper barrier (profit-taking): Label = 1
- Lower barrier (stop-loss): Label = -1
- Vertical barrier (time expiry): Label depends on return sign

This produces more stable, tradable labels that account for:
- Path dependency (not just endpoint returns)
- Risk management (stop-losses are real)
- Dynamic holding periods
"""

from typing import Tuple, Optional, Dict, Union, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numba import jit

from nexus2.data.sampling import get_daily_vol, get_events


@dataclass
class BarrierConfig:
    """Configuration for Triple Barrier Method."""
    profit_taking: float  # Upper barrier multiplier
    stop_loss: float      # Lower barrier multiplier (positive value)
    max_holding: int      # Vertical barrier (bars)
    vol_lookback: int     # Volatility lookback period
    min_return: float     # Minimum return threshold for non-zero label


@jit(nopython=True)
def _get_first_touch_numba(
    close: np.ndarray,
    entry_idx: int,
    upper_barrier: float,
    lower_barrier: float,
    vertical_idx: int,
    entry_price: float
) -> Tuple[int, int, float]:
    """
    Numba-accelerated first touch detection.
    
    Args:
        close: Close prices array
        entry_idx: Entry bar index
        upper_barrier: Upper barrier price
        lower_barrier: Lower barrier price
        vertical_idx: Vertical barrier index
        entry_price: Entry price
    
    Returns:
        Tuple of (exit_idx, label, return)
        - exit_idx: Index where barrier was touched
        - label: 1 (profit), -1 (stop), 0 (vertical)
        - return: Realized return
    """
    for i in range(entry_idx + 1, vertical_idx + 1):
        price = close[i]
        
        # Check upper barrier (profit-taking)
        if price >= upper_barrier:
            ret = (price - entry_price) / entry_price
            return i, 1, ret
        
        # Check lower barrier (stop-loss)
        if price <= lower_barrier:
            ret = (price - entry_price) / entry_price
            return i, -1, ret
    
    # Vertical barrier hit
    exit_price = close[vertical_idx]
    ret = (exit_price - entry_price) / entry_price
    label = 1 if ret > 0 else (-1 if ret < 0 else 0)
    
    return vertical_idx, label, ret


def get_horizontal_barriers(
    close: pd.Series,
    events: pd.DatetimeIndex,
    vol: pd.Series,
    pt_multiplier: float = 2.0,
    sl_multiplier: float = 2.0,
    side: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute horizontal barriers (profit-taking and stop-loss).
    
    Barriers are set as multiples of volatility from entry price:
    - Upper: entry_price * (1 + pt_multiplier * volatility)
    - Lower: entry_price * (1 - sl_multiplier * volatility)
    
    Args:
        close: Close prices
        events: Event timestamps (entry times)
        vol: Volatility series
        pt_multiplier: Profit-taking multiplier
        sl_multiplier: Stop-loss multiplier
        side: Optional side prediction (1 = long, -1 = short)
              If provided, barriers are set asymmetrically
    
    Returns:
        DataFrame with columns: upper_barrier, lower_barrier
        
    Side-aware barriers:
        If side=1 (long): upper is profit, lower is stop
        If side=-1 (short): lower is profit, upper is stop
    """
    barriers = pd.DataFrame(index=events)
    
    for event in events:
        if event not in close.index or event not in vol.index:
            continue
        
        entry_price = close.loc[event]
        event_vol = vol.loc[event]
        
        # Get side if provided
        if side is not None and event in side.index:
            event_side = side.loc[event]
        else:
            event_side = 1  # Default long
        
        # Compute barriers based on side
        if event_side >= 0:
            # Long position: upper is profit, lower is stop
            upper = entry_price * (1 + pt_multiplier * event_vol)
            lower = entry_price * (1 - sl_multiplier * event_vol)
        else:
            # Short position: lower is profit, upper is stop
            upper = entry_price * (1 + sl_multiplier * event_vol)
            lower = entry_price * (1 - pt_multiplier * event_vol)
        
        barriers.loc[event, 'upper_barrier'] = upper
        barriers.loc[event, 'lower_barrier'] = lower
        barriers.loc[event, 'entry_price'] = entry_price
        barriers.loc[event, 'volatility'] = event_vol
        barriers.loc[event, 'side'] = event_side
    
    return barriers


def get_triple_barrier_labels(
    close: pd.Series,
    events: pd.DatetimeIndex,
    barriers: pd.DataFrame,
    max_holding: int,
    min_return: float = 0.0001
) -> pd.DataFrame:
    """
    Generate labels based on which barrier is touched first.
    
    This is the core Triple Barrier Method algorithm.
    
    Args:
        close: Close prices
        events: Event timestamps
        barriers: DataFrame with upper_barrier, lower_barrier columns
        max_holding: Maximum holding period in bars
        min_return: Minimum return for non-zero label at vertical barrier
    
    Returns:
        DataFrame with columns:
        - t1: Exit timestamp (first barrier touch)
        - label: 1 (profit), -1 (loss), 0 (neutral)
        - return: Realized return
        - barrier_type: 'upper', 'lower', or 'vertical'
        - holding_period: Number of bars held
        
    Label logic:
        - Upper barrier hit first → label = 1 (profitable trade)
        - Lower barrier hit first → label = -1 (losing trade)
        - Vertical barrier hit:
            * If return > min_return → label = 1
            * If return < -min_return → label = -1
            * Otherwise → label = 0
    """
    close_arr = close.values
    close_idx = close.index
    
    results = []
    
    for event in events:
        if event not in barriers.index:
            continue
            
        try:
            entry_idx = close_idx.get_loc(event)
        except KeyError:
            continue
        
        # Get barriers
        upper = barriers.loc[event, 'upper_barrier']
        lower = barriers.loc[event, 'lower_barrier']
        entry_price = barriers.loc[event, 'entry_price']
        side = barriers.loc[event, 'side'] if 'side' in barriers.columns else 1
        
        # Vertical barrier index
        vertical_idx = min(entry_idx + max_holding, len(close_arr) - 1)
        
        # Find first touch
        exit_idx, raw_label, ret = _get_first_touch_numba(
            close_arr, entry_idx, upper, lower, vertical_idx, entry_price
        )
        
        # Determine barrier type
        exit_price = close_arr[exit_idx]
        if exit_price >= upper:
            barrier_type = 'upper'
        elif exit_price <= lower:
            barrier_type = 'lower'
        else:
            barrier_type = 'vertical'
        
        # Adjust label for vertical barrier
        if barrier_type == 'vertical':
            if abs(ret) < min_return:
                label = 0
            else:
                label = 1 if ret > 0 else -1
        else:
            label = raw_label
        
        # Adjust for short positions
        if side < 0:
            label = -label  # Flip label for shorts
        
        results.append({
            't0': event,
            't1': close_idx[exit_idx],
            'label': label,
            'return': ret,
            'barrier_type': barrier_type,
            'holding_period': exit_idx - entry_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'side': side
        })
    
    return pd.DataFrame(results).set_index('t0')


def get_events_with_barriers(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    open_: pd.Series,
    config: BarrierConfig,
    side: Optional[pd.Series] = None,
    events: Optional[pd.DatetimeIndex] = None
) -> pd.DataFrame:
    """
    Complete pipeline: events → barriers → labels.
    
    Convenience function that combines all TBM steps.
    
    Args:
        close: Close prices
        high: High prices
        low: Low prices
        open_: Open prices
        config: BarrierConfig with all parameters
        side: Optional side predictions
        events: Optional pre-computed events (if None, uses CUSUM)
    
    Returns:
        DataFrame with complete label information
    """
    # Step 1: Compute volatility
    vol = get_daily_vol(
        close, span=config.vol_lookback,
        method='yang_zhang',
        high=high, low=low, open_=open_
    )
    
    # Step 2: Generate events (if not provided)
    if events is None:
        events = get_events(close, vol, threshold_multiplier=2.0)
    
    # Step 3: Compute horizontal barriers
    barriers = get_horizontal_barriers(
        close, events, vol,
        pt_multiplier=config.profit_taking,
        sl_multiplier=config.stop_loss,
        side=side
    )
    
    # Step 4: Generate labels
    labels = get_triple_barrier_labels(
        close, events, barriers,
        max_holding=config.max_holding,
        min_return=config.min_return
    )
    
    # Merge volatility
    labels['volatility'] = vol.reindex(labels.index)
    
    return labels


class TripleBarrierLabeler:
    """
    Encapsulates Triple Barrier Method labeling logic.
    
    This class provides a scikit-learn compatible interface for
    generating TBM labels, making it easy to integrate with pipelines.
    
    Example:
        >>> labeler = TripleBarrierLabeler(pt=2.0, sl=2.0, max_holding=24)
        >>> labeler.fit(df)  # Fits volatility model
        >>> labels = labeler.transform(df, events)  # Generates labels
    """
    
    def __init__(
        self,
        pt_multiplier: float = 2.0,
        sl_multiplier: float = 2.0,
        max_holding: int = 24,
        vol_lookback: int = 20,
        vol_method: str = 'yang_zhang',
        min_return: float = 0.0001,
        event_threshold: float = 2.0
    ):
        """
        Initialize labeler.
        
        Args:
            pt_multiplier: Profit-taking barrier multiplier
            sl_multiplier: Stop-loss barrier multiplier
            max_holding: Maximum holding period in bars
            vol_lookback: Volatility lookback period
            vol_method: Volatility estimation method
            min_return: Minimum return for non-zero label
            event_threshold: CUSUM event threshold multiplier
        """
        self.pt_multiplier = pt_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding = max_holding
        self.vol_lookback = vol_lookback
        self.vol_method = vol_method
        self.min_return = min_return
        self.event_threshold = event_threshold
        
        self.vol_ = None
        self.events_ = None
        
    def fit(
        self,
        df: pd.DataFrame,
        close_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        open_col: str = 'open'
    ) -> 'TripleBarrierLabeler':
        """
        Fit volatility model and detect events.
        
        Args:
            df: DataFrame with OHLC data
            close_col, high_col, low_col, open_col: Column names
        
        Returns:
            self
        """
        # Compute volatility
        self.vol_ = get_daily_vol(
            df[close_col],
            span=self.vol_lookback,
            method=self.vol_method,
            high=df[high_col],
            low=df[low_col],
            open_=df[open_col]
        )
        
        # Detect events
        self.events_ = get_events(
            df[close_col], self.vol_,
            threshold_multiplier=self.event_threshold
        )
        
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
        events: Optional[pd.DatetimeIndex] = None,
        side: Optional[pd.Series] = None,
        close_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Generate labels for events.
        
        Args:
            df: DataFrame with price data
            events: Event timestamps (uses fitted events if None)
            side: Optional side predictions
            close_col: Close price column name
        
        Returns:
            DataFrame with labels
        """
        if events is None:
            events = self.events_
        
        if self.vol_ is None:
            raise ValueError("Must call fit() before transform()")
        
        close = df[close_col]
        
        # Compute barriers
        barriers = get_horizontal_barriers(
            close, events, self.vol_,
            pt_multiplier=self.pt_multiplier,
            sl_multiplier=self.sl_multiplier,
            side=side
        )
        
        # Generate labels
        labels = get_triple_barrier_labels(
            close, events, barriers,
            max_holding=self.max_holding,
            min_return=self.min_return
        )
        
        return labels
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        side: Optional[pd.Series] = None,
        close_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        open_col: str = 'open'
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        """
        self.fit(df, close_col, high_col, low_col, open_col)
        return self.transform(df, side=side, close_col=close_col)
    
    def get_label_distribution(self, labels: pd.DataFrame) -> Dict[str, float]:
        """
        Compute label distribution statistics.
        
        Returns:
            Dictionary with distribution info
        """
        dist = labels['label'].value_counts(normalize=True)
        
        return {
            'pct_positive': dist.get(1, 0),
            'pct_negative': dist.get(-1, 0),
            'pct_neutral': dist.get(0, 0),
            'total_events': len(labels),
            'avg_holding': labels['holding_period'].mean(),
            'avg_return': labels['return'].mean(),
            'hit_upper_pct': (labels['barrier_type'] == 'upper').mean(),
            'hit_lower_pct': (labels['barrier_type'] == 'lower').mean(),
            'hit_vertical_pct': (labels['barrier_type'] == 'vertical').mean(),
        }


def tune_barriers(
    df: pd.DataFrame,
    pt_range: List[float] = [1.0, 1.5, 2.0, 2.5, 3.0],
    sl_range: List[float] = [1.0, 1.5, 2.0, 2.5, 3.0],
    max_holding_range: List[int] = [12, 24, 48],
    target_balance: float = 0.4
) -> Dict[str, any]:
    """
    Tune barrier parameters to achieve target label balance.
    
    Goal: Find parameters that produce roughly balanced labels
    (e.g., 40% positive, 40% negative, 20% neutral).
    
    Args:
        df: DataFrame with OHLC data
        pt_range: Profit-taking multipliers to try
        sl_range: Stop-loss multipliers to try
        max_holding_range: Max holding periods to try
        target_balance: Target percentage for positive/negative labels
    
    Returns:
        Dictionary with best parameters and results
    """
    best_params = None
    best_score = float('inf')
    results = []
    
    for pt in pt_range:
        for sl in sl_range:
            for mh in max_holding_range:
                labeler = TripleBarrierLabeler(
                    pt_multiplier=pt,
                    sl_multiplier=sl,
                    max_holding=mh
                )
                
                try:
                    labels = labeler.fit_transform(df)
                    dist = labeler.get_label_distribution(labels)
                    
                    # Score: deviation from target balance
                    score = (
                        abs(dist['pct_positive'] - target_balance) +
                        abs(dist['pct_negative'] - target_balance)
                    )
                    
                    results.append({
                        'pt': pt, 'sl': sl, 'max_holding': mh,
                        'score': score, **dist
                    })
                    
                    if score < best_score:
                        best_score = score
                        best_params = {'pt': pt, 'sl': sl, 'max_holding': mh}
                        
                except Exception as e:
                    continue
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': pd.DataFrame(results)
    }

