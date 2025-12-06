"""
Position Sizing for Risk Management

Position sizing determines how much capital to allocate to each trade.
Key methods:
- Kelly criterion: Optimal fraction for geometric growth
- Volatility scaling: Constant volatility contribution
- Risk parity: Equal risk contribution

Reference: Kelly (1956), Thorp (2006)
"""

from typing import Optional
import numpy as np
import pandas as pd


def kelly_fraction(
    p_win: float,
    win_loss_ratio: float = 1.0,
    fractional: float = 0.25
) -> float:
    """
    Calculate Kelly criterion position size.
    
    Kelly formula:
        f* = (bp - q) / b
        
    where:
        b = win/loss ratio (odds)
        p = probability of winning
        q = 1 - p = probability of losing
    
    For even odds (b=1): f* = 2p - 1
    
    Args:
        p_win: Probability of winning
        win_loss_ratio: Ratio of win size to loss size
        fractional: Fraction of Kelly to use (0.25 = quarter Kelly)
    
    Returns:
        Optimal bet fraction
        
    Note:
        Full Kelly is aggressive and assumes perfect probability estimates.
        In practice, use fractional Kelly (0.25-0.5) for robustness.
    """
    b = win_loss_ratio
    p = p_win
    q = 1 - p
    
    # Kelly fraction
    f_star = (b * p - q) / b
    
    # Apply fractional scaling
    f_final = fractional * f_star
    
    # Clamp to [0, 1]
    return max(0, min(f_final, 1))


def volatility_scaled_position(
    current_vol: float,
    target_vol: float = 0.15,
    max_leverage: float = 2.0,
    vol_annualization: float = np.sqrt(252)
) -> float:
    """
    Compute position size based on volatility targeting.
    
    Position = target_vol / current_vol
    
    This ensures each position contributes roughly equal volatility
    to the portfolio.
    
    Args:
        current_vol: Current estimated volatility (same freq as returns)
        target_vol: Target portfolio volatility (annualized)
        max_leverage: Maximum position size
        vol_annualization: Annualization factor
    
    Returns:
        Position size (0 to max_leverage)
        
    Example:
        If target_vol = 15% and current_vol = 30% annualized:
        position = 0.15 / 0.30 = 0.5x
    """
    # Annualize current vol
    annualized_vol = current_vol * vol_annualization
    
    # Target position
    position = target_vol / (annualized_vol + 1e-10)
    
    # Cap at max leverage
    return min(position, max_leverage)


def risk_parity_weights(
    volatilities: np.ndarray,
    correlations: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute risk parity weights for portfolio.
    
    Risk parity allocates capital such that each asset contributes
    equal risk to the portfolio.
    
    Simplified (no correlations):
        w_i ∝ 1 / σ_i
        
    With correlations:
        Requires numerical optimization
    
    Args:
        volatilities: Asset volatilities
        correlations: Optional correlation matrix
    
    Returns:
        Weights summing to 1
    """
    if correlations is None:
        # Inverse volatility weighting
        inv_vol = 1.0 / (volatilities + 1e-10)
        weights = inv_vol / inv_vol.sum()
    else:
        # Full risk parity (simplified Newton iteration)
        n = len(volatilities)
        weights = np.ones(n) / n  # Start equal
        
        cov = np.outer(volatilities, volatilities) * correlations
        
        for _ in range(100):
            # Portfolio variance
            port_var = weights @ cov @ weights
            
            # Marginal risk contribution
            mrc = cov @ weights
            
            # Total risk contribution
            trc = weights * mrc
            
            # Target equal risk
            target_rc = port_var / n
            
            # Update weights
            weights = weights * (target_rc / (trc + 1e-10))
            weights = weights / weights.sum()
        
    return weights


class PositionSizer:
    """
    Complete position sizing system.
    
    Combines multiple sizing methodologies with risk limits.
    """
    
    def __init__(
        self,
        method: str = 'volatility_scaled',
        target_vol: float = 0.15,
        kelly_fraction: float = 0.25,
        max_position: float = 1.0,
        min_position: float = 0.0,
        max_daily_turnover: float = 2.0,
        max_drawdown_stop: float = 0.10
    ):
        """
        Initialize position sizer.
        
        Args:
            method: 'kelly', 'volatility_scaled', 'equal', 'confidence'
            target_vol: Target portfolio volatility
            kelly_fraction: Fraction of Kelly to use
            max_position: Maximum position size
            min_position: Minimum position size (below = 0)
            max_daily_turnover: Maximum daily turnover
            max_drawdown_stop: Stop trading if drawdown exceeds this
        """
        self.method = method
        self.target_vol = target_vol
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.min_position = min_position
        self.max_daily_turnover = max_daily_turnover
        self.max_drawdown_stop = max_drawdown_stop
        
        # State tracking
        self.current_position = 0.0
        self.peak_equity = 1.0
        self.current_equity = 1.0
        
    def compute_size(
        self,
        signal: int,
        p_win: Optional[float] = None,
        confidence: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> float:
        """
        Compute position size for a signal.
        
        Args:
            signal: Direction (1, -1, 0)
            p_win: Probability of winning (for Kelly)
            confidence: Model confidence (for confidence scaling)
            volatility: Current volatility (for vol scaling)
        
        Returns:
            Position size (signed)
        """
        if signal == 0:
            return 0.0
        
        # Check drawdown stop
        if self.current_equity < self.peak_equity * (1 - self.max_drawdown_stop):
            return 0.0  # Stop trading during drawdown
        
        # Base position size
        if self.method == 'kelly' and p_win is not None:
            size = kelly_fraction(p_win, fractional=self.kelly_fraction)
            
        elif self.method == 'volatility_scaled' and volatility is not None:
            size = volatility_scaled_position(
                volatility,
                target_vol=self.target_vol,
                max_leverage=self.max_position
            )
            
        elif self.method == 'confidence' and confidence is not None:
            size = confidence * self.max_position
            
        elif self.method == 'equal':
            size = 1.0
            
        else:
            size = 1.0
        
        # Apply confidence scaling if available
        if confidence is not None and self.method != 'confidence':
            size = size * confidence
        
        # Apply limits
        if size < self.min_position:
            size = 0.0
        size = min(size, self.max_position)
        
        # Check turnover constraint
        turnover = abs(signal * size - self.current_position)
        if turnover > self.max_daily_turnover:
            size = abs(self.current_position) + np.sign(turnover) * self.max_daily_turnover
        
        return signal * size
    
    def update_state(self, position: float, return_: float):
        """
        Update state after a trade.
        
        Args:
            position: Current position
            return_: Period return
        """
        self.current_position = position
        
        # Update equity
        pnl = position * return_
        self.current_equity *= (1 + pnl)
        self.peak_equity = max(self.peak_equity, self.current_equity)
    
    def reset_state(self):
        """Reset state tracking."""
        self.current_position = 0.0
        self.peak_equity = 1.0
        self.current_equity = 1.0


def compute_optimal_leverage(
    returns: np.ndarray,
    vol_target: float = 0.15,
    method: str = 'mean_variance'
) -> float:
    """
    Compute optimal leverage for a strategy.
    
    Methods:
    - mean_variance: L* = μ / σ² (maximize Sharpe)
    - kelly: L* = μ / σ² (same as MV for normal)
    - volatility: L* = vol_target / σ
    
    Args:
        returns: Historical returns
        vol_target: Target volatility
        method: Optimization method
    
    Returns:
        Optimal leverage
    """
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    if method == 'mean_variance' or method == 'kelly':
        # Optimal leverage for log utility
        leverage = mu / (sigma ** 2 + 1e-10)
    elif method == 'volatility':
        annualized_vol = sigma * np.sqrt(252 * 24)
        leverage = vol_target / (annualized_vol + 1e-10)
    else:
        leverage = 1.0
    
    return max(0, leverage)

