"""
Signal Generation from Model Outputs

Converts probabilistic model predictions into actionable trading signals.

The signal generation pipeline:
1. Primary model: P(barrier hit | features) or distribution params
2. Meta model: P(primary is correct | features)
3. Signal filter: Only act on high-confidence predictions
4. Position sizing: Scale positions by confidence
5. Risk management: Apply limits and constraints
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


def convert_proba_to_signal(
    p_upper: np.ndarray,
    p_lower: np.ndarray,
    threshold: float = 0.55,
    neutral_zone: float = 0.1
) -> np.ndarray:
    """
    Convert barrier probabilities to trading signals.
    
    Logic:
    - If P(upper) > threshold and P(upper) - P(lower) > neutral_zone → Long
    - If P(lower) > threshold and P(lower) - P(upper) > neutral_zone → Short
    - Otherwise → No position
    
    Args:
        p_upper: P(upper barrier hit) - profit for long
        p_lower: P(lower barrier hit) - profit for short
        threshold: Minimum probability for signal
        neutral_zone: Minimum difference between probabilities
    
    Returns:
        Signal array: 1 (long), -1 (short), 0 (no position)
    """
    signals = np.zeros(len(p_upper))
    
    # Long signals
    long_mask = (p_upper > threshold) & (p_upper - p_lower > neutral_zone)
    signals[long_mask] = 1
    
    # Short signals
    short_mask = (p_lower > threshold) & (p_lower - p_upper > neutral_zone)
    signals[short_mask] = -1
    
    return signals


def compute_expected_payoff(
    p_upper: np.ndarray,
    p_lower: np.ndarray,
    pt_return: float = 0.02,  # Profit-taking return
    sl_return: float = -0.02  # Stop-loss return
) -> np.ndarray:
    """
    Compute expected payoff for each trade.
    
    E[payoff] = P(upper) * PT_return + P(lower) * SL_return + P(vertical) * 0
    
    For long positions:
    - Upper barrier hit → positive return (PT)
    - Lower barrier hit → negative return (SL)
    
    Args:
        p_upper: P(upper barrier hit)
        p_lower: P(lower barrier hit)
        pt_return: Expected return if profit-taking hit
        sl_return: Expected return if stop-loss hit
    
    Returns:
        Expected payoff array
    """
    p_vertical = 1 - p_upper - p_lower
    
    expected = (
        p_upper * pt_return +
        p_lower * sl_return +
        p_vertical * 0  # Assume neutral at vertical
    )
    
    return expected


class SignalGenerator:
    """
    Complete signal generation pipeline.
    
    Combines:
    1. Primary model predictions (barrier probabilities)
    2. Meta model filtering (confidence threshold)
    3. Position sizing (Kelly, volatility-scaled)
    4. Risk limits (max position, drawdown limits)
    
    Example:
        >>> generator = SignalGenerator(
        ...     threshold=0.55,
        ...     min_confidence=0.5,
        ...     position_method='kelly'
        ... )
        >>> signals = generator.generate(
        ...     primary_proba=model.predict_proba(X),
        ...     meta_confidence=meta_model.predict_confidence(X),
        ...     volatility=vol
        ... )
    """
    
    def __init__(
        self,
        threshold: float = 0.55,
        neutral_zone: float = 0.1,
        min_confidence: float = 0.5,
        position_method: str = 'volatility_scaled',
        target_vol: float = 0.15,
        max_position: float = 1.0,
        kelly_fraction: float = 0.25,
        use_meta_labeling: bool = True
    ):
        """
        Initialize signal generator.
        
        Args:
            threshold: Minimum probability for primary signal
            neutral_zone: Minimum prob difference for conviction
            min_confidence: Minimum meta confidence for trading
            position_method: 'equal', 'kelly', 'volatility_scaled'
            target_vol: Target annualized volatility
            max_position: Maximum position size (fraction)
            kelly_fraction: Fraction of full Kelly to use
            use_meta_labeling: Whether to apply meta model filter
        """
        self.threshold = threshold
        self.neutral_zone = neutral_zone
        self.min_confidence = min_confidence
        self.position_method = position_method
        self.target_vol = target_vol
        self.max_position = max_position
        self.kelly_fraction = kelly_fraction
        self.use_meta_labeling = use_meta_labeling
        
    def generate(
        self,
        primary_proba: np.ndarray,
        meta_confidence: Optional[np.ndarray] = None,
        volatility: Optional[np.ndarray] = None,
        expected_return: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals from model predictions.
        
        Args:
            primary_proba: Shape (N, 3) with P(upper), P(lower), P(vertical)
            meta_confidence: P(primary is correct) from meta model
            volatility: Current volatility for each sample
            expected_return: Expected return if signal is correct
        
        Returns:
            DataFrame with columns:
            - signal: Direction (1, -1, 0)
            - position: Sized position
            - confidence: Meta model confidence
            - expected_payoff: Expected return
            - trade: Whether to execute
        """
        n = len(primary_proba)
        
        # Extract barrier probabilities
        if primary_proba.shape[1] == 3:
            p_upper = primary_proba[:, 0]
            p_lower = primary_proba[:, 1]
            p_vertical = primary_proba[:, 2]
        elif primary_proba.shape[1] == 2:
            # Binary: P(upper) and P(lower) only
            p_upper = primary_proba[:, 0]
            p_lower = primary_proba[:, 1]
        else:
            raise ValueError(f"Unexpected shape: {primary_proba.shape}")
        
        # Step 1: Generate raw signals from primary model
        signals = convert_proba_to_signal(
            p_upper, p_lower,
            threshold=self.threshold,
            neutral_zone=self.neutral_zone
        )
        
        # Step 2: Compute expected payoff
        payoffs = compute_expected_payoff(p_upper, p_lower)
        
        # Step 3: Apply meta-labeling filter
        if self.use_meta_labeling and meta_confidence is not None:
            trade_mask = meta_confidence >= self.min_confidence
        else:
            trade_mask = np.ones(n, dtype=bool)
        
        # Step 4: Position sizing
        if self.position_method == 'equal':
            positions = np.where(signals != 0, 1.0, 0.0)
            
        elif self.position_method == 'kelly':
            # Kelly: f* = p - q/b where p=win prob, q=lose prob, b=odds
            # Simplified: f* = 2p - 1 for even odds
            p_win = np.where(signals > 0, p_upper, p_lower)
            kelly = self.kelly_fraction * (2 * p_win - 1)
            positions = np.clip(kelly, 0, self.max_position)
            
        elif self.position_method == 'volatility_scaled':
            if volatility is None:
                volatility = np.ones(n) * 0.02  # Default 2% vol
            
            # Scale position inversely with volatility
            # Position = target_vol / (vol * sqrt(252 * 24))
            annualized_vol = volatility * np.sqrt(252 * 24)
            positions = self.target_vol / (annualized_vol + 1e-10)
            positions = np.clip(positions, 0, self.max_position)
            
        else:
            raise ValueError(f"Unknown position method: {self.position_method}")
        
        # Step 5: Apply meta-model confidence scaling
        if meta_confidence is not None:
            # Scale position by confidence
            confidence_scaling = (meta_confidence - self.min_confidence) / (1 - self.min_confidence)
            confidence_scaling = np.clip(confidence_scaling, 0, 1)
            positions = positions * confidence_scaling
        
        # Step 6: Zero out filtered positions
        positions = np.where(trade_mask & (signals != 0), positions, 0)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'signal': signals,
            'raw_position': positions,
            'position': signals * positions,  # Signed position
            'confidence': meta_confidence if meta_confidence is not None else np.ones(n),
            'p_upper': p_upper,
            'p_lower': p_lower,
            'expected_payoff': payoffs,
            'trade': trade_mask & (signals != 0),
        })
        
        return result
    
    def generate_from_quantiles(
        self,
        quantiles: np.ndarray,
        quantile_levels: list = [0.05, 0.25, 0.5, 0.75, 0.95],
        meta_confidence: Optional[np.ndarray] = None,
        volatility: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Generate signals from quantile predictions.
        
        Uses median for direction, IQR for uncertainty.
        
        Args:
            quantiles: Shape (N, len(quantile_levels))
            quantile_levels: List of quantile levels
            meta_confidence: Optional meta model confidence
            volatility: Current volatility
        
        Returns:
            DataFrame with signal information
        """
        n = len(quantiles)
        
        # Find median index
        if 0.5 in quantile_levels:
            median_idx = quantile_levels.index(0.5)
        else:
            median_idx = len(quantile_levels) // 2
        
        median_pred = quantiles[:, median_idx]
        
        # Direction from median
        signals = np.sign(median_pred)
        signals[np.abs(median_pred) < 0.0001] = 0  # Neutral zone
        
        # Confidence from IQR (smaller = more confident)
        if 0.25 in quantile_levels and 0.75 in quantile_levels:
            q25_idx = quantile_levels.index(0.25)
            q75_idx = quantile_levels.index(0.75)
            iqr = quantiles[:, q75_idx] - quantiles[:, q25_idx]
            
            # Confidence inversely related to IQR
            confidence = 1 / (1 + iqr * 100)  # Scale appropriately
        else:
            confidence = np.ones(n)
        
        # Combine with meta confidence if provided
        if meta_confidence is not None:
            confidence = confidence * meta_confidence
        
        # Position sizing based on confidence
        positions = np.clip(confidence, 0, self.max_position)
        
        # Volatility scaling
        if volatility is not None and self.position_method == 'volatility_scaled':
            annualized_vol = volatility * np.sqrt(252 * 24)
            vol_scale = self.target_vol / (annualized_vol + 1e-10)
            positions = positions * np.clip(vol_scale, 0, 2.0)
        
        # Apply filters
        trade_mask = confidence >= self.min_confidence
        positions = np.where(trade_mask & (signals != 0), positions, 0)
        
        return pd.DataFrame({
            'signal': signals,
            'position': signals * positions,
            'confidence': confidence,
            'median_pred': median_pred,
            'iqr': iqr if 'iqr' in dir() else np.nan,
            'trade': trade_mask & (signals != 0),
        })
    
    def evaluate_signals(
        self,
        signals_df: pd.DataFrame,
        actual_returns: np.ndarray,
        actual_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate signal quality.
        
        Args:
            signals_df: Output from generate()
            actual_returns: Realized returns
            actual_labels: Actual TBM labels (optional)
        
        Returns:
            Dictionary with performance metrics
        """
        # Strategy returns
        position = signals_df['position'].values
        strategy_returns = position * actual_returns
        
        # Filter to only trades taken
        traded = signals_df['trade'].values
        
        metrics = {}
        
        # Hit rate
        if traded.sum() > 0:
            correct_direction = (np.sign(position[traded]) == np.sign(actual_returns[traded]))
            metrics['hit_rate'] = correct_direction.mean()
        
        # Sharpe ratio (annualized hourly)
        if len(strategy_returns) > 0:
            sr = strategy_returns.mean() / (strategy_returns.std() + 1e-10)
            metrics['sharpe_ratio'] = sr * np.sqrt(252 * 24)
        
        # Total return
        metrics['total_return'] = strategy_returns.sum()
        
        # Average return per trade
        if traded.sum() > 0:
            metrics['avg_return_per_trade'] = strategy_returns[traded].mean()
        
        # Trade frequency
        metrics['trade_frequency'] = traded.mean()
        
        # Confidence calibration (if labels provided)
        if actual_labels is not None and traded.sum() > 0:
            predicted_correct = signals_df.loc[traded, 'signal'].values
            actually_correct = (actual_labels[traded] == np.sign(signals_df.loc[traded, 'signal'].values))
            metrics['precision'] = actually_correct.mean()
        
        return metrics

