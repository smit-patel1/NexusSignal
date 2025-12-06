"""
Meta-Labeling for Signal Filtering and Bet Sizing

Meta-labeling is a two-stage approach:
1. Primary model: Predicts trade direction (side)
2. Meta model: Predicts whether to take the trade (confidence)

Key benefits:
- Separates "what to do" from "how much to bet"
- Improves precision by filtering low-confidence signals
- Enables sophisticated bet sizing
- Reduces overfitting (models are simpler)

Reference: Lopez de Prado, "Advances in Financial Machine Learning"
"""

from typing import Tuple, Optional, Dict, Literal, Union
import numpy as np
import pandas as pd
from scipy.stats import norm


def get_meta_labels(
    primary_prediction: pd.Series,
    actual_labels: pd.Series,
    threshold: float = 0.0
) -> pd.Series:
    """
    Generate meta-labels based on primary model correctness.
    
    Meta-label = 1 if primary model was correct, 0 otherwise.
    
    This becomes the target for the meta-model: "Should we trust
    the primary model's prediction for this sample?"
    
    Args:
        primary_prediction: Primary model side predictions (-1, 0, 1)
        actual_labels: True labels from Triple Barrier Method
        threshold: Minimum predicted probability for "take trade"
    
    Returns:
        Series of meta-labels (1 = correct, 0 = incorrect)
        
    Example:
        Primary predicts: Long (+1)
        Actual label: +1 (profit-taking barrier hit)
        Meta-label: 1 (correct prediction → should have traded)
        
        Primary predicts: Long (+1)
        Actual label: -1 (stop-loss hit)
        Meta-label: 0 (wrong prediction → should not have traded)
    """
    # Align indices
    common_idx = primary_prediction.index.intersection(actual_labels.index)
    pred = primary_prediction.loc[common_idx]
    actual = actual_labels.loc[common_idx]
    
    # Meta-label: 1 if prediction matches actual direction
    meta_labels = (np.sign(pred) == np.sign(actual)).astype(int)
    
    # Handle neutral cases (actual=0 means inconclusive)
    neutral_mask = actual == 0
    meta_labels.loc[neutral_mask] = 0  # Don't trade on inconclusive
    
    return meta_labels


def apply_meta_labeling(
    primary_side: pd.Series,
    meta_probability: pd.Series,
    min_confidence: float = 0.5,
    bet_sizing: Literal["equal", "linear", "sigmoid", "kelly"] = "sigmoid",
    kelly_fraction: float = 0.25
) -> pd.DataFrame:
    """
    Apply meta-model predictions for signal filtering and bet sizing.
    
    The meta-model outputs P(primary is correct). This is used to:
    1. Filter signals below confidence threshold
    2. Size positions based on confidence
    
    Args:
        primary_side: Primary model predictions (-1, 0, 1)
        meta_probability: Meta-model P(primary is correct)
        min_confidence: Minimum probability to take trade
        bet_sizing: Method for converting probability to position size
        kelly_fraction: Fraction of Kelly criterion to use
    
    Returns:
        DataFrame with:
        - side: Final trading side
        - probability: Meta-model probability
        - bet_size: Position size [0, 1]
        - take_trade: Boolean flag
        
    Bet sizing methods:
    
    1. Equal: bet_size = 1 if p > threshold else 0
    
    2. Linear: bet_size = max(0, (p - threshold) / (1 - threshold))
       Maps [threshold, 1] → [0, 1]
       
    3. Sigmoid: bet_size = 2 * (1 / (1 + exp(-k*(p - 0.5)))) - 1
       S-shaped response with parameter k
       
    4. Kelly: bet_size = p - (1-p) = 2p - 1
       Optimal fraction based on edge (use kelly_fraction < 1)
    """
    # Align indices
    common_idx = primary_side.index.intersection(meta_probability.index)
    side = primary_side.loc[common_idx]
    prob = meta_probability.loc[common_idx]
    
    # Initialize result
    result = pd.DataFrame(index=common_idx)
    result['side'] = side
    result['probability'] = prob
    
    # Filter by confidence threshold
    take_trade = prob >= min_confidence
    result['take_trade'] = take_trade
    
    # Compute bet sizes
    if bet_sizing == "equal":
        bet_size = take_trade.astype(float)
        
    elif bet_sizing == "linear":
        # Linear scaling from threshold to 1
        bet_size = np.maximum(0, (prob - min_confidence) / (1 - min_confidence))
        bet_size = bet_size.where(take_trade, 0)
        
    elif bet_sizing == "sigmoid":
        # Sigmoid transformation
        k = 10  # Steepness parameter
        bet_size = 2 / (1 + np.exp(-k * (prob - 0.5))) - 1
        bet_size = np.maximum(0, bet_size)
        bet_size = bet_size.where(take_trade, 0)
        
    elif bet_sizing == "kelly":
        # Kelly criterion: f* = (bp - q) / b where b=1 (even odds)
        # Simplifies to f* = 2p - 1 for even odds
        bet_size = kelly_fraction * (2 * prob - 1)
        bet_size = np.maximum(0, bet_size)
        bet_size = bet_size.where(take_trade, 0)
        
    else:
        raise ValueError(f"Unknown bet sizing method: {bet_sizing}")
    
    result['bet_size'] = bet_size
    
    # Final position = side * bet_size (zeros out filtered signals)
    result['position'] = side * bet_size
    
    return result


class MetaLabeler:
    """
    Complete meta-labeling pipeline.
    
    Workflow:
    1. Train primary model (e.g., predict side from features)
    2. Generate meta-labels based on primary model correctness
    3. Train meta-model (predict meta-labels from features + primary output)
    4. Use meta-model for signal filtering and bet sizing
    
    Example:
        >>> # Stage 1: Primary model
        >>> primary_model = train_primary_model(X_train, y_side_train)
        >>> primary_pred = primary_model.predict(X_train)
        >>> 
        >>> # Stage 2: Generate meta-labels
        >>> meta_labeler = MetaLabeler()
        >>> meta_labels = meta_labeler.get_meta_training_data(
        ...     primary_pred, y_labels_train
        ... )
        >>> 
        >>> # Stage 3: Train meta-model
        >>> meta_model = train_meta_model(X_train, meta_labels)
        >>> 
        >>> # Stage 4: Apply at inference
        >>> primary_side = primary_model.predict(X_test)
        >>> meta_prob = meta_model.predict_proba(X_test)[:, 1]
        >>> final_signals = meta_labeler.apply(primary_side, meta_prob)
    """
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        bet_sizing: Literal["equal", "linear", "sigmoid", "kelly"] = "sigmoid",
        kelly_fraction: float = 0.25
    ):
        """
        Initialize meta-labeler.
        
        Args:
            min_confidence: Minimum probability threshold
            bet_sizing: Bet sizing method
            kelly_fraction: Fraction of Kelly to use
        """
        self.min_confidence = min_confidence
        self.bet_sizing = bet_sizing
        self.kelly_fraction = kelly_fraction
        
    def get_meta_training_data(
        self,
        primary_predictions: pd.Series,
        true_labels: pd.Series,
        features: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for meta-model.
        
        The meta-model learns to predict when the primary model is correct.
        Input features can include:
        - Original features
        - Primary model's predicted probabilities
        - Primary model's entropy (uncertainty)
        
        Args:
            primary_predictions: Primary model side predictions
            true_labels: True TBM labels
            features: Optional additional features
        
        Returns:
            Tuple of (meta_features, meta_labels)
        """
        meta_labels = get_meta_labels(primary_predictions, true_labels)
        
        # Create meta-features
        meta_features = pd.DataFrame(index=meta_labels.index)
        
        # Include primary prediction as feature
        meta_features['primary_pred'] = primary_predictions.loc[meta_labels.index]
        meta_features['primary_abs'] = np.abs(meta_features['primary_pred'])
        
        # Merge additional features if provided
        if features is not None:
            common_idx = meta_labels.index.intersection(features.index)
            meta_features = meta_features.loc[common_idx]
            meta_labels = meta_labels.loc[common_idx]
            
            for col in features.columns:
                if col not in meta_features.columns:
                    meta_features[col] = features.loc[common_idx, col]
        
        return meta_features, meta_labels
    
    def apply(
        self,
        primary_side: pd.Series,
        meta_probability: pd.Series
    ) -> pd.DataFrame:
        """
        Apply meta-labeling for signal filtering and bet sizing.
        
        Args:
            primary_side: Primary model side predictions
            meta_probability: Meta-model P(correct) predictions
        
        Returns:
            DataFrame with filtered signals and bet sizes
        """
        return apply_meta_labeling(
            primary_side,
            meta_probability,
            min_confidence=self.min_confidence,
            bet_sizing=self.bet_sizing,
            kelly_fraction=self.kelly_fraction
        )
    
    def compute_metrics(
        self,
        signals: pd.DataFrame,
        true_labels: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        """
        Compute meta-labeling performance metrics.
        
        Args:
            signals: Output from apply() method
            true_labels: True TBM labels
            returns: Realized returns
        
        Returns:
            Dictionary with performance metrics
        """
        # Align indices
        common_idx = signals.index.intersection(true_labels.index)
        common_idx = common_idx.intersection(returns.index)
        
        sig = signals.loc[common_idx]
        labels = true_labels.loc[common_idx]
        rets = returns.loc[common_idx]
        
        # Filter performance
        taken = sig['take_trade']
        filtered_pct = (~taken).mean()
        
        # Precision: Of trades taken, how many were correct?
        if taken.sum() > 0:
            precision = (np.sign(sig.loc[taken, 'side']) == np.sign(labels.loc[taken])).mean()
        else:
            precision = 0.0
        
        # Weighted returns
        weighted_returns = sig['position'] * rets
        total_return = weighted_returns.sum()
        sharpe = weighted_returns.mean() / weighted_returns.std() * np.sqrt(252 * 24)  # Hourly
        
        # Compare to unfiltered
        unfiltered_returns = sig['side'] * rets
        unfiltered_sharpe = unfiltered_returns.mean() / unfiltered_returns.std() * np.sqrt(252 * 24)
        
        return {
            'filtered_pct': filtered_pct,
            'precision': precision,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'unfiltered_sharpe': unfiltered_sharpe,
            'sharpe_improvement': sharpe - unfiltered_sharpe,
            'avg_bet_size': sig.loc[taken, 'bet_size'].mean() if taken.sum() > 0 else 0,
            'num_trades': taken.sum(),
        }


def create_sequential_bootstrap_weights(
    labels: pd.DataFrame,
    num_samples: int
) -> np.ndarray:
    """
    Generate sample weights for sequential bootstrap.
    
    Standard bootstrap is biased for time series because overlapping
    labels cause samples to be non-independent. Sequential bootstrap
    downweights samples based on label overlap.
    
    Args:
        labels: DataFrame with 't0' (start) and 't1' (end) columns
        num_samples: Number of bootstrap samples to generate
    
    Returns:
        Array of sample weights
        
    Reference: Lopez de Prado, Ch. 4.5
    """
    # Compute indicator matrix (which samples overlap with which)
    n = len(labels)
    t0 = labels['t0'] if 't0' in labels.columns else labels.index
    t1 = labels['t1']
    
    # Average uniqueness of each sample
    uniqueness = np.zeros(n)
    
    for i in range(n):
        # Find overlapping samples
        overlap = ((t0 <= t1.iloc[i]) & (t1 >= t0.iloc[i])).sum()
        uniqueness[i] = 1.0 / max(overlap, 1)
    
    # Normalize to probabilities
    weights = uniqueness / uniqueness.sum()
    
    return weights


def avg_uniqueness(labels: pd.DataFrame) -> float:
    """
    Compute average uniqueness of samples.
    
    Uniqueness measures how much each sample's label period overlaps
    with other samples. Higher uniqueness = more independent samples.
    
    Args:
        labels: DataFrame with t0 (start) and t1 (end) timestamps
    
    Returns:
        Average uniqueness [0, 1]
    """
    weights = create_sequential_bootstrap_weights(labels, 1)
    return weights.mean() * len(labels)

