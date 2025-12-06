"""
Financial ML Evaluation Metrics

Standard ML metrics (accuracy, MSE) are insufficient for trading systems.
This module implements finance-specific metrics:

- Precision@K: Accuracy of top-K predictions
- Brier Score: Calibration of probability predictions
- Deflated Sharpe Ratio: Sharpe adjusted for multiple testing
- Probability of Barrier Hit: Accuracy of Triple Barrier predictions

Reference: Lopez de Prado, Bailey et al.
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd
from scipy.stats import norm


def precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
    pos_label: int = 1
) -> float:
    """
    Precision@K: Accuracy among top-K predicted samples.
    
    In trading, we often only act on the strongest signals.
    Precision@K measures: "Of my top K predictions, how many were correct?"
    
    Args:
        y_true: True labels (0 or 1)
        y_score: Predicted scores/probabilities
        k: Number of top predictions to evaluate
        pos_label: Positive label value
    
    Returns:
        Precision among top-k predictions
        
    Example:
        If k=10 and 7 of the top 10 predictions were correct:
        Precision@10 = 0.7
    """
    if len(y_true) < k:
        k = len(y_true)
    
    # Get indices of top-k scores
    top_k_idx = np.argsort(y_score)[-k:]
    
    # Count correct predictions in top-k
    n_correct = np.sum(y_true[top_k_idx] == pos_label)
    
    return n_correct / k


def brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """
    Brier Score for probability calibration.
    
    Measures how well predicted probabilities match actual outcomes.
    
    Formula:
        BS = (1/N) * sum((p_i - o_i)^2)
        
    where p_i is predicted probability and o_i is actual outcome.
    
    Args:
        y_true: True binary outcomes (0 or 1)
        y_prob: Predicted probabilities [0, 1]
    
    Returns:
        Brier score [0, 1] where 0 is perfect calibration
        
    Interpretation:
    - BS < 0.1: Excellent calibration
    - BS 0.1-0.2: Good calibration
    - BS > 0.25: Poor calibration
    """
    return np.mean((y_prob - y_true) ** 2)


def brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """
    Brier Skill Score: Improvement over climatological baseline.
    
    BSS = 1 - BS / BS_climatology
    
    where BS_climatology = p * (1-p) for base rate p.
    
    Returns:
        Skill score where 1 = perfect, 0 = baseline, <0 = worse than baseline
    """
    bs = brier_score(y_true, y_prob)
    base_rate = np.mean(y_true)
    bs_baseline = base_rate * (1 - base_rate)
    
    return 1 - bs / (bs_baseline + 1e-10)


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int = 1,
    annualization: float = np.sqrt(252 * 24),  # Hourly
    expected_max_sr: Optional[float] = None
) -> float:
    """
    Deflated Sharpe Ratio (DSR).
    
    Standard Sharpe ratio is biased upward when:
    1. Multiple strategies are tested
    2. Best performer is selected
    
    DSR adjusts for this selection bias using the expected maximum
    of n_trials random Sharpe ratios.
    
    Formula:
        DSR = SR * (1 - gamma / SR_expected_max)
        
    where gamma is the Euler-Mascheroni constant ≈ 0.5772
    
    Args:
        returns: Strategy returns
        n_trials: Number of strategies/backtests tried
        annualization: Annualization factor
        expected_max_sr: Expected max SR (auto-computed if None)
    
    Returns:
        Deflated Sharpe ratio
        
    Reference: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
    """
    # Standard Sharpe
    sr = np.mean(returns) / (np.std(returns) + 1e-10) * annualization
    
    if n_trials <= 1:
        return sr
    
    # Expected maximum of n independent standard normals
    # E[max] ≈ sqrt(2 * log(n)) - (gamma + log(sqrt(log(n)))) / sqrt(2 * log(n))
    if expected_max_sr is None:
        gamma = 0.5772156649  # Euler-Mascheroni constant
        expected_max_sr = (
            np.sqrt(2 * np.log(n_trials)) -
            (gamma + np.log(np.sqrt(np.log(n_trials)))) / np.sqrt(2 * np.log(n_trials))
        )
    
    # Deflation factor
    deflation = 1 - gamma / (expected_max_sr + 1e-10)
    
    return sr * max(deflation, 0)


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    annualization: float = np.sqrt(252 * 24)
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR).
    
    Computes the probability that the true Sharpe ratio exceeds a benchmark,
    accounting for estimation error due to limited sample size.
    
    Formula:
        PSR = Φ((SR - SR*) * sqrt(n-1) / sqrt(1 - γ3*SR + (γ4-1)/4 * SR^2))
        
    where γ3 is skewness, γ4 is kurtosis, and Φ is the standard normal CDF.
    
    Args:
        returns: Strategy returns
        benchmark_sr: Benchmark Sharpe ratio (default 0)
        annualization: Annualization factor
    
    Returns:
        Probability that true SR > benchmark_sr
        
    Interpretation:
    - PSR > 0.95: High confidence SR exceeds benchmark
    - PSR 0.5-0.95: Some evidence of skill
    - PSR < 0.5: No evidence of skill
    
    Reference: Bailey & Lopez de Prado (2012)
    """
    n = len(returns)
    
    # Moments
    sr = np.mean(returns) / (np.std(returns) + 1e-10) * annualization
    skew = pd.Series(returns).skew()
    kurt = pd.Series(returns).kurtosis() + 3  # Excess → raw kurtosis
    
    # Standard error of SR
    se_sr = np.sqrt(
        (1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1)
    )
    
    # Z-score
    z = (sr - benchmark_sr) / (se_sr + 1e-10)
    
    # Probability
    psr = norm.cdf(z)
    
    return psr


def probability_of_barrier_hit(
    y_true: np.ndarray,
    y_prob_upper: np.ndarray,
    y_prob_lower: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate Triple Barrier probability predictions.
    
    For TBM models, we predict probability of hitting upper (profit)
    vs lower (stop) barrier. This measures calibration of those predictions.
    
    Args:
        y_true: Actual barrier hit (1 = upper, -1 = lower, 0 = vertical)
        y_prob_upper: Predicted P(upper barrier hit)
        y_prob_lower: Predicted P(lower barrier hit)
    
    Returns:
        Dictionary with accuracy and calibration metrics
    """
    # Accuracy: Did we predict the right barrier?
    pred_upper = y_prob_upper > y_prob_lower
    actual_upper = y_true == 1
    actual_lower = y_true == -1
    
    # Only evaluate where a barrier was hit (not vertical)
    barrier_hit = (y_true != 0)
    
    if barrier_hit.sum() == 0:
        return {'accuracy': np.nan, 'brier_upper': np.nan, 'brier_lower': np.nan}
    
    accuracy = (pred_upper[barrier_hit] == actual_upper[barrier_hit]).mean()
    
    # Brier scores for each barrier
    brier_upper = brier_score(actual_upper[barrier_hit].astype(float), y_prob_upper[barrier_hit])
    brier_lower = brier_score(actual_lower[barrier_hit].astype(float), y_prob_lower[barrier_hit])
    
    # Calibration: binned reliability
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    calibration_upper = []
    for i in range(n_bins):
        mask = (y_prob_upper >= bin_edges[i]) & (y_prob_upper < bin_edges[i + 1])
        if mask.sum() > 0:
            pred_mean = y_prob_upper[mask].mean()
            actual_mean = actual_upper[mask].mean()
            calibration_upper.append(abs(pred_mean - actual_mean))
    
    avg_calibration_error = np.mean(calibration_upper) if calibration_upper else np.nan
    
    return {
        'accuracy': accuracy,
        'brier_upper': brier_upper,
        'brier_lower': brier_lower,
        'calibration_error': avg_calibration_error,
        'n_samples': barrier_hit.sum()
    }


def information_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Information Coefficient (IC): Spearman rank correlation.
    
    IC measures how well predicted ranks match actual ranks.
    Standard metric in quantitative finance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Spearman correlation [-1, 1]
    """
    from scipy.stats import spearmanr
    ic, _ = spearmanr(y_true, y_pred)
    return ic


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    returns: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    n_trials: int = 1
) -> Dict[str, float]:
    """
    Compute comprehensive set of financial ML metrics.
    
    Args:
        y_true: True labels or returns
        y_pred: Predicted labels or values
        y_prob: Optional probability predictions
        returns: Optional realized returns
        positions: Optional position sizes
        n_trials: Number of strategies tested (for DSR)
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Classification metrics (if binary labels)
    if len(np.unique(y_true[~np.isnan(y_true)])) <= 3:
        # Accuracy
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() > 0:
            metrics['accuracy'] = np.mean(y_true[valid] == y_pred[valid])
            
            # Precision at K
            for k in [5, 10, 20]:
                if y_prob is not None and valid.sum() >= k:
                    metrics[f'precision@{k}'] = precision_at_k(
                        (y_true[valid] == 1).astype(int),
                        y_prob[valid],
                        k=k
                    )
    
    # Regression metrics
    if y_prob is None:
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() > 0:
            # Information Coefficient
            metrics['ic'] = information_coefficient(y_true[valid], y_pred[valid])
            
            # Directional accuracy
            metrics['directional_accuracy'] = np.mean(
                np.sign(y_true[valid]) == np.sign(y_pred[valid])
            )
    
    # Probability calibration
    if y_prob is not None:
        binary_true = (y_true == 1).astype(float)
        valid = ~(np.isnan(binary_true) | np.isnan(y_prob))
        if valid.sum() > 0:
            metrics['brier_score'] = brier_score(binary_true[valid], y_prob[valid])
            metrics['brier_skill'] = brier_skill_score(binary_true[valid], y_prob[valid])
    
    # Strategy performance metrics
    if returns is not None and positions is not None:
        strategy_returns = returns * positions
        valid = ~np.isnan(strategy_returns)
        if valid.sum() > 10:
            # Sharpe ratio
            sr = np.mean(strategy_returns[valid]) / (np.std(strategy_returns[valid]) + 1e-10)
            metrics['sharpe_ratio'] = sr * np.sqrt(252 * 24)  # Annualized
            
            # Deflated Sharpe
            metrics['deflated_sharpe'] = deflated_sharpe_ratio(
                strategy_returns[valid], n_trials=n_trials
            )
            
            # Probabilistic Sharpe
            metrics['probabilistic_sharpe'] = probabilistic_sharpe_ratio(
                strategy_returns[valid]
            )
            
            # Max drawdown
            cumulative = np.cumsum(strategy_returns[valid])
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = cumulative - running_max
            metrics['max_drawdown'] = np.min(drawdowns)
            
            # Sortino ratio
            downside = strategy_returns[strategy_returns < 0]
            downside_std = np.std(downside) if len(downside) > 0 else 1e-10
            metrics['sortino_ratio'] = np.mean(strategy_returns[valid]) / downside_std * np.sqrt(252 * 24)
    
    return metrics

