"""
Utility functions for model training and evaluation.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data chronologically without shuffling.

    Args:
        df: DataFrame with datetime index
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def create_sequences(
    data: np.ndarray,
    lookback: int,
    include_last: bool = True
) -> np.ndarray:
    """
    Create sliding window sequences for time series models.

    Args:
        data: 2D array (timesteps, features)
        lookback: Number of past timesteps to include
        include_last: If True, include final incomplete sequence

    Returns:
        3D array (samples, lookback, features)
    """
    sequences = []

    for i in range(len(data) - lookback + 1):
        sequences.append(data[i:i + lookback])

    return np.array(sequences)


def create_sequence_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequence dataset for LSTM/TCN training.

    Args:
        X: Feature dataframe
        y: Target series
        lookback: Sequence length

    Returns:
        Tuple of (X_sequences, y_aligned)
        - X_sequences: shape (samples, lookback, n_features)
        - y_aligned: shape (samples,) - targets aligned with last timestep of each sequence
    """
    X_values = X.values
    y_values = y.values

    # Create sequences
    X_seq = create_sequences(X_values, lookback)

    # Align targets with the last timestep of each sequence
    y_aligned = y_values[lookback - 1:]

    return X_seq, y_aligned


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics including financial-specific metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with MAE, RMSE, R2, directional accuracy, IC, and hit rates
    """
    from scipy.stats import spearmanr

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Directional accuracy: percentage of correct sign predictions
    directional_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    # FIX: Add financial metrics
    # Information Coefficient (Spearman correlation)
    ic, ic_pval = spearmanr(y_true, y_pred)

    # Hit rates for positive/negative returns
    pos_mask = y_true > 0
    neg_mask = y_true < 0
    hit_rate_pos = np.mean(y_pred[pos_mask] > 0) if pos_mask.sum() > 0 else 0.0
    hit_rate_neg = np.mean(y_pred[neg_mask] < 0) if neg_mask.sum() > 0 else 0.0

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'directional_accuracy': directional_acc,
        'information_coefficient': ic,
        'ic_pvalue': ic_pval,
        'hit_rate_positive': hit_rate_pos,
        'hit_rate_negative': hit_rate_neg,
    }


def print_metrics(metrics: dict, model_name: str, prefix: str = "") -> None:
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
        prefix: Optional prefix for indentation
    """
    print(f"{prefix}{model_name}:")
    print(f"{prefix}  MAE: {metrics['mae']:.6f}")
    print(f"{prefix}  RMSE: {metrics['rmse']:.6f}")
    print(f"{prefix}  R²: {metrics['r2']:.4f}")
    print(f"{prefix}  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    # FIX: Print financial metrics
    if 'information_coefficient' in metrics:
        print(f"{prefix}  IC: {metrics['information_coefficient']:.4f} (p={metrics['ic_pvalue']:.4f})")
        print(f"{prefix}  Hit Rate+: {metrics['hit_rate_positive']:.2%}, Hit Rate-: {metrics['hit_rate_negative']:.2%}")
