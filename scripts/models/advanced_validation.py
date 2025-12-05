"""
Advanced validation strategies for time series.

Includes:
- Rolling expanding window validation
- Blocked walk-forward validation
- Purged/embargoed cross-validation
- Time series specific metrics
"""

from typing import List, Tuple, Generator
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TimeSeriesSplit:
    """
    Advanced time series cross-validation with multiple strategies.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = None,
        gap: int = 0,
        strategy: str = "expanding",
    ):
        """
        Initialize time series splitter.

        Args:
            n_splits: Number of splits
            test_size: Size of test set (if None, determined automatically)
            gap: Gap between train and test to prevent leakage
            strategy: 'expanding' (growing train) or 'rolling' (fixed train size)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.strategy = strategy

    def split(
        self, X: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for time series splits.

        Args:
            X: Input data (to get length)

        Yields:
            (train_indices, test_indices) tuples
        """
        n_samples = len(X)

        # Determine test size
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        # Calculate split points
        for i in range(self.n_splits):
            # Test set end
            test_end = n_samples - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size

            if test_start <= 0:
                continue

            if self.strategy == "expanding":
                # Expanding window: train on all data up to gap before test
                train_end = test_start - self.gap
                train_start = 0
            elif self.strategy == "rolling":
                # Rolling window: fixed size train window
                train_end = test_start - self.gap
                # Make train size equal to test size
                train_start = max(0, train_end - test_size)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            if train_start >= train_end:
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices


class BlockedWalkForward:
    """
    Blocked walk-forward validation with purging and embargo.
    """

    def __init__(
        self,
        n_blocks: int = 10,
        test_ratio: float = 0.2,
        purge_pct: float = 0.02,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize blocked walk-forward validator.

        Args:
            n_blocks: Number of blocks to create
            test_ratio: Ratio of data to use for testing in each fold
            purge_pct: Percentage of data to purge before test set
            embargo_pct: Percentage of data to embargo after test set
        """
        self.n_blocks = n_blocks
        self.test_ratio = test_ratio
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    def split(
        self, X: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate purged/embargoed train/test indices.

        Args:
            X: Input data

        Yields:
            (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        block_size = n_samples // self.n_blocks

        for i in range(1, self.n_blocks):
            # Test set: block i
            test_start = i * block_size
            test_end = min(test_start + int(block_size * self.test_ratio), n_samples)

            # Purge: remove samples immediately before test
            purge_size = int(n_samples * self.purge_pct)
            purge_start = max(0, test_start - purge_size)

            # Embargo: remove samples immediately after test
            embargo_size = int(n_samples * self.embargo_pct)
            embargo_end = min(test_end + embargo_size, n_samples)

            # Train set: all data before purge
            train_indices = np.arange(0, purge_start)

            # Test set
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


def expanding_window_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_fn,
    min_train_size: int = 100,
    step_size: int = 50,
    horizon: int = 1,
) -> Tuple[List[dict], pd.DataFrame]:
    """
    Perform expanding window validation.

    Args:
        X: Features
        y: Targets
        model_fn: Function that returns a fitted model (callable)
        min_train_size: Minimum training size to start
        step_size: Step size for expanding window
        horizon: Prediction horizon

    Returns:
        List of fold metrics and DataFrame of predictions
    """
    n_samples = len(X)
    fold_metrics = []
    all_predictions = []

    train_end = min_train_size

    while train_end + step_size + horizon <= n_samples:
        # Training data: from start to train_end
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        # Test data: next step_size samples after train_end
        test_start = train_end
        test_end = min(train_end + step_size, n_samples)

        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        # Fit model
        model = model_fn()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Compute metrics
        metrics = compute_fold_metrics(y_test, y_pred)
        metrics["train_size"] = len(X_train)
        metrics["test_size"] = len(X_test)
        fold_metrics.append(metrics)

        # Store predictions
        pred_df = pd.DataFrame(
            {
                "y_true": y_test,
                "y_pred": y_pred,
                "fold": len(fold_metrics),
            },
            index=y_test.index,
        )
        all_predictions.append(pred_df)

        # Expand window
        train_end += step_size

    predictions_df = pd.concat(all_predictions)

    return fold_metrics, predictions_df


def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute metrics for a single fold.

    Args:
        y_true: True targets
        y_pred: Predicted targets

    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Directional accuracy
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    # Information coefficient (correlation)
    ic = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0

    # Mean absolute percentage error (handle zeros)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-10, None))) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": dir_acc,
        "information_coefficient": ic,
        "mape": mape,
    }


def compute_cross_val_metrics(fold_metrics: List[dict]) -> dict:
    """
    Aggregate metrics across folds.

    Args:
        fold_metrics: List of fold metric dictionaries

    Returns:
        Aggregated metrics
    """
    if not fold_metrics:
        return {}

    # Convert to DataFrame for easier aggregation
    df = pd.DataFrame(fold_metrics)

    agg_metrics = {
        "mae_mean": df["mae"].mean(),
        "mae_std": df["mae"].std(),
        "rmse_mean": df["rmse"].mean(),
        "rmse_std": df["rmse"].std(),
        "r2_mean": df["r2"].mean(),
        "r2_std": df["r2"].std(),
        "dir_acc_mean": df["directional_accuracy"].mean(),
        "dir_acc_std": df["directional_accuracy"].std(),
        "ic_mean": df["information_coefficient"].mean(),
        "ic_std": df["information_coefficient"].std(),
        "n_folds": len(fold_metrics),
    }

    return agg_metrics


def walk_forward_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    model_class,
    param_grid: dict,
    n_splits: int = 5,
    gap: int = 0,
) -> Tuple[dict, List[dict]]:
    """
    Walk-forward optimization for hyperparameter tuning.

    Args:
        X: Features
        y: Targets
        model_class: Model class to instantiate
        param_grid: Parameter grid to search
        n_splits: Number of walk-forward splits
        gap: Gap between train and test

    Returns:
        Best parameters and all trial results
    """
    from itertools import product

    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, strategy="expanding")

    trial_results = []

    for param_combo in param_combinations:
        params = dict(zip(param_names, param_combo))

        fold_metrics = []

        for train_idx, test_idx in tscv.split(X):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # Train model with these params
            try:
                model = model_class(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = compute_fold_metrics(y_test, y_pred)
                fold_metrics.append(metrics)

            except Exception as e:
                warnings.warn(f"Model failed with params {params}: {e}")
                continue

        if fold_metrics:
            agg_metrics = compute_cross_val_metrics(fold_metrics)
            agg_metrics["params"] = params
            trial_results.append(agg_metrics)

    # Find best parameters (by mean R²)
    if not trial_results:
        return {}, []

    best_trial = max(trial_results, key=lambda x: x.get("r2_mean", -np.inf))
    best_params = best_trial["params"]

    return best_params, trial_results


def check_overfitting(
    train_metrics: dict, test_metrics: dict, threshold: float = 0.2
) -> dict:
    """
    Check for overfitting by comparing train and test metrics.

    Args:
        train_metrics: Metrics on training set
        test_metrics: Metrics on test set
        threshold: Threshold for overfitting detection (R² difference)

    Returns:
        Dictionary with overfitting flags
    """
    r2_diff = train_metrics.get("r2", 0) - test_metrics.get("r2", 0)
    mae_ratio = test_metrics.get("mae", 1) / max(train_metrics.get("mae", 1), 1e-10)

    return {
        "r2_train": train_metrics.get("r2"),
        "r2_test": test_metrics.get("r2"),
        "r2_diff": r2_diff,
        "mae_ratio": mae_ratio,
        "is_overfitting": r2_diff > threshold,
        "overfitting_severity": "high" if r2_diff > 0.3 else "moderate" if r2_diff > 0.2 else "low",
    }
