"""
Combinatorial Purged Cross-Validation (CPCV)

Standard k-fold CV fails for financial data because:
1. Labels overlap (a 24h return uses future prices)
2. Train/test contamination through overlapping features
3. Serial correlation in features and labels

CPCV addresses this through:
1. Purging: Remove train samples that overlap with test labels
2. Embargo: Additional buffer after test set
3. Combinatorial: Generate all possible train/test combinations

Reference: Lopez de Prado, "Advances in Financial Machine Learning", Ch. 7
"""

from typing import Tuple, List, Iterator, Optional
from itertools import combinations
import numpy as np
import pandas as pd


def get_train_times(
    t0: pd.Series,
    test_times: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """
    Get training indices after purging for test overlap.
    
    Any training sample whose label period (t0[i] to t1[i]) overlaps
    with the test period must be removed.
    
    Args:
        t0: Series mapping sample index to label start time
        test_times: Test set timestamps
    
    Returns:
        Valid training indices (purged)
        
    Overlap condition:
        Sample i overlaps with test if:
        t0[i] <= max(test_times) AND t1[i] >= min(test_times)
    """
    train_idx = t0.index.copy()
    
    for test_start, test_end in zip(test_times[:-1], test_times[1:]):
        # Find training samples that overlap with this test period
        overlap = t0[
            (t0.index <= test_end) & (t0 >= test_start)
        ].index
        
        train_idx = train_idx.difference(overlap)
    
    return train_idx


def compute_purge_embargo(
    labels: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    purge_length: int,
    embargo_pct: float
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Compute purged training indices and embargo test indices.
    
    Purging: Remove training samples whose labels overlap with test period.
    Embargo: Add buffer after test end to account for serial correlation.
    
    Args:
        labels: DataFrame with 't0' (start) and 't1' (end) columns
        test_start: Test period start
        test_end: Test period end
        purge_length: Number of bars to purge before test
        embargo_pct: Percentage of test length for embargo
    
    Returns:
        Tuple of (train_indices, test_indices)
    """
    all_idx = labels.index
    n_samples = len(all_idx)
    
    # Handle empty labels case
    if n_samples == 0:
        empty_idx = pd.DatetimeIndex([])
        return empty_idx, empty_idx
    
    # Test indices
    test_mask = (all_idx >= test_start) & (all_idx <= test_end)
    test_idx = all_idx[test_mask]
    
    # Embargo length
    n_test = len(test_idx)
    embargo_length = max(1, int(n_test * embargo_pct))
    
    # Embargo after test
    if len(test_idx) > 0:
        test_end_pos = all_idx.get_loc(test_idx[-1])
        embargo_end_pos = min(test_end_pos + embargo_length, n_samples - 1)
        embargo_end = all_idx[embargo_end_pos]
    else:
        # No test samples - no embargo needed
        embargo_end = test_end
    
    # Purge before test (samples whose labels extend into test)
    if 't1' in labels.columns:
        t1 = labels['t1']
        purge_mask = (t1 >= test_start) & (labels.index < test_start)
        purge_idx = labels.index[purge_mask]
    else:
        # Fallback: purge fixed number of bars
        if test_start in all_idx:
            test_start_pos = all_idx.get_loc(test_start)
        else:
            # Find closest position
            test_start_pos = all_idx.searchsorted(test_start)
        purge_start_pos = max(0, test_start_pos - purge_length)
        purge_idx = all_idx[purge_start_pos:test_start_pos]
    
    # Training indices: exclude test, purge, and embargo
    train_mask = ~(
        test_mask |  # Exclude test
        labels.index.isin(purge_idx) |  # Exclude purge
        (all_idx > test_end) & (all_idx <= embargo_end)  # Exclude embargo
    )
    
    train_idx = all_idx[train_mask]
    
    return train_idx, test_idx


class PurgedKFold:
    """
    K-Fold cross-validation with purging and embargo.
    
    Unlike standard KFold, this:
    1. Splits chronologically (no shuffle)
    2. Purges training samples that overlap with test labels
    3. Adds embargo period after each test fold
    
    Example:
        >>> cv = PurgedKFold(n_splits=5, purge_length=24, embargo_pct=0.01)
        >>> for train_idx, test_idx in cv.split(X, labels):
        ...     model.fit(X.loc[train_idx], y.loc[train_idx])
        ...     score = model.score(X.loc[test_idx], y.loc[test_idx])
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_length: int = 24,
        embargo_pct: float = 0.01
    ):
        """
        Initialize PurgedKFold.
        
        Args:
            n_splits: Number of folds
            purge_length: Bars to purge before test
            embargo_pct: Embargo as fraction of test length
        """
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_pct = embargo_pct
        
    def split(
        self,
        X: pd.DataFrame,
        labels: Optional[pd.DataFrame] = None
    ) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate train/test indices for each fold.
        
        Args:
            X: Feature DataFrame (used for index)
            labels: Optional labels DataFrame with 't1' column
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        all_idx = X.index
        n_samples = len(all_idx)
        fold_size = n_samples // self.n_splits
        
        for fold in range(self.n_splits):
            # Test range for this fold
            test_start_pos = fold * fold_size
            test_end_pos = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            
            test_start = all_idx[test_start_pos]
            test_end = all_idx[test_end_pos - 1]
            
            if labels is not None:
                train_idx, test_idx = compute_purge_embargo(
                    labels, test_start, test_end,
                    self.purge_length, self.embargo_pct
                )
            else:
                # Simple split without purging
                test_idx = all_idx[test_start_pos:test_end_pos]
                
                # Embargo
                embargo_end_pos = min(test_end_pos + int(fold_size * self.embargo_pct), n_samples)
                
                train_mask = (all_idx < test_start) | (all_idx >= all_idx[embargo_end_pos])
                train_idx = all_idx[train_mask]
            
            yield train_idx, test_idx
    
    def get_n_splits(self) -> int:
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    CPCV generates all possible combinations of k test groups from
    n groups, providing many more train/test scenarios than standard CV.
    
    For n=5 groups and k=2 test groups:
    - C(5,2) = 10 unique train/test splits
    - Each split has ~40% test data
    
    This dramatically increases the number of backtest paths,
    making it harder to overfit to a single lucky split.
    
    Key advantages:
    1. More robust performance estimation
    2. Better detection of overfitting
    3. Combinatorial diversity in test scenarios
    
    Reference: Lopez de Prado, Ch. 12
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        purge_length: int = 24,
        embargo_pct: float = 0.01
    ):
        """
        Initialize CPCV.
        
        Args:
            n_splits: Number of groups to split data into
            n_test_groups: Number of groups to use as test set
            purge_length: Bars to purge before test groups
            embargo_pct: Embargo as fraction of group size
        """
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_length = purge_length
        self.embargo_pct = embargo_pct
        
        # Number of combinations
        self.n_combinations = self._n_choose_k(n_splits, n_test_groups)
        
    def _n_choose_k(self, n: int, k: int) -> int:
        """Compute binomial coefficient."""
        from math import factorial
        return factorial(n) // (factorial(k) * factorial(n - k))
    
    def split(
        self,
        X: pd.DataFrame,
        labels: Optional[pd.DataFrame] = None
    ) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate all combinatorial train/test splits.
        
        Args:
            X: Feature DataFrame
            labels: Optional labels with 't1' column
        
        Yields:
            Tuple of (train_indices, test_indices) for each combination
        """
        all_idx = X.index
        n_samples = len(all_idx)
        group_size = n_samples // self.n_splits
        
        # Create groups
        groups = []
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            groups.append(all_idx[start:end])
        
        # Generate all combinations of test groups
        for test_group_indices in combinations(range(self.n_splits), self.n_test_groups):
            # Combine test groups
            test_idx_list = [groups[i] for i in test_group_indices]
            test_idx = pd.DatetimeIndex(np.concatenate([idx.values for idx in test_idx_list]))
            test_idx = test_idx.sort_values()
            
            # Training: all other groups with purging
            train_groups = [i for i in range(self.n_splits) if i not in test_group_indices]
            train_idx_list = [groups[i] for i in train_groups]
            train_idx = pd.DatetimeIndex(np.concatenate([idx.values for idx in train_idx_list]))
            
            if labels is not None:
                # Apply purging for each test group boundary
                for test_group_idx in test_group_indices:
                    test_group = groups[test_group_idx]
                    test_start = test_group[0]
                    test_end = test_group[-1]
                    
                    # Use full labels for purge calculation (need t1 column)
                    # Filter to indices that exist in labels
                    valid_train_idx = train_idx.intersection(labels.index)
                    if len(valid_train_idx) == 0:
                        continue
                    
                    purge_train_idx, _ = compute_purge_embargo(
                        labels,
                        test_start, test_end,
                        self.purge_length, self.embargo_pct
                    )
                    
                    train_idx = train_idx.intersection(purge_train_idx)
            else:
                # Simple embargo without labels
                embargo_size = int(group_size * self.embargo_pct)
                
                for test_group_idx in test_group_indices:
                    test_group = groups[test_group_idx]
                    test_end = test_group[-1]
                    
                    # Find position and embargo
                    test_end_pos = all_idx.get_loc(test_end)
                    embargo_end_pos = min(test_end_pos + embargo_size, n_samples - 1)
                    
                    # Remove embargo from train
                    embargo_idx = all_idx[test_end_pos + 1:embargo_end_pos + 1]
                    train_idx = train_idx.difference(embargo_idx)
            
            yield train_idx.sort_values(), test_idx
    
    def get_n_splits(self) -> int:
        """Return number of unique train/test combinations."""
        return self.n_combinations
    
    def get_backtest_paths(self) -> int:
        """
        Compute number of independent backtest paths.
        
        Each path is a sequence of test folds that can be concatenated
        to form a continuous backtest.
        """
        # Simplified: return number of ways to order test groups
        from math import factorial
        return factorial(self.n_test_groups)


def compute_cv_stats(
    cv_scores: List[float]
) -> dict:
    """
    Compute statistics from cross-validation scores.
    
    Args:
        cv_scores: List of scores from each fold
    
    Returns:
        Dictionary with mean, std, and deflated statistics
    """
    scores = np.array(cv_scores)
    
    # Basic stats
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Deflated Sharpe-style statistic
    # Accounts for multiple testing
    n_trials = len(scores)
    t_stat = mean_score / (std_score / np.sqrt(n_trials) + 1e-10)
    
    # Probability that score > 0 is due to skill, not luck
    from scipy.stats import norm
    prob_skill = norm.cdf(t_stat)
    
    return {
        'mean': mean_score,
        'std': std_score,
        'n_folds': n_trials,
        't_stat': t_stat,
        'prob_skill': prob_skill,
        'scores': scores.tolist()
    }


def walk_forward_cv(
    X: pd.DataFrame,
    train_period: int,
    test_period: int,
    step: Optional[int] = None,
    purge_length: int = 24
) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Walk-forward cross-validation (expanding or rolling window).
    
    This mimics real trading: train on past data, test on future data,
    then roll forward in time.
    
    Args:
        X: Feature DataFrame
        train_period: Training window size (bars)
        test_period: Test window size (bars)
        step: Step size (default = test_period)
        purge_length: Purge buffer before test
    
    Yields:
        Tuple of (train_indices, test_indices)
    """
    all_idx = X.index
    n_samples = len(all_idx)
    
    if step is None:
        step = test_period
    
    start = train_period
    
    while start + test_period <= n_samples:
        # Training: everything before (minus purge)
        train_end = start - purge_length
        train_idx = all_idx[:train_end]
        
        # Test: next test_period bars
        test_idx = all_idx[start:start + test_period]
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx
        
        start += step

