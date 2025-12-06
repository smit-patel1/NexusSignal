"""
Validation modules for NexusSignal 2.0.
"""

from nexus2.validation.cpcv import (
    CombinatorialPurgedCV,
    PurgedKFold,
    compute_purge_embargo,
    get_train_times,
)
from nexus2.validation.metrics import (
    precision_at_k,
    brier_score,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    probability_of_barrier_hit,
    compute_all_metrics,
)

