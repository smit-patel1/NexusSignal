"""
Feature engineering modules for NexusSignal 2.0.
"""

from nexus2.features.microstructure import (
    compute_vpin,
    compute_kyle_lambda,
    compute_roll_spread,
    compute_amihud,
    compute_order_flow_imbalance,
    build_microstructure_features,
)
from nexus2.features.entropy import (
    approximate_entropy,
    sample_entropy,
    permutation_entropy,
    rolling_entropy,
    build_entropy_features,
)
from nexus2.features.regime import (
    fit_hmm,
    get_regime_probabilities,
    HMMRegimeDetector,
    build_regime_features,
)
from nexus2.features.builder import FeatureBuilder

