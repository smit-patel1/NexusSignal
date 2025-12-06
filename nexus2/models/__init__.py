"""
Model architectures for NexusSignal 2.0.
"""

from nexus2.models.quantile_nn import (
    QuantileRegressionNN,
    QuantileLoss,
    create_quantile_model,
)
from nexus2.models.mdn import (
    MixtureDensityNetwork,
    MDNLoss,
    create_mdn_model,
)
from nexus2.models.classifier import (
    BarrierClassifier,
    create_barrier_classifier,
)

