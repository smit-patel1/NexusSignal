"""
Signal generation modules for NexusSignal 2.0.
"""

from nexus2.signals.generator import (
    SignalGenerator,
    convert_proba_to_signal,
    compute_expected_payoff,
)
from nexus2.signals.sizing import (
    PositionSizer,
    kelly_fraction,
    volatility_scaled_position,
)

