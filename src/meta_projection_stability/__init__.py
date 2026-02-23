from .config import MetaProjectionStabilityConfig
from .adapter import MetaProjectionStabilityAdapter

from .governor import (
    GovernorState,
    compute_autonomy_index,
    determine_mode,
    governor_step,
    governor_step_from_config,
)

from .integrity_barometer import (
    IntegrityBarometerConfig,
    IntegrityBarometerState,
    barometer_step,
)

__version__ = "0.1.0-dev"

__all__ = [
    # Core
    "MetaProjectionStabilityConfig",
    "MetaProjectionStabilityAdapter",
    # Governor
    "GovernorState",
    "compute_autonomy_index",
    "determine_mode",
    "governor_step",
    "governor_step_from_config",
    # Integrity barometer
    "IntegrityBarometerConfig",
    "IntegrityBarometerState",
    "barometer_step",
]
