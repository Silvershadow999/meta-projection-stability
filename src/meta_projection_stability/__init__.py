from .config import MetaProjectionStabilityConfig
from .adapter import MetaProjectionStabilityAdapter

__version__ = "0.1.0-dev"

# Basis-Exports (immer verfügbar)
__all__ = [
    "MetaProjectionStabilityConfig",
    "MetaProjectionStabilityAdapter",
]

# Optionale Governor-API (nur exportieren, wenn vorhanden)
try:
    from .governor import (
        GovernorState,
        governor_step,
        compute_autonomy_index,
        determine_mode,
        summarize_governor_history,
        compute_trinity_autonomy_histories,
    )

    __all__ += [
        "GovernorState",
        "governor_step",
        "compute_autonomy_index",
        "determine_mode",
        "summarize_governor_history",
        "compute_trinity_autonomy_histories",
    ]
except Exception:
    # Governor ist optional / evtl. ältere Version im Repo
    pass

# Optionale Integrity-Barometer-API (nur exportieren, wenn vorhanden)
try:
    from .integrity_barometer import (
        IntegrityBarometerConfig,
        IntegrityBarometerState,
        barometer_step,
        summarize_barometer_history,
    )

    __all__ += [
        "IntegrityBarometerConfig",
        "IntegrityBarometerState",
        "barometer_step",
        "summarize_barometer_history",
    ]
except Exception:
    # integrity_barometer.py evtl. noch nicht lokal synchron
    pass
