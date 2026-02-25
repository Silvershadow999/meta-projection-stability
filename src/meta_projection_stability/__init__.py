"""
meta_projection_stability package exports
"""

from .config import MetaProjectionStabilityConfig, get_default_config, print_config
from .simulation import run_simulation

# Plotting ist optional (damit Package-Import nicht crasht, wenn plotting API abweicht)
try:
    from .plotting import plot_results, print_summary
except ImportError:
    plot_results = None
    print_summary = None

__all__ = [
    "MetaProjectionStabilityConfig",
    "get_default_config",
    "print_config",
    "run_simulation",
    "plot_results",
    "print_summary",
    "ExperimentBatchConfig",
    "run_experiment_batch",
    "print_experiment_batch_summary",
    "save_experiment_batch_json",
    "save_experiment_batch_csv",
    "ProfileSpec",
    "get_profile_specs",
    "list_profiles",
    "describe_profiles",
    "apply_profile",
    "make_profile_config",
]


from .experiment_runner import (
    ExperimentBatchConfig,
    run_experiment_batch,
    print_experiment_batch_summary,
    save_experiment_batch_json,
    save_experiment_batch_csv,
)


from .profiles import (
    ProfileSpec,
    get_profile_specs,
    list_profiles,
    describe_profiles,
    apply_profile,
    make_profile_config,
)
