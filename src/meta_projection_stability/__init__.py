"""
meta_projection_stability package exports
"""

from .config import MetaProjectionStabilityConfig, get_default_config, print_config
from .simulation import run_simulation
from .analytics import (
    calculate_safety_score,
    summarize_risk_profile,
    print_risk_summary,
    print_enhanced_summary,
)
from .level0_axiom import (
    DecisionStatus,
    ActionProposal,
    Level0Thresholds,
    Level0Decision,
    Level0AxiomEngine,
)
from .adapter import MetaProjectionStabilityAdapter

try:
    from .plotting import plot_results, print_summary
except Exception:
    plot_results = None
    print_summary = None

try:
    from .adversarial import run_adversarial_scenario, run_all_scenarios
except Exception:
    run_adversarial_scenario = None
    run_all_scenarios = None

__all__ = [
    "MetaProjectionStabilityConfig",
    "get_default_config",
    "print_config",
    "run_simulation",
    "plot_results",
    "print_summary",
    "calculate_safety_score",
    "summarize_risk_profile",
    "print_risk_summary",
    "print_enhanced_summary",
    "DecisionStatus",
    "ActionProposal",
    "Level0Thresholds",
    "Level0Decision",
    "Level0AxiomEngine",
    "MetaProjectionStabilityAdapter",
    "run_adversarial_scenario",
    "run_all_scenarios",
]
# --- version surface ---
try:
    from importlib.metadata import version as _pkg_version  # py3.8+
    __version__ = _pkg_version("meta-projection-stability")
except Exception:
    __version__ = "0.0.0"
