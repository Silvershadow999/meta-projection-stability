"""
meta_projection_stability package exports
"""

from .config import MetaProjectionStabilityConfig, get_default_config, print_config
from .adapter import MetaProjectionStabilityAdapter
from .simulation import run_simulation
from .plotting import plot_results, print_summary
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
from .adversarial import run_adversarial_scenario, run_all_scenarios
from .axiom_handshake import AxiomHandshakeModule
from .axiom_gateway import AxiomCompatibilityGateway

__all__ = [
    "MetaProjectionStabilityConfig",
    "get_default_config",
    "print_config",
    "MetaProjectionStabilityAdapter",
    "run_simulation",
    "plot_results",
    "print_summary",
    # analytics
    "calculate_safety_score",
    "summarize_risk_profile",
    "print_risk_summary",
    "print_enhanced_summary",
    # level0
    "DecisionStatus",
    "ActionProposal",
    "Level0Thresholds",
    "Level0Decision",
    "Level0AxiomEngine",
    # adversarial
    "run_adversarial_scenario",
    "run_all_scenarios",
    # axiom / compatibility
    "AxiomHandshakeModule",
    "AxiomCompatibilityGateway",
]
