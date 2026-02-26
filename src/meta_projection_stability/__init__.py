"""
meta_projection_stability package exports
"""

from .config import MetaProjectionStabilityConfig, get_default_config, print_config
from .adapter import MetaProjectionStabilityAdapter
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
from .adversarial import run_adversarial_scenario, run_all_scenarios
from .axiom_handshake import AxiomHandshakeModule
from .axiom_gateway import AxiomCompatibilityGateway

# Optional plotting exports
try:
    from .plotting import plot_results, print_summary
except Exception:
    plot_results = None
    print_summary = None

# Optional adversarial exports
try:
    from .adversarial import run_adversarial_scenario, run_all_scenarios
except Exception:
    run_adversarial_scenario = None
    run_all_scenarios = None

# Optional axiom handshake / gateway exports
try:
    from .axiom_handshake import AxiomHandshakeModule
except Exception:
    AxiomHandshakeModule = None

try:
    from .axiom_gateway import AxiomCompatibilityGateway
except Exception:
    AxiomCompatibilityGateway = None

__all__ = [
    "MetaProjectionStabilityConfig",
    "get_default_config",
    "print_config",
    "MetaProjectionStabilityAdapter",
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
<<<<<<< HEAD
    "run_adversarial_scenario",
    "run_all_scenarios",
=======
    # adversarial
    "run_adversarial_scenario",
    "run_all_scenarios",
    # axiom / compatibility
>>>>>>> origin/main
    "AxiomHandshakeModule",
    "AxiomCompatibilityGateway",
]
