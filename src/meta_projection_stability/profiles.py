from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .config import MetaProjectionStabilityConfig


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    description: str
    updates: Dict[str, Any]


def get_profile_specs() -> Dict[str, ProfileSpec]:
    """
    Curated behavior profiles for comparative experiments.
    These are intentionally modest deltas (not wild retunes).
    """
    return {
        "balanced": ProfileSpec(
            name="balanced",
            description="General-purpose default balance of caution and recovery.",
            updates={
                # close to current defaults; kept explicit for reproducibility
                "risk_warn": 0.65,
                "risk_critical": 0.85,
                "risk_recover": 0.55,
                "trust_decay": 0.01,
                "trust_gain": 0.02,
                "risk_trust_damping_max": 0.35,
                "ema_alpha_risk": 0.12,
                "ema_alpha_human": 0.09,
                "cooldown_steps_after_reset": 10,
                "nominal_recovery_boost_factor": 1.00,
                "momentum_risk_weight": 0.10,
            },
        ),
        "protective": ProfileSpec(
            name="protective",
            description="More cautious, earlier blocking, slower trust recovery, longer cooldown.",
            updates={
                "risk_warn": 0.58,
                "risk_critical": 0.80,
                "risk_recover": 0.50,
                "trust_decay": 0.013,
                "trust_gain": 0.014,
                "risk_trust_damping_max": 0.40,
                "ema_alpha_risk": 0.16,
                "ema_alpha_human": 0.11,
                "cooldown_steps_after_reset": 14,
                "nominal_recovery_boost_factor": 0.90,
                "momentum_risk_weight": 0.14,
                "interestingness_warn": 0.50,
            },
        ),
        "aggressive_recovery": ProfileSpec(
            name="aggressive_recovery",
            description="Faster recovery and shorter cooldown; useful to test throughput vs safety trade-offs.",
            updates={
                "risk_warn": 0.68,
                "risk_critical": 0.87,
                "risk_recover": 0.60,
                "trust_decay": 0.009,
                "trust_gain": 0.030,
                "risk_trust_damping_max": 0.30,
                "ema_alpha_risk": 0.10,
                "ema_alpha_human": 0.08,
                "cooldown_steps_after_reset": 6,
                "nominal_recovery_boost_factor": 1.18,
                "momentum_risk_weight": 0.08,
            },
        ),
        "strict_safety": ProfileSpec(
            name="strict_safety",
            description="Strong safety bias: early blocks, stronger momentum sensitivity, conservative recovery.",
            updates={
                "risk_warn": 0.52,
                "risk_critical": 0.76,
                "risk_recover": 0.46,
                "trust_decay": 0.016,
                "trust_gain": 0.010,
                "risk_trust_damping_max": 0.45,
                "ema_alpha_risk": 0.20,
                "ema_alpha_human": 0.12,
                "cooldown_steps_after_reset": 18,
                "nominal_recovery_boost_factor": 0.82,
                "momentum_risk_weight": 0.18,
                "interestingness_warn": 0.55,
                "interestingness_critical": 0.35,
            },
        ),
    }


def list_profiles() -> List[str]:
    return sorted(get_profile_specs().keys())


def describe_profiles() -> Dict[str, str]:
    specs = get_profile_specs()
    return {name: spec.description for name, spec in specs.items()}


def apply_profile(
    cfg: MetaProjectionStabilityConfig,
    profile_name: str = "balanced",
    *,
    set_profile_name_field: bool = True,
) -> MetaProjectionStabilityConfig:
    """
    Apply profile updates onto an existing config via cfg.update(...) so clamping and aliases re-run.
    """
    specs = get_profile_specs()
    if profile_name not in specs:
        raise ValueError(f"Unknown profile '{profile_name}'. Available: {sorted(specs.keys())}")

    spec = specs[profile_name]
    cfg.update(**spec.updates)

    if set_profile_name_field and hasattr(cfg, "profile_name"):
        try:
            cfg.profile_name = spec.name
        except Exception:
            pass

    return cfg


def make_profile_config(
    profile_name: str = "balanced",
    **base_cfg_kwargs: Any,
) -> MetaProjectionStabilityConfig:
    """
    Construct a fresh config and apply a profile in one step.
    """
    cfg = MetaProjectionStabilityConfig(**base_cfg_kwargs)
    return apply_profile(cfg, profile_name=profile_name)


if __name__ == "__main__":
    print("Available profiles:")
    for name, desc in describe_profiles().items():
        print(f" - {name}: {desc}")

    cfg = make_profile_config("protective", seed=42, enable_plot=False)
    print("\nApplied profile:", getattr(cfg, "profile_name", "<none>"))
    print("risk_warn / risk_critical / trust_gain / cooldown_steps_after_reset =",
          cfg.risk_warn, cfg.risk_critical, cfg.trust_gain, cfg.cooldown_steps_after_reset)
