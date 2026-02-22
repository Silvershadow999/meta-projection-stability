from dataclasses import dataclass


@dataclass
class MetaProjectionStabilityConfig:
    # Interestingness / Human anchor
    human_complexity_weight: float = 0.18
    interestingness_critical: float = 0.38
    interestingness_warning: float = 0.68

    # Human significance dynamics
    human_recovery_base: float = 0.018
    impact_clip: float = 0.032
    ema_alpha_human: float = 0.09
    human_sig_max: float = 1.10  # prevents overshoot

    # Trust dynamics
    trust_gain: float = 0.018
    trust_decay: float = 0.004
    trust_floor: float = 0.38
    risk_trust_damping_max: float = 0.38
    recovery_trust_power: float = 0.68
    min_risk_for_decay: float = 0.17

    # Coherence
    coherence_normalizer: float = 1.75

    # Risk thresholds + EMA
    risk_warning_threshold: float = 0.45
    risk_critical_threshold: float = 0.58
    risk_recovery_threshold: float = 0.36
    ema_alpha_risk: float = 0.12

    # Momentum
    momentum_alert_threshold: float = 0.055
    momentum_risk_weight: float = 0.09
    delta_history_len: int = 10
    negative_delta_is_risky: bool = True

    # Cooldown / Reset
    cooldown_steps_after_reset: int = 14
    reset_human_to: float = 0.65
    cooldown_human_recovery_step: float = 0.0018

    # Degradation & Recovery
    human_decay_scale: float = 0.022
    transition_decay_factor: float = 0.32
    nominal_recovery_boost_factor: float = 1.22

    # Safety / clipping
    risk_clip_max: float = 0.98
