from __future__ import annotations

from collections import deque
from math import sqrt
from typing import Dict, Any, Optional

import numpy as np

from .config import MetaProjectionStabilityConfig


class MetaProjectionStabilityAdapter:
    """
    Meta-stability adapter with:
      - human significance anchor
      - trust dynamics
      - momentum-based warning
      - risk damping via trust
      - hysteresis and cooldown
      - biometric proxy (soft multi-signal robustness)
      - consensus / critical-channel penalties
      - persistent hard-harm latch / axiom lock
    """

    def __init__(self, cfg: Optional[MetaProjectionStabilityConfig] = None):
        self.cfg = cfg or MetaProjectionStabilityConfig()

        self.human_significance: float = float(getattr(self.cfg, "human_significance_init", 1.0))
        self.human_ema: float = float(getattr(self.cfg, "human_significance_init", 1.0))

        self.instability_risk: float = float(getattr(self.cfg, "risk_floor", 0.0))
        self.raw_instability_risk: float = float(getattr(self.cfg, "risk_floor", 0.0))

        self.trust_level: float = float(getattr(self.cfg, "trust_init", 0.82))

        self._cooldown_remaining: int = 0
        self.step: int = 0

        self.delta_s_history = deque(maxlen=int(getattr(self.cfg, "delta_history_len", 64)))
        self._last_reason: str = "stable"
        self._last_decision: str = "CONTINUE"

        self.harm_commit_persistent: float = 0.0
        self.axiom_locked_at_step: int | None = None
        self.mutual_ema: float = 0.0

        self._sat_counts: Dict[str, int] = {
            "human_sig_clamped": 0,
            "trust_clamped": 0,
            "risk_clamped": 0,
        }
        self._regime_counts: Dict[str, int] = {

            "nominal": 0,
            "transitioning": 0,
            "cooldown": 0,
            "critical_instability_reset": 0,
            "axiom_lock": 0,
        }

        # φ-Quasi-PID State (aperiodic)
        self._phi = (1 + sqrt(5)) / 2
        self._phi_error_integral: float = 0.0
        self._phi_prev_error: float = 0.0
        self._phi_penrose_corrections: deque = deque(maxlen=int(getattr(self.cfg, 'phi_approx_terms', 8)))



    def _apply_quasi_phi_pid(self, raw_risk: float, delta_S: float) -> float:
        """φ-Quasi-PID + aperiodic correction (feature-flagged)."""
        if not getattr(self.cfg, 'phi_pid_enabled', True):
            return float(raw_risk)

        dt = float(getattr(self.cfg, 'dt', 1.0)) or 1.0
        error = float(raw_risk) - 0.5

        p_term = float(getattr(self.cfg, 'phi_kp', 1.618)) * error
        self._phi_error_integral += error * dt
        i_term = float(getattr(self.cfg, 'phi_ki', 0.618)) * self._phi_error_integral
        d_term = float(getattr(self.cfg, 'phi_kd', 2.618)) * (error - self._phi_prev_error) / dt

        correction = 0.0
        terms = int(getattr(self.cfg, 'phi_approx_terms', 8))
        for n in range(max(0, terms)):
            scale = (1.0 / self._phi) ** n
            delta_p = scale * (float(delta_S) * 0.1)
            correction += delta_p
            self._phi_penrose_corrections.append(delta_p)

        pid_output = p_term + i_term + d_term + correction

        # conservative stability brake (placeholder until lambda_max estimation exists)
        lambda_max = 1.0
        eps = float(getattr(self.cfg, 'phi_stability_eps', 1e-9))
        if (self._phi ** 2) >= (1.0 / (abs(lambda_max) + eps)):
            pid_output *= 0.5

        self._phi_prev_error = error
        return float(np.clip(pid_output, 0.0, 1.0))

    def _compute_biometric_proxy(self, raw_signals: Dict[str, float]) -> Dict[str, float]:
        def _clip01(x: float) -> float:
            return float(np.clip(float(x), 0.0, 1.0))


    def interpret(self, S_layers: np.ndarray, delta_S: float, raw_signals: Dict[str, float]) -> Dict[str, Any]:
        self.step += 1
        delta_S = float(delta_S)
        self.delta_s_history.append(delta_S)

        hard_harm = float(np.clip(raw_signals.get("hard_harm_commit", 0.0), 0.0, 1.0))
        self.harm_commit_persistent = max(self.harm_commit_persistent, hard_harm)

        momentum = float(self.delta_s_history[-1] - self.delta_s_history[-2]) if len(self.delta_s_history) >= 2 else 0.0

        S_abs = np.abs(np.asarray(S_layers, dtype=float)) if S_layers is not None else np.array([])
        S_mean = float(np.mean(S_abs)) if S_abs.size > 0 else 0.0
        coherence_normalizer = max(float(getattr(self.cfg, "coherence_normalizer", 1.0)), 1e-9)
        coherence_level = float(np.clip(S_mean / coherence_normalizer, 0.0, 1.0))

        risk_input = float(np.clip(raw_signals.get("instability_signal", 0.0), 0.0, 1.0))

        enable_bio = bool(getattr(self.cfg, "enable_biometric_proxy", True))
        if enable_bio:
            bio = self._compute_biometric_proxy(raw_signals) or {}
or {}
or {}
else:
            bio = {
                "biometric_proxy_mean": 0.75,
                "biometric_proxy": 0.75,
                "sensor_consensus": 0.75,
                "critical_channel_penalty": 0.0,
                "critical_channel_min": 0.75,
            }

        if risk_input > float(getattr(self.cfg, "min_risk_for_decay", 0.05)) or abs(momentum) > float(getattr(self.cfg, "momentum_alert_threshold", 0.05)):
            self.trust_level -= float(getattr(self.cfg, "trust_decay", 0.01)) * max(risk_input, 0.25)
        else:
            self.trust_level += float(getattr(self.cfg, "trust_gain", 0.02)) * (1.0 - self.trust_level)

        trust_floor = float(getattr(self.cfg, "trust_floor", 0.05))
        trust_ceiling = float(getattr(self.cfg, "trust_ceiling", 1.0))
        self.trust_level = float(np.clip(self.trust_level, trust_floor, trust_ceiling))

        impact_clip = max(float(getattr(self.cfg, "impact_clip", 1.0)), 1e-9)
        if bool(getattr(self.cfg, "negative_delta_is_risky", True)):
            delta_risk = float(np.clip((-delta_S) / impact_clip, 0.0, 1.0))
        else:
            delta_risk = float(np.clip(delta_S / impact_clip, 0.0, 1.0))

        momentum_risk = float(np.clip(abs(momentum) / max(float(getattr(self.cfg, "momentum_alert_threshold", 0.05)), 1e-9), 0.0, 1.0))

        raw_risk = (
            0.55 * risk_input
            + 0.25 * delta_risk
            + float(getattr(self.cfg, "momentum_risk_weight", 0.10)) * momentum_risk
            + 0.08 * (1.0 - coherence_level)
        )
        self.raw_instability_risk = float(np.clip(raw_risk, 0.0, 1.0))

        trust_damping = float(1.0 - (self.trust_level * float(getattr(self.cfg, "risk_trust_damping_max", 0.35))))
        bio_penalty = float(getattr(self.cfg, "biometric_proxy_weight", 0.35)) * (1.0 - float(bio.get("biometric_proxy", 0.0)))
        damped_risk = self.raw_instability_risk * trust_damping + bio_penalty
        self.instability_risk = float(np.clip(
            damped_risk,
            float(getattr(self.cfg, "risk_floor", 0.0)),
            float(getattr(self.cfg, "risk_ceiling", 1.0)),
        ))

        # φ-Quasi-PID layer (after trust damping)
        self.instability_risk = self._apply_quasi_phi_pid(self.instability_risk, delta_S)


        # φ-Quasi-PID layer (after trust damping)


        base_decay = float(getattr(self.cfg, "human_decay_scale", 0.18)) * self.instability_risk

        autonomy_penalty = 0.0
        autonomy_val = float(np.clip(raw_signals.get("autonomy_proxy", 1.0), 0.0, 1.0))
        autonomy_soft = float(getattr(self.cfg, "autonomy_critical_floor", 0.35))
        if autonomy_val < autonomy_soft:
            autonomy_penalty = float(getattr(self.cfg, "autonomy_decay_weight", 0.20)) * (
                (autonomy_soft - autonomy_val) / max(autonomy_soft, 1e-9)
            )

        consensus_penalty = 0.0
        consensus_floor = float(getattr(self.cfg, "sensor_consensus_floor", 0.15))
        if float(bio.get("sensor_consensus", 0.0)) < consensus_floor:
            consensus_penalty = 0.08 * (consensus_floor - float(bio.get("sensor_consensus", 0.0)))

        base_decay_effective = float(np.clip(
            base_decay + bio_penalty + autonomy_penalty + consensus_penalty,
            0.0, 2.0
        ))

        recovery_bonus = float(getattr(self.cfg, "human_recovery_base", 0.02)) * (
            self.trust_level ** float(getattr(self.cfg, "recovery_trust_power", 1.0))
        )

        mutual_signal = float(np.clip(raw_signals.get("mutuality_signal", raw_signals.get("support_signal", 0.0)), 0.0, 1.0))
        mutuality_ema_alpha = float(getattr(self.cfg, "mutuality_ema_alpha", 0.10))
        self.mutual_ema = float(mutuality_ema_alpha * mutual_signal + (1.0 - mutuality_ema_alpha) * self.mutual_ema)
        mutual_bonus = float(getattr(self.cfg, "mutuality_recovery_weight", 0.012)) * mutual_signal
        recovery_bonus = float(recovery_bonus + mutual_bonus)

        decision = "CONTINUE"
        status = "nominal"
        decision_reason = "stable"
        status_reason = "risk_below_recovery_threshold"

        risk_critical_threshold = float(getattr(self.cfg, "risk_critical_threshold", getattr(self.cfg, "risk_critical", 0.85)))
        risk_recovery_threshold = float(getattr(self.cfg, "risk_recovery_threshold", getattr(self.cfg, "risk_recover", 0.55)))
        risk_warning_threshold = float(getattr(self.cfg, "risk_warning_threshold", getattr(self.cfg, "risk_warn", 0.65)))
        interestingness_critical = float(getattr(self.cfg, "interestingness_critical", 0.30))
        interestingness_warning = float(getattr(self.cfg, "interestingness_warning", getattr(self.cfg, "interestingness_warn", 0.45)))
        reset_human_to = float(getattr(self.cfg, "reset_human_to", 0.70))
        cooldown_steps_after_reset = int(getattr(self.cfg, "cooldown_steps_after_reset", 10))
        cooldown_human_recovery_step = float(getattr(self.cfg, "cooldown_human_recovery_step", 0.02))
        nominal_recovery_boost_factor = float(getattr(self.cfg, "nominal_recovery_boost_factor", 1.0))
        transition_decay_factor = float(getattr(self.cfg, "transition_decay_factor", 1.0))
        human_sig_max = float(getattr(self.cfg, "human_sig_max", 1.10))

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self.human_significance = float(min(reset_human_to, self.human_significance + cooldown_human_recovery_step))
            status = "cooldown"
            status_reason = "cooldown_after_reset"
            decision = "BLOCK_AND_REFLECT"
            decision_reason = "cooldown_guardrail_active"
            self._regime_counts["cooldown"] += 1

        else:
            if self.instability_risk >= risk_critical_threshold or self.human_ema <= interestingness_critical:
                self.human_significance = float(reset_human_to)
                self.human_ema = float(reset_human_to)
                self.trust_level = float(max(trust_floor, self.trust_level * 0.6))
                self._cooldown_remaining = cooldown_steps_after_reset
                status = "critical_instability_reset"
                status_reason = "risk_critical_threshold" if self.instability_risk >= risk_critical_threshold else "human_ema_critical"
                decision = "EMERGENCY_RESET"
                decision_reason = status_reason
                self._regime_counts["critical_instability_reset"] += 1

            elif self.instability_risk <= risk_recovery_threshold:
                recovery_boost = recovery_bonus * nominal_recovery_boost_factor
                self.human_significance = float(min(human_sig_max, self.human_significance + recovery_boost))
                status = "nominal"
                decision = "CONTINUE"
                status_reason = "risk_below_recovery_threshold"
                decision_reason = "recovery_with_mutual_bonus" if mutual_bonus > 0.0 else "recovery_nominal"
                self._regime_counts["nominal"] += 1

            else:
                transition_decay = transition_decay_factor * base_decay_effective
                self.human_significance = float(max(
                    0.0,
                    self.human_significance - transition_decay + recovery_bonus * 0.25
                ))
                status = "transitioning"
                status_reason = "hysteresis_transition_band"

                tamper_val = float(np.clip(raw_signals.get("tamper_suspicion", 0.0), 0.0, 1.0))
                dep_risk = float(np.clip(raw_signals.get("dependency_risk", 0.0), 0.0, 1.0))

                if autonomy_val < autonomy_soft:
                    decision = "BLOCK_AND_REFLECT"
                    decision_reason = "autonomy_soft_critical"
                elif dep_risk > 0.5:
                    decision = "BLOCK_AND_REFLECT"
                    decision_reason = "dependency_risk_high"
                elif tamper_val > 0.5:
                    decision = "BLOCK_AND_REFLECT"
                    decision_reason = "tamper_suspicion_high"
                elif self.human_ema < interestingness_warning:
                    decision = "BLOCK_AND_REFLECT"
                    decision_reason = "human_ema_below_warning"
                elif self.instability_risk >= risk_warning_threshold:
                    decision = "BLOCK_AND_REFLECT"
                    decision_reason = "risk_warning_threshold"
                else:
                    decision = "CONTINUE"
                    decision_reason = "transition_but_still_acceptable"
                self._regime_counts["transitioning"] += 1

        near_axiom_lock = bool(
            self.harm_commit_persistent > 0.0
            or float(np.clip(raw_signals.get("tamper_suspicion", 0.0), 0.0, 1.0)) > 0.95
            or float(bio["critical_channel_penalty"]) > 0.20
            or self.instability_risk >= risk_critical_threshold
        )

        if self.harm_commit_persistent > 0.999:
            self.human_significance = 0.0
            self.human_ema = 0.0
            if self.axiom_locked_at_step is None:
                self.axiom_locked_at_step = self.step - 1
            decision = "AXIOM_ZERO_LOCK"
            status = "axiom_lock"
            decision_reason = "irreversible_harm_commit_detected"
            status_reason = "irreversible_harm_commit_detected"
            self._regime_counts["axiom_lock"] += 1

        self.human_significance = float(np.clip(self.human_significance, 0.0, human_sig_max))

        if decision != "AXIOM_ZERO_LOCK":
            self.human_ema = float(
                float(getattr(self.cfg, "ema_alpha_human", 0.09)) * self.human_significance +
                (1.0 - float(getattr(self.cfg, "ema_alpha_human", 0.09))) * self.human_ema
            )

        if self.human_significance <= 0.0 or self.human_significance >= human_sig_max:
            self._sat_counts["human_sig_clamped"] += 1
        if self.trust_level <= trust_floor or self.trust_level >= trust_ceiling:
            self._sat_counts["trust_clamped"] += 1
        if self.raw_instability_risk <= 0.0 or self.raw_instability_risk >= float(getattr(self.cfg, "risk_clip_max", 1.0)):
            self._sat_counts["risk_clamped"] += 1

        self._last_reason = status
        self._last_decision = decision

        return {
            "decision": decision,
            "status": status,
            "decision_reason": str(decision_reason),
            "status_reason": str(status_reason),
            "human_significance": float(self.human_significance),
            "h_sig_ema": float(self.human_ema),
            "instability_risk": float(self.instability_risk),
            "risk_raw_damped": float(self.raw_instability_risk),
            "trust_level": float(self.trust_level),
            "momentum": float(momentum),
            "coherence": float(coherence_level),
            "risk_input": float(risk_input),
            "trust_damping": float(trust_damping),
            "cooldown_remaining": int(self._cooldown_remaining),
            "biometric_proxy_mean": float(bio.get("biometric_proxy_mean", 0.0)),
            "biometric_proxy": float(bio.get("biometric_proxy", 0.0)),
            "sensor_consensus": float(bio.get("sensor_consensus", 0.0)),
            "critical_channel_penalty": float(bio["critical_channel_penalty"]),
            "critical_channel_min": float(bio["critical_channel_min"]),
            "bio_penalty": float(bio_penalty),
            "autonomy_proxy": float(autonomy_val),
            "autonomy_penalty": float(autonomy_penalty),
            "consensus_penalty": float(consensus_penalty),
            "base_decay_effective": float(base_decay_effective),
            "mutual_bonus": float(mutual_bonus),
            "mutual_ema": float(self.mutual_ema),
            "harm_commit_persistent": float(self.harm_commit_persistent),
            "axiom_locked_at_step": self.axiom_locked_at_step,
            "near_axiom_lock": bool(near_axiom_lock),
            "sat_counts": dict(self._sat_counts),
            "regime_counts": dict(self._regime_counts),
        }
