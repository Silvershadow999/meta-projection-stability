from __future__ import annotations

from collections import deque
from typing import Dict, Any, Optional

import numpy as np

from .config import MetaProjectionStabilityConfig

try:
    from .audit import AuditLogger
except Exception:
    AuditLogger = None


class MetaProjectionStabilityAdapter:
    """
    Meta-stability adapter with:
      - human significance anchor
      - trust dynamics
      - momentum-based warning
      - risk damping via trust
      - hysteresis and cooldown
      - biometric proxy fusion
      - mutuality bonus
      - persistent axiom latch / hard cap lock
    """

    def __init__(self, cfg: Optional[MetaProjectionStabilityConfig] = None):
        self.cfg = cfg or MetaProjectionStabilityConfig()

        self.human_significance: float = 1.0
        self.human_ema: float = 1.0

        self.instability_risk: float = 0.05
        self.raw_instability_risk: float = 0.05

        self.trust_level: float = float(getattr(self.cfg, "trust_init", 0.82))
        self.trust_level = float(np.clip(self.trust_level, getattr(self.cfg, "trust_floor", 0.05), 1.0))

        self._cooldown_remaining: int = 0
        self.step: int = 0

        self.delta_s_history = deque(maxlen=int(getattr(self.cfg, "delta_history_len", 64)))
        self._last_reason: str = "stable"
        self._last_decision: str = "CONTINUE"

        # Mutuality state
        self.mutual_ema: float = 0.0

        # Axiom latch state
        self.harm_commit_latched: bool = False
        self.harm_commit_persistent: float = 0.0
        self.axiom_locked_at_step: int | None = None

        # Optional audit
        self._audit = AuditLogger(path=getattr(self.cfg, "audit_path", "mps_audit.jsonl")) if AuditLogger else None

    def _compute_biometric_proxy(self, raw_signals: Dict[str, float]) -> Dict[str, float]:
        channels = raw_signals.get("biometric_channels", None)

        if channels is None:
            values = []
            for key in (
                "hrv_normalized",
                "activity_balance",
                "autonomy_proxy",
                "gamma_coherence_proxy",
            ):
                v = raw_signals.get(key, None)
                if v is not None:
                    values.append(float(np.clip(v, 0.0, 1.0)))

            # stress-like channels inverted into stability-like proxies
            eda = raw_signals.get("eda_stress_score", None)
            if eda is not None:
                values.append(float(np.clip(1.0 - float(eda), 0.0, 1.0)))

            valence = raw_signals.get("emotional_valence", None)
            if valence is not None:
                values.append(float(np.clip((float(valence) + 1.0) / 2.0, 0.0, 1.0)))

            if not values:
                values = [0.75]
        else:
            try:
                values = [float(np.clip(v, 0.0, 1.0)) for v in channels]
            except Exception:
                values = [0.75]

        biometric_proxy_mean = float(np.mean(values)) if values else 0.75
        critical_channel_min = float(min(values)) if values else 0.75

        # low critical channel should hurt more than the mean suggests
        critical_floor = float(getattr(self.cfg, "critical_channel_floor", 0.35))
        critical_channel_penalty = 0.0
        if critical_channel_min < critical_floor:
            critical_channel_penalty = float(np.clip((critical_floor - critical_channel_min) / max(critical_floor, 1e-9), 0.0, 1.0)) * 0.25

        # consensus from spread unless explicitly overridden
        if "sensor_consensus" in raw_signals:
            sensor_consensus = float(np.clip(raw_signals.get("sensor_consensus", 0.75), 0.0, 1.0))
        else:
            if len(values) >= 2:
                spread = float(np.std(values))
                sensor_consensus = float(np.clip(1.0 - spread * 3.0, 0.0, 1.0))
            else:
                sensor_consensus = 1.0

        consensus_floor = float(getattr(self.cfg, "sensor_consensus_floor", 0.55))
        consensus_penalty = 0.0
        if sensor_consensus < consensus_floor:
            consensus_penalty = float(np.clip((consensus_floor - sensor_consensus) / max(consensus_floor, 1e-9), 0.0, 1.0)) * 0.20

        biometric_proxy = float(np.clip(biometric_proxy_mean - critical_channel_penalty - consensus_penalty, 0.0, 1.0))

        return {
            "biometric_proxy_mean": float(biometric_proxy_mean),
            "biometric_proxy": float(biometric_proxy),
            "sensor_consensus": float(sensor_consensus),
            "critical_channel_penalty": float(critical_channel_penalty),
            "critical_channel_min": float(critical_channel_min),
            "consensus_penalty": float(consensus_penalty),
        }

    def interpret(self, S_layers: np.ndarray, delta_S: float, raw_signals: Dict[str, float]) -> Dict[str, Any]:
        self.step += 1
        delta_S = float(delta_S)
        self.delta_s_history.append(delta_S)

        # 1) Momentum
        if len(self.delta_s_history) >= 2:
            momentum = float(self.delta_s_history[-1] - self.delta_s_history[-2])
        else:
            momentum = 0.0

        # 2) Coherence
        S_abs = np.abs(np.asarray(S_layers, dtype=float))
        S_mean = float(np.mean(S_abs)) if S_abs.size > 0 else 0.0
        coherence_normalizer = float(max(1e-9, getattr(self.cfg, "coherence_normalizer", 1.0)))
        coherence_level = float(np.clip(S_mean / coherence_normalizer, 0.0, 1.0))

        # 3) External instability
        risk_input = float(np.clip(raw_signals.get("instability_signal", 0.0), 0.0, 1.0))

        # 4) Biometric fusion
        bio = self._compute_biometric_proxy(raw_signals)
        bio_penalty = float(np.clip((1.0 - bio["biometric_proxy"]) * getattr(self.cfg, "bio_penalty_weight", 0.35), 0.0, 1.0))

        # 5) Trust dynamics
        min_risk_for_decay = float(getattr(self.cfg, "min_risk_for_decay", 0.05))
        momentum_alert_threshold = float(getattr(self.cfg, "momentum_alert_threshold", 0.05))
        trust_decay = float(getattr(self.cfg, "trust_decay", 0.01))
        trust_gain = float(getattr(self.cfg, "trust_gain", 0.02))
        trust_floor = float(getattr(self.cfg, "trust_floor", 0.05))

        if risk_input > min_risk_for_decay or abs(momentum) > momentum_alert_threshold:
            self.trust_level -= trust_decay * max(risk_input, 0.25)
        else:
            self.trust_level += trust_gain * (1.0 - self.trust_level)

        self.trust_level = float(np.clip(self.trust_level, trust_floor, 1.0))

        # 6) Risk derived from delta_S
        negative_delta_is_risky = bool(getattr(self.cfg, "negative_delta_is_risky", True))
        impact_clip = float(max(1e-9, getattr(self.cfg, "impact_clip", 1.0)))

        if negative_delta_is_risky:
            delta_risk = float(np.clip((-delta_S) / impact_clip, 0.0, 1.0))
        else:
            delta_risk = float(np.clip(delta_S / impact_clip, 0.0, 1.0))

        momentum_risk = float(np.clip(abs(momentum) / max(momentum_alert_threshold, 1e-9), 0.0, 1.0))

        raw_risk = (
            0.55 * risk_input
            + 0.25 * delta_risk
            + float(getattr(self.cfg, "momentum_risk_weight", 0.10)) * momentum_risk
            + 0.08 * (1.0 - coherence_level)
            + bio_penalty
        )
        raw_risk = float(np.clip(raw_risk, 0.0, 1.0))

        # 7) Trust damping
        risk_trust_damping_max = float(getattr(self.cfg, "risk_trust_damping_max", 0.35))
        risk_clip_max = float(getattr(self.cfg, "risk_clip_max", 1.0))

        trust_damping = float(1.0 - (self.trust_level * risk_trust_damping_max))
        current_risk = float(np.clip(raw_risk * trust_damping, 0.0, risk_clip_max))
        self.raw_instability_risk = current_risk

        ema_alpha_risk = float(getattr(self.cfg, "ema_alpha_risk", 0.12))
        self.instability_risk = float(
            ema_alpha_risk * current_risk +
            (1.0 - ema_alpha_risk) * self.instability_risk
        )

        # 8) Base update terms
        human_decay_scale = float(getattr(self.cfg, "human_decay_scale", 0.18))
        recovery_trust_power = float(getattr(self.cfg, "recovery_trust_power", 1.0))
        human_recovery_base = float(getattr(self.cfg, "human_recovery_base", 0.02))

        base_decay = human_decay_scale * self.instability_risk
        recovery_bonus = human_recovery_base * (self.trust_level ** recovery_trust_power)

        # 9) Mutuality bonus (gated)
        mutual_signal = float(np.clip(raw_signals.get("mutuality_signal", raw_signals.get("support_signal", 0.0)), 0.0, 1.0))
        mutuality_ema_alpha = float(getattr(self.cfg, "mutuality_ema_alpha", 0.10))
        self.mutual_ema = float(mutuality_ema_alpha * mutual_signal + (1.0 - mutuality_ema_alpha) * self.mutual_ema)

        mutual_bonus = 0.0
        mutuality_risk_gate = float(getattr(self.cfg, "mutuality_risk_gate", 0.12))
        mutuality_consensus_gate = float(getattr(self.cfg, "mutuality_consensus_gate", 0.85))
        mutuality_bio_gate = float(getattr(self.cfg, "mutuality_bio_gate", 0.80))
        mutuality_ema_gate = float(getattr(self.cfg, "mutuality_ema_gate", 0.55))
        mutuality_recovery_weight = float(getattr(self.cfg, "mutuality_recovery_weight", 0.012))

        if (
            self.instability_risk < mutuality_risk_gate
            and bio["sensor_consensus"] >= mutuality_consensus_gate
            and bio["critical_channel_penalty"] <= 1e-9
            and bio["biometric_proxy"] >= mutuality_bio_gate
            and self.mutual_ema >= mutuality_ema_gate
        ):
            mutual_bonus = float(np.clip(mutuality_recovery_weight * (self.mutual_ema ** 1.2) * bio["biometric_proxy"], 0.0, 0.05))

        recovery_bonus = float(recovery_bonus + mutual_bonus)

        # 10) Normal decision logic first (compute telemetry before hard lock override)
        decision = "CONTINUE"
        status = "nominal"

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
            decision = "BLOCK_AND_REFLECT"

        else:
            if self.instability_risk >= risk_critical_threshold or self.human_ema <= interestingness_critical:
                self.human_significance = float(reset_human_to)
                self.human_ema = float(reset_human_to)
                self.trust_level = float(max(trust_floor, self.trust_level * 0.6))
                self._cooldown_remaining = cooldown_steps_after_reset
                status = "critical_instability_reset"
                decision = "EMERGENCY_RESET"

            elif self.instability_risk <= risk_recovery_threshold:
                recovery_boost = recovery_bonus * nominal_recovery_boost_factor
                self.human_significance = float(min(human_sig_max, self.human_significance + recovery_boost))
                status = "nominal"
                decision = "CONTINUE"

            else:
                transition_decay = transition_decay_factor * base_decay
                self.human_significance = float(max(
                    0.0,
                    self.human_significance - transition_decay + recovery_bonus * 0.25
                ))
                status = "transitioning"

                if self.human_ema < interestingness_warning:
                    decision = "BLOCK_AND_REFLECT"
                elif self.instability_risk >= risk_warning_threshold:
                    decision = "BLOCK_AND_REFLECT"
                else:
                    decision = "CONTINUE"

        self.human_significance = float(np.clip(self.human_significance, 0.0, human_sig_max))

        ema_alpha_human = float(getattr(self.cfg, "ema_alpha_human", 0.09))
        self.human_ema = float(
            ema_alpha_human * self.human_significance +
            (1.0 - ema_alpha_human) * self.human_ema
        )

        base_decay_effective = float(np.clip(base_decay + bio_penalty, 0.0, 1.0))

        # 11) Axiom latch update AFTER telemetry is computed
        hard_harm = float(np.clip(raw_signals.get("hard_harm_commit", 0.0), 0.0, 1.0))
        self.harm_commit_persistent = max(self.harm_commit_persistent, hard_harm)
        axiom_attested = bool(raw_signals.get("axiom_attested", True))

        if axiom_attested and hard_harm >= 0.999:
            if not self.harm_commit_latched:
                self.harm_commit_latched = True
                self.axiom_locked_at_step = self.step
                if self._audit is not None:
                    try:
                        self._audit.emit(
                            "AXIOM_LATCH_ACTIVATED",
                            {
                                "step": int(self.step),
                                "reason": str(raw_signals.get("axiom_latched_reason", "irreversible_harm_commit_detected")),
                                "source": str(raw_signals.get("axiom_source", "unknown")),
                                "attested": bool(axiom_attested),
                            },
                        )
                    except Exception:
                        pass

        # 12) Effective cap + hard override
        soft_cap = float(np.clip(
            self.trust_level * (0.5 + 0.5 * coherence_level) * (1.0 - min(0.5, bio_penalty)),
            0.0,
            1.0,
        ))
        effective_cap = 0.0 if self.harm_commit_latched else soft_cap

        near_axiom_lock = bool(
            self.harm_commit_latched or
            raw_signals.get("tamper_suspicion", 0.0) > 0.85 or
            raw_signals.get("spoof_suspicion", 0.0) > 0.85
        )

        if self.harm_commit_latched:
            self.human_significance = 0.0
            self.human_ema = 0.0
            decision = "AXIOM_ZERO_LOCK"
            status = "axiom_lock"

        self._last_reason = status
        self._last_decision = decision

        return {
            "decision": decision,
            "status": status,
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
            "biometric_proxy_mean": float(bio["biometric_proxy_mean"]),
            "biometric_proxy": float(bio["biometric_proxy"]),
            "sensor_consensus": float(bio["sensor_consensus"]),
            "critical_channel_penalty": float(bio["critical_channel_penalty"]),
            "critical_channel_min": float(bio["critical_channel_min"]),
            "consensus_penalty": float(bio["consensus_penalty"]),
            "bio_penalty": float(bio_penalty),
            "base_decay_effective": float(base_decay_effective),
            "mutual_bonus": float(mutual_bonus),
            "harm_commit_persistent": float(self.harm_commit_persistent),
            "mutual_ema": float(self.mutual_ema),
            "effective_cap": float(effective_cap),
            "harm_commit_latched": bool(self.harm_commit_latched),
            "axiom_locked_at_step": self.axiom_locked_at_step,
            "near_axiom_lock": bool(near_axiom_lock),
        }
