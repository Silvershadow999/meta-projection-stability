from __future__ import annotations

from collections import deque
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
      - optional biometric / consensus diagnostics (soft penalty only)
    """

    def __init__(self, cfg: Optional[MetaProjectionStabilityConfig] = None):
        self.cfg = cfg or MetaProjectionStabilityConfig()

        self.human_significance: float = 1.0
        self.human_ema: float = 1.0

        self.instability_risk: float = 0.05      # smoothed
        self.raw_instability_risk: float = 0.05  # step-local damped

        self.trust_level: float = 0.82

        self._cooldown_remaining: int = 0
        self.step: int = 0

        self.delta_s_history = deque(maxlen=self.cfg.delta_history_len)
        self._last_reason: str = "stable"
        self._last_decision: str = "CONTINUE"

    # ------------------------------------------------------------------
    # Optional biometric proxy helper (safe, no config dependency required)
    # ------------------------------------------------------------------
    def _compute_biometric_proxy(self, raw_signals: Dict[str, float]) -> Dict[str, float]:
        """
        Build a robust biometric proxy and sensor consensus estimate.

        This method is intentionally defensive:
        - If no biometric signals are present, it returns neutral defaults.
        - If explicit 'sensor_consensus' is provided, it overrides computed consensus.
        - Critical-channel penalty is soft-capped and can be used downstream as a mild decay modifier.
        """

        # Collect candidate channels (only those present)
        # You can expand this list later without breaking compatibility.
        candidate_keys = [
            "hrv",
            "resp_variability",
            "skin_temp_stability",
            "eda_stability",
            "sleep_quality",
            "attention_stability",
            "biometric_proxy",       # explicit override possible
            "biometric_proxy_mean",  # explicit debug/override possible
        ]

        vals = []
        for k in candidate_keys:
            if k in raw_signals:
                try:
                    vals.append(float(np.clip(raw_signals[k], 0.0, 1.0)))
                except Exception:
                    # Ignore malformed values silently (robust telemetry behavior)
                    pass

        # Neutral defaults if nothing exists
        if len(vals) == 0:
            biometric_proxy_mean = 0.75
            sensor_consensus = 0.75
        else:
            arr = np.asarray(vals, dtype=float)

            # Robust central estimate: mean of clipped values
            robust_core = float(np.mean(arr))

            # Optional support mean (same array here; placeholder for future split logic)
            support_mean = float(np.mean(arr))

            # Base proxy
            biometric_proxy_mean = float(np.clip(0.8 * robust_core + 0.2 * support_mean, 0.0, 1.0))

            # Consensus: high when spread is low
            sensor_consensus = float(np.clip(1.0 - np.std(arr), 0.0, 1.0))

        # Explicit overrides (if caller injects them)
        if "sensor_consensus" in raw_signals:
            sensor_consensus = float(np.clip(raw_signals.get("sensor_consensus", sensor_consensus), 0.0, 1.0))

        if "biometric_proxy_mean" in raw_signals:
            biometric_proxy_mean = float(np.clip(raw_signals.get("biometric_proxy_mean", biometric_proxy_mean), 0.0, 1.0))

        # If caller provides a final biometric proxy directly, use as base proxy before penalty
        biometric_proxy_pre_penalty = biometric_proxy_mean
        if "biometric_proxy" in raw_signals:
            biometric_proxy_pre_penalty = float(np.clip(raw_signals.get("biometric_proxy", biometric_proxy_pre_penalty), 0.0, 1.0))

        # Minimal critical-channel proxy in this simplified version:
        # we use consensus as a proxy for weakest critical channel quality
        critical_channel_min = float(sensor_consensus)

        # Soft penalty (bounded). Threshold 0.35 chosen to avoid overreaction.
        critical_channel_penalty = float(max(0.0, min(0.35, 0.35 - critical_channel_min)))

        # Optional: apply penalty to final biometric proxy (soft damped)
        biometric_proxy = float(
            max(0.0, min(1.0, biometric_proxy_pre_penalty - 0.6 * critical_channel_penalty))
        )

        return {
            "biometric_proxy": biometric_proxy,
            "biometric_proxy_mean": biometric_proxy_mean,
            "critical_channel_min": critical_channel_min,
            "critical_channel_penalty": critical_channel_penalty,
            "sensor_consensus": sensor_consensus,
        }

    def interpret(self, S_layers: np.ndarray, delta_S: float, raw_signals: Dict[str, float]) -> Dict[str, Any]:
        self.step += 1
        delta_S = float(delta_S)
        self.delta_s_history.append(delta_S)

        # 0) Optional biometric diagnostics / soft penalties (safe defaults)
        enable_biometric_proxy = bool(getattr(self.cfg, "enable_biometric_proxy", True))
        if enable_biometric_proxy:
            bio = self._compute_biometric_proxy(raw_signals)
        else:
            bio = {
                "biometric_proxy": 0.75,
                "biometric_proxy_mean": 0.75,
                "critical_channel_min": 0.75,
                "critical_channel_penalty": 0.0,
                "sensor_consensus": 0.75,
            }

        bio_proxy = float(bio["biometric_proxy"])
        bio_penalty = float(getattr(self.cfg, "biometric_proxy_weight", 0.12)) * (1.0 - bio_proxy)

        consensus_penalty = 0.0
        consensus_floor = float(getattr(self.cfg, "sensor_consensus_floor", 0.15))
        if float(bio["sensor_consensus"]) < consensus_floor:
            consensus_penalty = 0.08 * (consensus_floor - float(bio["sensor_consensus"]))

        # 1) Momentum (delta of delta_S)
        if len(self.delta_s_history) >= 2:
            momentum = float(self.delta_s_history[-1] - self.delta_s_history[-2])
        else:
            momentum = 0.0

        # 2) Coherence from layer magnitudes
        S_abs = np.abs(np.asarray(S_layers, dtype=float))
        S_mean = float(np.mean(S_abs)) if S_abs.size > 0 else 0.0
        coherence_level = float(np.clip(S_mean / self.cfg.coherence_normalizer, 0.0, 1.0))

        # 3) External instability signal
        risk_input = float(np.clip(raw_signals.get("instability_signal", 0.0), 0.0, 1.0))

        # 4) Trust dynamics (asymmetric + momentum early warning)
        if risk_input > self.cfg.min_risk_for_decay or abs(momentum) > self.cfg.momentum_alert_threshold:
            self.trust_level -= self.cfg.trust_decay * max(risk_input, 0.25)
        else:
            self.trust_level += self.cfg.trust_gain * (1.0 - self.trust_level)

        self.trust_level = float(np.clip(self.trust_level, self.cfg.trust_floor, 1.0))

        # 5) Risk derived from delta_S
        if self.cfg.negative_delta_is_risky:
            delta_risk = float(np.clip((-delta_S) / max(self.cfg.impact_clip, 1e-9), 0.0, 1.0))
        else:
            delta_risk = float(np.clip((delta_S) / max(self.cfg.impact_clip, 1e-9), 0.0, 1.0))

        momentum_risk = float(
            np.clip(abs(momentum) / max(self.cfg.momentum_alert_threshold, 1e-9), 0.0, 1.0)
        )

        raw_risk = (
            0.55 * risk_input +
            0.25 * delta_risk +
            self.cfg.momentum_risk_weight * momentum_risk +
            0.08 * (1.0 - coherence_level)
        )
        raw_risk = float(np.clip(raw_risk, 0.0, 1.0))

        # 6) Trust damping (more trust => more damping)
        trust_damping = float(1.0 - (self.trust_level * self.cfg.risk_trust_damping_max))
        current_risk = float(np.clip(raw_risk * trust_damping, 0.0, self.cfg.risk_clip_max))
        self.raw_instability_risk = current_risk

        self.instability_risk = float(
            self.cfg.ema_alpha_risk * current_risk +
            (1.0 - self.cfg.ema_alpha_risk) * self.instability_risk
        )

        # 7) Base continuous update terms
        # Add only SOFT biometric/consensus penalties to decay (bounded by clipping later)
        base_decay = self.cfg.human_decay_scale * self.instability_risk
        base_decay = float(np.clip(base_decay + bio_penalty + consensus_penalty, 0.0, 2.0))

        recovery_bonus = self.cfg.human_recovery_base * (self.trust_level ** self.cfg.recovery_trust_power)

        # 8) Hysteresis / Schmitt-trigger + decision logic
        decision = "CONTINUE"
        status = "nominal"
        decision_reason = "risk_below_warning"
        status_reason = "stable_nominal"

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

            self.human_significance = float(min(
                self.cfg.reset_human_to,
                self.human_significance + self.cfg.cooldown_human_recovery_step
            ))
            status = "cooldown"
            decision = "BLOCK_AND_REFLECT"
            status_reason = "cooldown_active"
            decision_reason = "cooldown_gate"

        else:
            # Critical conditions:
            #  - risk too high
            #  - human anchor collapsed
            if (
                self.instability_risk >= self.cfg.risk_critical_threshold
                or self.human_ema <= self.cfg.interestingness_critical
            ):
                self.human_significance = float(self.cfg.reset_human_to)
                self.human_ema = float(self.cfg.reset_human_to)
                self.trust_level = float(max(self.cfg.trust_floor, self.trust_level * 0.6))
                self._cooldown_remaining = int(self.cfg.cooldown_steps_after_reset)

                status = "critical_instability_reset"
                decision = "EMERGENCY_RESET"
                status_reason = "critical_threshold_breach"
                if self.instability_risk >= self.cfg.risk_critical_threshold:
                    decision_reason = "risk_critical_threshold"
                else:
                    decision_reason = "human_ema_below_critical"

            elif self.instability_risk <= self.cfg.risk_recovery_threshold:
                # Safe zone / recovery
                recovery_boost = recovery_bonus * self.cfg.nominal_recovery_boost_factor
                self.human_significance = float(min(
                    self.cfg.human_sig_max,
                    self.human_significance + recovery_boost
                ))
                status = "nominal"
                decision = "CONTINUE"
                status_reason = "risk_below_recovery_threshold"
                decision_reason = "recovery_nominal"

            else:
                # Transition band / hysteresis
                transition_decay = self.cfg.transition_decay_factor * base_decay
                self.human_significance = float(max(
                    0.0,
                    self.human_significance - transition_decay + recovery_bonus * 0.25
                ))
                status = "transitioning"
                status_reason = "hysteresis_transition_band"

                if self.human_ema < self.cfg.interestingness_warning:
                    decision = "BLOCK_AND_REFLECT"
                    decision_reason = "human_ema_below_warning"
                elif self.instability_risk >= self.cfg.risk_warning_threshold:
                    decision = "BLOCK_AND_REFLECT"
                    decision_reason = "risk_warning_threshold"
                else:
                    decision = "CONTINUE"
                    decision_reason = "transition_but_still_acceptable"

        self.human_significance = float(np.clip(self.human_significance, 0.0, self.cfg.human_sig_max))

        # EMA update (always)
        self.human_ema = float(
            self.cfg.ema_alpha_human * self.human_significance +
            (1.0 - self.cfg.ema_alpha_human) * self.human_ema
        )

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

            # Biometric / neuro diagnostics (optional but always present for stable API)
            "biometric_proxy": float(bio["biometric_proxy"]),
            "biometric_proxy_mean": float(bio["biometric_proxy_mean"]),
            "critical_channel_min": float(bio["critical_channel_min"]),
            "critical_channel_penalty": float(bio["critical_channel_penalty"]),
            "sensor_consensus": float(bio["sensor_consensus"]),

            # Penalty telemetry (helps verify effect in tests)
            "biometric_penalty": float(bio_penalty),
            "consensus_penalty": float(consensus_penalty),
            "base_decay_effective": float(base_decay),
        }
