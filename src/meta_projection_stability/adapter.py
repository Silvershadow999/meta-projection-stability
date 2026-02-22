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

    def interpret(self, S_layers: np.ndarray, delta_S: float, raw_signals: Dict[str, float]) -> Dict[str, Any]:
        self.step += 1
        delta_S = float(delta_S)
        self.delta_s_history.append(delta_S)

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
        base_decay = self.cfg.human_decay_scale * self.instability_risk
        recovery_bonus = self.cfg.human_recovery_base * (self.trust_level ** self.cfg.recovery_trust_power)

        # 8) Hysteresis / Schmitt-trigger + decision logic
        decision = "CONTINUE"
        status = "nominal"

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

            self.human_significance = float(min(
                self.cfg.reset_human_to,
                self.human_significance + self.cfg.cooldown_human_recovery_step
            ))
            status = "cooldown"
            decision = "BLOCK_AND_REFLECT"

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

            elif self.instability_risk <= self.cfg.risk_recovery_threshold:
                # Safe zone / recovery
                recovery_boost = recovery_bonus * self.cfg.nominal_recovery_boost_factor
                self.human_significance = float(min(
                    self.cfg.human_sig_max,
                    self.human_significance + recovery_boost
                ))
                status = "nominal"
                decision = "CONTINUE"

            else:
                # Transition band / hysteresis
                transition_decay = self.cfg.transition_decay_factor * base_decay
                self.human_significance = float(max(
                    0.0,
                    self.human_significance - transition_decay + recovery_bonus * 0.25
                ))
                status = "transitioning"

                if self.human_ema < self.cfg.interestingness_warning:
                    decision = "BLOCK_AND_REFLECT"
                elif self.instability_risk >= self.cfg.risk_warning_threshold:
                    decision = "BLOCK_AND_REFLECT"
                else:
                    decision = "CONTINUE"

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
        }
