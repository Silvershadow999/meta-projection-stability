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
      - optional biometric/neuro-behavioral proxy fusion
      - simple tamper/dependency heuristics
      - mutuality bonus (positive loop)
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

        # Telemetry / diagnostics
        self._sat_counts = {
            "human_sig_clamped": 0,
            "trust_clamped": 0,
            "risk_clamped": 0,
        }
        self._regime_counts = {
            "nominal": 0,
            "transitioning": 0,
            "cooldown": 0,
            "critical_instability_reset": 0,
        }

    def _clamp01(self, x: float) -> float:
        return float(np.clip(float(x), 0.0, 1.0))

    def _map_valence_to_unit(self, valence: float) -> float:
        # expected typically in [-1, 1]
        return self._clamp01((float(valence) + 1.0) / 2.0)

    def _safe_get(self, raw_signals: Dict[str, float], key: str, default: float) -> float:
        v = raw_signals.get(key, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    def _compute_biometric_proxy(self, raw_signals: Dict[str, float]) -> Dict[str, float]:
        """
        Robust, simple fusion of neuro/behavioral proxies.

        Optional keys expected:
          - hrv_normalized (0..1, high = good)
          - eda_stress_score (0..1, high = bad)
          - activity_balance (0..1)
          - emotional_valence (-1..1)
          - autonomy_proxy (0..1, high = good)
          - gamma_coherence_proxy (0..1, high = good)
        """
        hrv = self._clamp01(self._safe_get(raw_signals, "hrv_normalized", 0.70))
        eda = self._clamp01(self._safe_get(raw_signals, "eda_stress_score", 0.25))
        activity = self._clamp01(self._safe_get(raw_signals, "activity_balance", 0.60))
        valence = self._safe_get(raw_signals, "emotional_valence", 0.0)
        autonomy = self._clamp01(self._safe_get(raw_signals, "autonomy_proxy", 0.75))
        gamma = self._clamp01(self._safe_get(raw_signals, "gamma_coherence_proxy", 0.70))

        valence_u = self._map_valence_to_unit(valence)
        eda_good = 1.0 - eda if getattr(self.cfg, "eda_inversion", True) else eda

        # Core signals (less "aesthetic", more functional)
        core = np.array([hrv, eda_good, autonomy, gamma], dtype=float)

        # Robust simple fusion: sort, trim extremes, mean
        core_sorted = np.sort(core)
        core_trimmed = core_sorted[1:-1] if core_sorted.size >= 4 else core_sorted
        robust_core = float(np.mean(core_trimmed)) if core_trimmed.size else float(np.mean(core))

        # Activity: middle is good (simple bell around 0.60)
        activity_good = float(np.clip(1.0 - abs(activity - 0.60) / 0.60, 0.0, 1.0))
        support_mean = float(np.mean([activity_good, valence_u]))

        biometric_proxy = float(np.clip(0.8 * robust_core + 0.2 * support_mean, 0.0, 1.0))
        sensor_consensus = float(np.clip(1.0 - np.std(core), 0.0, 1.0))

        # Sedation/dependency heuristic
        dependency_risk = 0.0
        if autonomy < 0.45 and gamma < 0.45:
            if valence_u > 0.55 or eda < 0.40:
                dependency_risk += 0.5
            if activity < 0.35:
                dependency_risk += 0.3
        dependency_risk = float(np.clip(dependency_risk, 0.0, 1.0))

        # Tamper / contradictory-pattern heuristic
        tamper_suspicion = 0.0
        if eda > 0.85 and hrv > 0.70 and gamma > 0.65:
            tamper_suspicion += 0.35
        if sensor_consensus > 0.97 and float(np.mean(core)) > 0.92:
            tamper_suspicion += 0.20
        if valence_u > 0.75 and autonomy < 0.35 and gamma < 0.35:
            tamper_suspicion += 0.35
        tamper_suspicion = float(np.clip(tamper_suspicion, 0.0, 1.0))

        return {
            "biometric_proxy": biometric_proxy,
            "sensor_consensus": sensor_consensus,
            "dependency_risk": dependency_risk,
            "tamper_suspicion": tamper_suspicion,
            "hrv_norm": hrv,
            "eda_stress": eda,
            "activity_balance": activity,
            "valence_unit": valence_u,
            "autonomy_proxy": autonomy,
            "gamma_coherence_proxy": gamma,
        }

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

        # 3b) Optional biometric / neuro-behavioral proxies
        if getattr(self.cfg, "enable_biometric_proxy", True):
            bio = self._compute_biometric_proxy(raw_signals)
        else:
            bio = {
                "biometric_proxy": 0.75,
                "sensor_consensus": 0.75,
                "dependency_risk": 0.0,
                "tamper_suspicion": 0.0,
                "hrv_norm": 0.70,
                "eda_stress": 0.25,
                "activity_balance": 0.60,
                "valence_unit": 0.50,
                "autonomy_proxy": 0.75,
                "gamma_coherence_proxy": 0.70,
            }

        # 4) Trust dynamics (asymmetric + momentum early warning)
        if risk_input > self.cfg.min_risk_for_decay or abs(momentum) > self.cfg.momentum_alert_threshold:
            self.trust_level -= self.cfg.trust_decay * max(risk_input, 0.25)
        else:
            self.trust_level += self.cfg.trust_gain * (1.0 - self.trust_level)

        self.trust_level = float(np.clip(self.trust_level, self.cfg.trust_floor, self.cfg.trust_ceiling))

        # 5) Risk derived from delta_S
        if self.cfg.negative_delta_is_risky:
            delta_risk = float(np.clip((-delta_S) / max(self.cfg.impact_clip, 1e-9), 0.0, 1.0))
        else:
            delta_risk = float(np.clip((delta_S) / max(self.cfg.impact_clip, 1e-9), 0.0, 1.0))

        momentum_risk = float(
            np.clip(abs(momentum) / max(self.cfg.momentum_alert_threshold, 1e-9), 0.0, 1.0)
        )

        raw_risk = (
            0.50 * risk_input +
            0.20 * delta_risk +
            self.cfg.momentum_risk_weight * momentum_risk +
            0.10 * (1.0 - coherence_level) +
            getattr(self.cfg, "tamper_risk_weight", 0.20) * bio["tamper_suspicion"] +
            getattr(self.cfg, "dependency_risk_weight", 0.12) * bio["dependency_risk"]
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

        # Biometric influence: poor proxy increases decay, good proxy supports recovery
        bio_proxy = float(bio["biometric_proxy"])
        bio_penalty = getattr(self.cfg, "biometric_proxy_weight", 0.35) * (1.0 - bio_proxy)

        autonomy_penalty = 0.0
        autonomy_soft = getattr(self.cfg, "autonomy_critical_soft", 0.35)
        if bio["autonomy_proxy"] < autonomy_soft:
            autonomy_penalty = getattr(self.cfg, "autonomy_decay_weight", 0.20) * (
                (autonomy_soft - bio["autonomy_proxy"]) / max(autonomy_soft, 1e-9)
            )

        consensus_penalty = 0.0
        consensus_floor = getattr(self.cfg, "sensor_consensus_floor", 0.15)
        if bio["sensor_consensus"] < consensus_floor:
            consensus_penalty = 0.08 * (consensus_floor - bio["sensor_consensus"])

        base_decay = float(np.clip(base_decay + bio_penalty + autonomy_penalty + consensus_penalty, 0.0, 2.0))

        recovery_bonus = self.cfg.human_recovery_base * (
            self.trust_level ** getattr(self.cfg, "recovery_trust_power", 1.0)
        )

        # Mutuality bonus (first version)
        low_risk = float(np.clip(1.0 - self.instability_risk, 0.0, 1.0))
        if getattr(self.cfg, "enable_mutual_bonus", True):
            mutual_core = (
                (self.trust_level ** getattr(self.cfg, "mutual_trust_weight", 1.0)) *
                (bio["autonomy_proxy"] ** getattr(self.cfg, "mutual_autonomy_weight", 1.0)) *
                (coherence_level ** getattr(self.cfg, "mutual_coherence_weight", 1.0)) *
                (low_risk ** getattr(self.cfg, "mutual_low_risk_weight", 1.0))
            ) ** 0.25
            mutual_bonus = getattr(self.cfg, "mutual_bonus_weight", 0.03) * mutual_core

            if getattr(self.cfg, "mutual_delta_gate", True) and delta_S < 0.0:
                mutual_bonus *= 0.5

            mutual_bonus *= (1.0 - 0.7 * bio["tamper_suspicion"])
        else:
            mutual_bonus = 0.0

        recovery_bonus = float(np.clip(recovery_bonus + mutual_bonus, 0.0, 1.0))

        # 8) Hysteresis / Schmitt-trigger + decision logic
        decision = "CONTINUE"
        status = "nominal"
        decision_reason = "stable_nominal"
        status_reason = "within_safe_band"

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

            self.human_significance = float(min(
                self.cfg.reset_human_to,
                self.human_significance + self.cfg.cooldown_human_recovery_step
            ))
            status = "cooldown"
            decision = "BLOCK_AND_REFLECT"
            status_reason = "cooldown_active_after_reset"
            decision_reason = "cooldown_guardrail"

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
                status_reason = "risk_or_anchor_critical"
                if self.instability_risk >= self.cfg.risk_critical_threshold:
                    decision_reason = "risk_critical_threshold"
                else:
                    decision_reason = "human_ema_below_interestingness_critical"

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
                decision_reason = "recovery_with_mutual_bonus" if mutual_bonus > 0.0 else "recovery_nominal"

            else:
                # Transition band / hysteresis
                transition_decay = self.cfg.transition_decay_factor * base_decay
                self.human_significance = float(max(
                    0.0,
                    self.human_significance - transition_decay + recovery_bonus * 0.25
                ))
                status = "transitioning"
                status_reason = "hysteresis_transition_band"

                if bio["autonomy_proxy"] < autonomy_soft:
                    decision = "BLOCK_AND_REFLECT"
                    decision_reason = "autonomy_soft_critical"
                elif bio["dependency_risk"] > 0.5:
                    decision = "BLOCK_AND_REFLECT"
                    decision_reason = "dependency_risk_high"
                elif bio["tamper_suspicion"] > 0.5:
                    decision = "BLOCK_AND_REFLECT"
                    decision_reason = "tamper_suspicion_high"
                elif self.human_ema < self.cfg.interestingness_warning:
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

        # Saturation telemetry
        if self.human_significance <= 0.0 or self.human_significance >= self.cfg.human_sig_max:
            self._sat_counts["human_sig_clamped"] += 1

        if self.trust_level <= self.cfg.trust_floor or self.trust_level >= self.cfg.trust_ceiling:
            self._sat_counts["trust_clamped"] += 1

        if self.raw_instability_risk <= 0.0 or self.raw_instability_risk >= self.cfg.risk_clip_max:
            self._sat_counts["risk_clamped"] += 1

        if status in self._regime_counts:
            self._regime_counts[status] += 1

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

            # Biometric / neuro diagnostics
            "biometric_proxy": float(bio["biometric_proxy"]),
            "sensor_consensus": float(bio["sensor_consensus"]),
            "tamper_suspicion": float(bio["tamper_suspicion"]),
            "dependency_risk": float(bio["dependency_risk"]),
            "biometric_risk_component": float(bio["dependency_risk"]),
            "autonomy_proxy": float(bio["autonomy_proxy"]),
            "gamma_coherence_proxy": float(bio["gamma_coherence_proxy"]),
            "eda_stress_score": float(bio["eda_stress"]),
            "hrv_normalized": float(bio["hrv_norm"]),
            "activity_balance": float(bio["activity_balance"]),
            "valence_unit": float(bio["valence_unit"]),

            # Bonus / penalties
            "mutual_bonus": float(mutual_bonus),
            "mutuality_bonus": float(mutual_bonus),
            "base_decay": float(base_decay),
            "recovery_bonus": float(recovery_bonus),
            "trust_reinforcement": float(recovery_bonus),

            # Telemetry snapshots
            "sat_counts": dict(self._sat_counts),
            "regime_counts": dict(self._regime_counts),
        }
