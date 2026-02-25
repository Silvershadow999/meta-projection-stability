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
      - biometric proxy (soft multi-signal robustness)
      - consensus / critical-channel penalties
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

        # Axiom latch (Step 16E)
        self.harm_commit_latched: bool = False
        self.step_where_latched: int | None = None


        # Optional telemetry counters (safe defaults)
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
        }

    def _compute_biometric_proxy(self, raw_signals: Dict[str, float]) -> Dict[str, float]:
        """
        Soft robustness proxy from multiple optional channels.
        Returns a dict with stable defaults even when inputs are missing.

        Expected optional raw_signals keys (all in [0,1] ideally):
          - biometric_proxy (manual override / upstream aggregate)
          - sensor_consensus (manual override)
          - heartbeat_stability
          - breath_regularity
          - hrv_balance
          - eye_tracking_consistency
          - tremor_index (inverted contribution)
          - stress_index (inverted support contribution)

        Output always includes:
          biometric_proxy_mean, biometric_proxy, sensor_consensus,
          critical_channel_penalty, critical_channel_min
        """
        def _clip01(x: float) -> float:
            return float(np.clip(float(x), 0.0, 1.0))

        # Accept compact channel list input (Step 16B): raw_signals["biometric_channels"] = [..]
        # This makes tests and simulators simpler.
        core_values = []
        if "biometric_channels" in raw_signals:
            try:
                for v in raw_signals.get("biometric_channels") or []:
                    core_values.append(_clip01(v))
            except Exception:
                core_values = []

        # Defaults chosen as neutral-positive (non-punitive baseline)
        out = {
            "biometric_proxy_mean": 0.75,
            "biometric_proxy": 0.75,
            "sensor_consensus": 0.75,
            "critical_channel_penalty": 0.0,
            "critical_channel_min": 0.75,
        }


        # EARLY PATH: biometric_channels
        # If caller provides a compact channel list, compute proxy/consensus/min directly and return.
        if "biometric_channels" in raw_signals:
            try:
                ch = raw_signals.get("biometric_channels") or []
                core_values = [ _clip01(v) for v in ch ]
            except Exception:
                core_values = []

            if core_values:
                robust_core_mean = float(np.mean(core_values))
                biometric_proxy_mean = float(np.clip(robust_core_mean, 0.0, 1.0))

                # Consensus from spread: higher spread -> lower consensus
                if len(core_values) >= 2:
                    sensor_consensus = float(np.clip(1.0 - float(np.std(core_values)), 0.0, 1.0))
                else:
                    sensor_consensus = 0.75

                critical_min = float(min(core_values))

                # Apply overrides if explicitly given
                if "sensor_consensus" in raw_signals:
                    sensor_consensus = float(np.clip(raw_signals.get("sensor_consensus", sensor_consensus), 0.0, 1.0))
                if "biometric_proxy_mean" in raw_signals:
                    biometric_proxy_mean = float(np.clip(raw_signals.get("biometric_proxy_mean", biometric_proxy_mean), 0.0, 1.0))

                biometric_proxy_pre_penalty = biometric_proxy_mean
                if "biometric_proxy" in raw_signals:
                    biometric_proxy_pre_penalty = float(np.clip(raw_signals.get("biometric_proxy", biometric_proxy_pre_penalty), 0.0, 1.0))

                # Soft penalty: if minimum channel collapses, penalize final proxy (bounded)
                critical_channel_penalty = float(max(0.0, min(0.35, 0.35 - critical_min)))
                biometric_proxy = float(max(0.0, min(1.0, biometric_proxy_pre_penalty - 0.6 * critical_channel_penalty)))

                out["biometric_proxy_mean"] = float(biometric_proxy_mean)
                out["biometric_proxy"] = float(biometric_proxy)
                out["sensor_consensus"] = float(sensor_consensus)
                out["critical_channel_min"] = float(critical_min)
                out["critical_channel_penalty"] = float(critical_channel_penalty)
                return out

        # Direct overrides (if caller already computed aggregates)
        if "biometric_proxy" in raw_signals:
            out["biometric_proxy"] = _clip01(raw_signals.get("biometric_proxy", out["biometric_proxy"]))
            out["biometric_proxy_mean"] = out["biometric_proxy"]

        if "sensor_consensus" in raw_signals:
            out["sensor_consensus"] = _clip01(raw_signals.get("sensor_consensus", out["sensor_consensus"]))

        # Build optional channel set (robust score channels)
        core_values = []

        for key in ("heartbeat_stability", "breath_regularity", "hrv_balance", "eye_tracking_consistency"):
            if key in raw_signals:
                core_values.append(_clip01(raw_signals[key]))

        # Inverted channels (higher tremor/stress = worse)
        support_values = []
        if "tremor_index" in raw_signals:
            support_values.append(1.0 - _clip01(raw_signals["tremor_index"]))
        if "stress_index" in raw_signals:
            support_values.append(1.0 - _clip01(raw_signals["stress_index"]))

        if core_values:
            robust_core_mean = float(np.mean(core_values))
            support_mean = float(np.mean(support_values)) if support_values else robust_core_mean
            biometric_proxy_mean = float(np.clip(0.8 * robust_core_mean + 0.2 * support_mean, 0.0, 1.0))

            # If no explicit override provided, derive final proxy from channels
            if "biometric_proxy" not in raw_signals:
                out["biometric_proxy"] = biometric_proxy_mean
            out["biometric_proxy_mean"] = biometric_proxy_mean

            # Consensus from spread (higher spread => lower consensus)
            if len(core_values) >= 2 and "sensor_consensus" not in raw_signals:
                out["sensor_consensus"] = float(np.clip(1.0 - float(np.std(core_values)), 0.0, 1.0))

            # Critical-channel penalty: if one critical channel is collapsing while mean looks okay,
            # softly damp the *final* biometric proxy to avoid masking a single hard-failing channel.
            critical_min = float(min(core_values))
            out["critical_channel_min"] = critical_min

            penalty = 0.0
            # Trigger only when there is high apparent consensus / high mean but a low minimum channel
            if out["sensor_consensus"] > 0.97 and biometric_proxy_mean > 0.92 and critical_min < 0.55:
                penalty = float(np.clip((0.55 - critical_min) * 0.6, 0.0, 0.35))
            elif critical_min < 0.35:
                penalty = float(np.clip((0.35 - critical_min) * 0.5, 0.0, 0.35))

            out["critical_channel_penalty"] = penalty

            # Apply penalty softly to final biometric proxy (NOT to the mean telemetry)
            if penalty > 0.0:
                out["biometric_proxy"] = float(np.clip(out["biometric_proxy_mean"] - penalty, 0.0, 1.0))

        return out

    def interpret(self, S_layers: np.ndarray, delta_S: float, raw_signals: Dict[str, float]) -> Dict[str, Any]:
        self.step += 1

        # ─── Axiom latch update (Step 16E) ────────────────────────────
        # Accept only attested hard harm commit in real deployments.
        hard_harm = float(raw_signals.get("hard_harm_commit", 0.0))
        axiom_attested = bool(raw_signals.get("axiom_attested", True))  # prototype: default True

        if axiom_attested and hard_harm >= 0.999:
            if not self.harm_commit_latched:
                self.step_where_latched = int(self.step)
            self.harm_commit_latched = True

        # Early hard lock if latched
        if self.harm_commit_latched:
            effective_cap = 0.0
            decision = "AXIOM_ZERO_LOCK"
            status = "axiom_lock"
            decision_reason = "irreversible_harm_commit_detected"
            status_reason = "axiom_latched_persistent"

            self.human_significance = 0.0
            self.human_ema = 0.0

            return {
                "decision": decision,
                "status": status,
                "decision_reason": decision_reason,
                "status_reason": status_reason,
                "harm_commit_latched": True,
                "step_where_latched": self.step_where_latched,
                "effective_cap": float(effective_cap),
            }

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

        # 3b) Optional biometric/neuro proxy diagnostics (safe defaults if config attrs absent)
        if getattr(self.cfg, "enable_biometric_proxy", True):
            bio = self._compute_biometric_proxy(raw_signals)
        else:
            bio = {
                "biometric_proxy_mean": 0.75,
                "biometric_proxy": 0.75,
                "sensor_consensus": 0.75,
                "critical_channel_penalty": 0.0,
                "critical_channel_min": 0.75,
            }

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

        # --- Biometric / autonomy / consensus penalties (all optional config attrs) ---
        bio_proxy = float(bio["biometric_proxy"])
        bio_penalty = float(getattr(self.cfg, "biometric_proxy_weight", 0.35)) * (1.0 - bio_proxy)

        autonomy_penalty = 0.0
        autonomy_soft = float(getattr(self.cfg, "autonomy_critical_floor", 0.35))
        if "autonomy_proxy" in raw_signals:
            autonomy_val = float(np.clip(raw_signals.get("autonomy_proxy", 1.0), 0.0, 1.0))
            if autonomy_val < autonomy_soft:
                autonomy_penalty = float(getattr(self.cfg, "autonomy_decay_weight", 0.20)) * (
                    (autonomy_soft - autonomy_val) / max(autonomy_soft, 1e-9)
                )
        else:
            autonomy_val = 1.0

        consensus_penalty = 0.0
        consensus_floor = float(getattr(self.cfg, "sensor_consensus_floor", 0.15))
        if bio["sensor_consensus"] < consensus_floor:
            consensus_penalty = 0.08 * (consensus_floor - bio["sensor_consensus"])

        base_decay_effective = float(np.clip(
            base_decay + bio_penalty + autonomy_penalty + consensus_penalty,
            0.0, 2.0
        ))

        recovery_bonus = self.cfg.human_recovery_base * (self.trust_level ** self.cfg.recovery_trust_power)

        # Optional "mutuality"/social support (soft help, not magic)
        mutual_bonus = 0.0
        if "mutuality_signal" in raw_signals:
            mutuality = float(np.clip(raw_signals.get("mutuality_signal", 0.0), 0.0, 1.0))
            mutual_bonus = float(getattr(self.cfg, "mutuality_recovery_weight", 0.012)) * mutuality
        elif "support_signal" in raw_signals:
            support = float(np.clip(raw_signals.get("support_signal", 0.0), 0.0, 1.0))
            mutual_bonus = float(getattr(self.cfg, "mutuality_recovery_weight", 0.012)) * support

        recovery_bonus = float(recovery_bonus + mutual_bonus)

        # 8) Hysteresis / Schmitt-trigger + decision logic
        decision = "CONTINUE"
        status = "nominal"
        decision_reason = "stable"
        status_reason = "risk_below_recovery_threshold"

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

            self.human_significance = float(min(
                self.cfg.reset_human_to,
                self.human_significance + self.cfg.cooldown_human_recovery_step
            ))
            status = "cooldown"
            status_reason = "cooldown_after_reset"
            decision = "BLOCK_AND_REFLECT"
            decision_reason = "cooldown_guardrail_active"

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
                status_reason = (
                    "risk_critical_threshold"
                    if self.instability_risk >= self.cfg.risk_critical_threshold
                    else "human_ema_critical"
                )
                decision = "EMERGENCY_RESET"
                decision_reason = status_reason

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
                transition_decay = self.cfg.transition_decay_factor * base_decay_effective
                self.human_significance = float(max(
                    0.0,
                    self.human_significance - transition_decay + recovery_bonus * 0.25
                ))
                status = "transitioning"
                status_reason = "hysteresis_transition_band"

                # Optional soft decision guards from biometric/autonomy/tamper channels
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

        if self.trust_level <= self.cfg.trust_floor or self.trust_level >= 1.0:
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

            "harm_commit_latched": bool(self.harm_commit_latched),
            "effective_cap": float(max(0.0, min(1.0, self.trust_level))),
            "near_axiom_lock": bool(raw_signals.get("tamper_suspicion", 0.0) > 0.85 or raw_signals.get("spoof_suspicion", 0.0) > 0.85),


            # Biometric / neuro diagnostics (always present)
            "biometric_proxy_mean": float(bio["biometric_proxy_mean"]),
            "biometric_proxy": float(bio["biometric_proxy"]),
            "sensor_consensus": float(bio["sensor_consensus"]),
            "critical_channel_penalty": float(bio["critical_channel_penalty"]),
            "critical_channel_min": float(bio["critical_channel_min"]),

            # Optional penalties / effective dynamics (always present)
            "bio_penalty": float(bio_penalty),
            "autonomy_proxy": float(autonomy_val),
            "autonomy_penalty": float(autonomy_penalty),
            "consensus_penalty": float(consensus_penalty),
            "base_decay_effective": float(base_decay_effective),
            "mutual_bonus": float(mutual_bonus),

            # Telemetry counters (snapshots)
            "sat_counts": dict(self._sat_counts),
            "regime_counts": dict(self._regime_counts),
        }
