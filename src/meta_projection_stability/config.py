from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetaProjectionStabilityConfig:
    """
    Zentrale Konfiguration für Meta Projection Stability.

    Diese Version ist:
    - kompatibel mit der CLI (risk_* Felder)
    - erweiterbar für GlobalSense (human_recovery_base etc.)
    - defensiv validiert (keine negativen/unsinnigen Werte)
    """

    # ------------------------------------------------------------------
    # Core simulation defaults
    # ------------------------------------------------------------------
    n_steps_default: int = 1400
    levels_default: int = 3
    seed_default: int = 42

    # ------------------------------------------------------------------
    # Risk regulation / trust modulation (CLI nutzt diese explizit)
    # ------------------------------------------------------------------
    risk_critical_threshold: float = 0.58
    risk_recovery_threshold: float = 0.36
    risk_warning_threshold: float = 0.45

    momentum_risk_weight: float = 0.09
    ema_alpha_risk: float = 0.12

    # ------------------------------------------------------------------
    # Human-significance anchoring / recovery dynamics
    # (GlobalSense mappt primär auf human_recovery_base)
    # ------------------------------------------------------------------
    human_recovery_base: float = 1.0
    trust_anchor_strength: float = 0.50
    significance_anchor_strength: float = 0.50

    # Optional erweiterte Parameter (GlobalSense passt diese nur an, wenn vorhanden)
    noise_scale: float = 0.20
    system_drag: float = 0.10
    adaptation_rate: float = 0.05

    # ------------------------------------------------------------------
    # Additional optional knobs (harmlos, falls simulation.py sie ignoriert)
    # ------------------------------------------------------------------
    baseline_stability: float = 1.0
    coupling_strength: float = 0.40
    damping_factor: float = 0.15

    # Stress-window behaviour (falls simulation.py das nutzt)
    stress_window_start: int = 450
    stress_window_end: int = 650
    stress_intensity: float = 1.0

    # Plot / reporting toggles (falls genutzt)
    enable_logging: bool = True
    summary_precision: int = 4

    # ------------------------------------------------------------------
    # Internal baseline fields for GlobalSense (werden automatisch gesetzt)
    # ------------------------------------------------------------------
    _baseline_human_recovery_base: float = field(init=False, repr=False)
    _baseline_noise_scale: float = field(init=False, repr=False)
    _baseline_system_drag: float = field(init=False, repr=False)
    _baseline_adaptation_rate: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # ---- Basic numeric sanity checks ----
        self.n_steps_default = max(1, int(self.n_steps_default))
        self.levels_default = max(1, int(self.levels_default))
        self.seed_default = int(self.seed_default)

        self.risk_critical_threshold = float(self.risk_critical_threshold)
        self.risk_recovery_threshold = float(self.risk_recovery_threshold)
        self.risk_warning_threshold = float(self.risk_warning_threshold)

        self.momentum_risk_weight = float(self.momentum_risk_weight)
        self.ema_alpha_risk = float(self.ema_alpha_risk)

        self.human_recovery_base = float(self.human_recovery_base)
        self.trust_anchor_strength = float(self.trust_anchor_strength)
        self.significance_anchor_strength = float(self.significance_anchor_strength)

        self.noise_scale = float(self.noise_scale)
        self.system_drag = float(self.system_drag)
        self.adaptation_rate = float(self.adaptation_rate)

        self.baseline_stability = float(self.baseline_stability)
        self.coupling_strength = float(self.coupling_strength)
        self.damping_factor = float(self.damping_factor)

        self.stress_window_start = int(self.stress_window_start)
        self.stress_window_end = int(self.stress_window_end)
        self.stress_intensity = float(self.stress_intensity)

        self.summary_precision = max(0, int(self.summary_precision))

        # ---- Clamp ranges (defensive, without being too strict) ----
        self.risk_critical_threshold = self._clamp(self.risk_critical_threshold, 0.0, 1.0)
        self.risk_recovery_threshold = self._clamp(self.risk_recovery_threshold, 0.0, 1.0)
        self.risk_warning_threshold = self._clamp(self.risk_warning_threshold, 0.0, 1.0)

        self.ema_alpha_risk = self._clamp(self.ema_alpha_risk, 0.001, 1.0)
        self.momentum_risk_weight = self._clamp(self.momentum_risk_weight, 0.0, 5.0)

        self.human_recovery_base = max(0.0, self.human_recovery_base)
        self.trust_anchor_strength = self._clamp(self.trust_anchor_strength, 0.0, 5.0)
        self.significance_anchor_strength = self._clamp(self.significance_anchor_strength, 0.0, 5.0)

        self.noise_scale = max(0.0, self.noise_scale)
        self.system_drag = max(0.0, self.system_drag)
        self.adaptation_rate = max(0.0, self.adaptation_rate)

        self.baseline_stability = max(0.0, self.baseline_stability)
        self.coupling_strength = self._clamp(self.coupling_strength, 0.0, 10.0)
        self.damping_factor = self._clamp(self.damping_factor, 0.0, 10.0)

        self.stress_intensity = max(0.0, self.stress_intensity)

        # Fensterreihenfolge korrigieren, falls vertauscht
        if self.stress_window_end < self.stress_window_start:
            self.stress_window_start, self.stress_window_end = (
                self.stress_window_end,
                self.stress_window_start,
            )

        # Schwellen logisch ordnen (recovery <= warning <= critical)
        ordered = sorted(
            [self.risk_recovery_threshold, self.risk_warning_threshold, self.risk_critical_threshold]
        )
        self.risk_recovery_threshold, self.risk_warning_threshold, self.risk_critical_threshold = ordered

        # ---- Baselines für GlobalSense speichern ----
        self._baseline_human_recovery_base = self.human_recovery_base
        self._baseline_noise_scale = self.noise_scale
        self._baseline_system_drag = self.system_drag
        self._baseline_adaptation_rate = self.adaptation_rate

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def reset_globalsense_baselines(self) -> None:
        """
        Setzt die aktuell gespeicherten Baselines neu auf die derzeitigen Werte.
        Nützlich, wenn du manuell Parameter verändert hast und GlobalSense
        von dort aus weiter modulieren soll.
        """
        self._baseline_human_recovery_base = float(self.human_recovery_base)
        self._baseline_noise_scale = float(self.noise_scale)
        self._baseline_system_drag = float(self.system_drag)
        self._baseline_adaptation_rate = float(self.adaptation_rate)

    def to_dict(self) -> dict:
        """
        Praktische Serialisierung für Debugging / Logging.
        Interne Baselines werden absichtlich ausgelassen.
        """
        return {
            "n_steps_default": self.n_steps_default,
            "levels_default": self.levels_default,
            "seed_default": self.seed_default,
            "risk_critical_threshold": self.risk_critical_threshold,
            "risk_recovery_threshold": self.risk_recovery_threshold,
            "risk_warning_threshold": self.risk_warning_threshold,
            "momentum_risk_weight": self.momentum_risk_weight,
            "ema_alpha_risk": self.ema_alpha_risk,
            "human_recovery_base": self.human_recovery_base,
            "trust_anchor_strength": self.trust_anchor_strength,
            "significance_anchor_strength": self.significance_anchor_strength,
            "noise_scale": self.noise_scale,
            "system_drag": self.system_drag,
            "adaptation_rate": self.adaptation_rate,
            "baseline_stability": self.baseline_stability,
            "coupling_strength": self.coupling_strength,
            "damping_factor": self.damping_factor,
            "stress_window_start": self.stress_window_start,
            "stress_window_end": self.stress_window_end,
            "stress_intensity": self.stress_intensity,
            "enable_logging": self.enable_logging,
            "summary_precision": self.summary_precision,
        }
