from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any, Dict


@dataclass
class MetaProjectionStabilityConfig:
    """
    Zentrale Konfiguration für meta_projection_stability.

    Ziel dieser Version:
    - breite Kompatibilität mit älteren / experimentellen Codepfaden
    - viele Fallback-Namen (Alias-Attribute) gegen AttributeError-Schleifen
    - keine Imports aus governor.py (wichtig!)
    """

    # ------------------------------------------------------------
    # Core simulation shape / runtime
    # ------------------------------------------------------------
    n_steps: int = 300
    dt: float = 1.0
    seed: int = 42
    enable_plot: bool = True

    # ------------------------------------------------------------
    # Signal smoothing / memory / dynamics
    # ------------------------------------------------------------
    ema_alpha: float = 0.15
    momentum_alpha: float = 0.25
    delta_history_len: int = 64
    coherence_normalizer: float = 1.0

    # ------------------------------------------------------------
    # Risk system (normalized ranges 0..1)
    # ------------------------------------------------------------
    risk_critical: float = 0.85
    risk_recover: float = 0.55
    risk_warn: float = 0.65
    risk_floor: float = 0.00
    risk_ceiling: float = 1.00

    # Minimum risk used in decays / exponentials to avoid zero-lock
    min_risk_for_decay: float = 0.05

    # ------------------------------------------------------------
    # Trust / human significance / impact
    # ------------------------------------------------------------
    trust_floor: float = 0.05
    trust_ceiling: float = 1.00
    trust_init: float = 0.75
    trust_decay: float = 0.01
    trust_gain: float = 0.02
    trust_flow: float = 0.10   # broad fallback for experiments

    human_significance_init: float = 0.80
    human_significance_floor: float = 0.05
    human_significance_ceiling: float = 1.00

    impact_clip: float = 1.00
    impact_scale: float = 1.00

    # ------------------------------------------------------------
    # Interest / relevance / adaptation
    # ------------------------------------------------------------
    interestingness_critical: float = 0.30
    interestingness_warn: float = 0.45
    interestingness_target: float = 0.60

    # ------------------------------------------------------------
    # Human recovery / reset / anchoring
    # ------------------------------------------------------------
    human_recovery_base: float = 0.02
    human_recovery_gain: float = 0.015
    reset_human_to: float = 0.70

    anchoring_strength: float = 0.20
    significance_anchor_weight: float = 0.25

    # ------------------------------------------------------------
    # Governor thresholds (for governor.py compatibility)
    # ------------------------------------------------------------
    collaboration_threshold: float = 0.70
    safety_threshold: float = 0.40

    risk_penalty_factor: float = 1.80
    trust_penalty_factor: float = 1.20

    autonomy_floor: float = 0.05
    autonomy_ceiling: float = 1.00

    # ------------------------------------------------------------
    # Optional world / globalsense toggles
    # ------------------------------------------------------------
    use_global_sense: bool = False
    worldbank_indicator: str = "NY.GDP.MKTP.KD.ZG"  # World GDP growth
    globalsense_scale: float = 1.0

    # ------------------------------------------------------------
    # Misc / debug / compatibility
    # ------------------------------------------------------------
    verbose: bool = False
    debug: bool = False
    profile_name: str = "default"

    def __post_init__(self) -> None:
        """
        Setzt zusätzliche Alias-Attribute, damit ältere/andere Module
        nicht an unterschiedlichen Feldnamen scheitern.
        """
        # Numerische Klammerung / Absicherung
        self._clamp_basics()

        # Alias-/Fallback-Namen anlegen (nur setzen, wenn nicht vorhanden)
        aliases: Dict[str, Any] = {
            # ---- häufige alternative Namen für trust / risk ----
            "risk_min": self.risk_floor,
            "risk_max": self.risk_ceiling,
            "trust_min": self.trust_floor,
            "trust_max": self.trust_ceiling,

            # ---- Governor / autonomy ----
            "collab_th": self.collaboration_threshold,
            "collab_threshold": self.collaboration_threshold,
            "safety_th": self.safety_threshold,
            "autonomy_min": self.autonomy_floor,
            "autonomy_max": self.autonomy_ceiling,

            # ---- penalty aliases ----
            "risk_p": self.risk_penalty_factor,
            "trust_p": self.trust_penalty_factor,
            "risk_penalty": self.risk_penalty_factor,
            "trust_penalty": self.trust_penalty_factor,

            # ---- coherence / memory aliases ----
            "coherence_scale": self.coherence_normalizer,
            "delta_window": self.delta_history_len,
            "history_len": self.delta_history_len,

            # ---- impact / clipping aliases ----
            "impact_limit": self.impact_clip,
            "impact_cap": self.impact_clip,

            # ---- interestingness aliases ----
            "interest_critical": self.interestingness_critical,
            "interest_warn": self.interestingness_warn,
            "interest_target": self.interestingness_target,

            # ---- reset / recovery aliases ----
            "human_reset_value": self.reset_human_to,
            "human_reset_to": self.reset_human_to,
            "recovery_base": self.human_recovery_base,

            # ---- plot / runtime aliases ----
            "plot": self.enable_plot,
            "steps": self.n_steps,
        }

        for name, value in aliases.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def _clamp_basics(self) -> None:
        """Sichert ein paar Basisbereiche gegen Ausreißer / Tippfehler."""
        # 0..1 clamps
        one_range_fields = [
            "ema_alpha",
            "momentum_alpha",
            "risk_floor",
            "risk_ceiling",
            "min_risk_for_decay",
            "trust_floor",
            "trust_ceiling",
            "trust_init",
            "trust_decay",
            "trust_gain",
            "trust_flow",
            "human_significance_init",
            "human_significance_floor",
            "human_significance_ceiling",
            "impact_clip",
            "impact_scale",
            "interestingness_critical",
            "interestingness_warn",
            "interestingness_target",
            "human_recovery_base",
            "human_recovery_gain",
            "reset_human_to",
            "anchoring_strength",
            "significance_anchor_weight",
            "collaboration_threshold",
            "safety_threshold",
            "autonomy_floor",
            "autonomy_ceiling",
            "globalsense_scale",
        ]

        for fname in one_range_fields:
            v = getattr(self, fname, None)
            if isinstance(v, (int, float)):
                setattr(self, fname, float(max(0.0, min(1.0, v))))

        # non-negative / sane ints
        if self.n_steps < 1:
            self.n_steps = 1
        if self.delta_history_len < 2:
            self.delta_history_len = 2
        if self.dt <= 0:
            self.dt = 1.0
        if self.seed < 0:
            self.seed = 0

        # Ensure ceilings/floors are ordered
        if self.risk_floor > self.risk_ceiling:
            self.risk_floor, self.risk_ceiling = self.risk_ceiling, self.risk_floor

        if self.trust_floor > self.trust_ceiling:
            self.trust_floor, self.trust_ceiling = self.trust_ceiling, self.trust_floor

        if self.human_significance_floor > self.human_significance_ceiling:
            self.human_significance_floor, self.human_significance_ceiling = (
                self.human_significance_ceiling,
                self.human_significance_floor,
            )

        if self.autonomy_floor > self.autonomy_ceiling:
            self.autonomy_floor, self.autonomy_ceiling = self.autonomy_ceiling, self.autonomy_floor

        # Threshold consistency (soft correction)
        # safety threshold should generally be <= collaboration threshold
        if self.safety_threshold > self.collaboration_threshold:
            self.safety_threshold = min(self.safety_threshold, 0.40)
            self.collaboration_threshold = max(self.collaboration_threshold, 0.70)

        # Risk thresholds ordering (soft)
        # recover <= warn <= critical is a useful default
        ordered = sorted([self.risk_recover, self.risk_warn, self.risk_critical])
        self.risk_recover, self.risk_warn, self.risk_critical = ordered[0], ordered[1], ordered[2]

    def to_dict(self) -> Dict[str, Any]:
        """Saubere dict-Ausgabe (Dataclass-Felder, nicht alle Aliase)."""
        return asdict(self)

    def update(self, **kwargs: Any) -> "MetaProjectionStabilityConfig":
        """
        Komfort-Update (chainable), z. B.:
        cfg = MetaProjectionStabilityConfig().update(n_steps=1000, debug=True)
        """
        valid = {f.name for f in fields(self)}
        for key, val in kwargs.items():
            if key in valid:
                setattr(self, key, val)
            else:
                # Unbekannte Keys als Alias/Fallback zulassen, um harte Fehler zu vermeiden
                setattr(self, key, val)

        # Nach Update erneut normalisieren + Aliase refreshen
        self.__post_init__()
        return self


# ------------------------------------------------------------
# Convenience helpers (optional, aber praktisch für CLI/Debug)
# ------------------------------------------------------------
def get_default_config() -> MetaProjectionStabilityConfig:
    """Erzeugt eine robuste Standard-Config."""
    return MetaProjectionStabilityConfig()


def show_config(config: MetaProjectionStabilityConfig | None = None) -> Dict[str, Any]:
    """
    Gibt Konfiguration als dict zurück und druckt optional lesbar aus.
    Praktisch für schnellen Codespaces-Check.
    """
    cfg = config or get_default_config()
    data = cfg.to_dict()

    print("MetaProjectionStabilityConfig")
    print("-" * 40)
    for k in sorted(data.keys()):
        print(f"{k}: {data[k]}")
    return data
