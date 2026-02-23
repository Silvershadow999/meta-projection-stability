"""
MetaProjectionStabilityConfig – zentrale Konfiguration für das Projekt

Enthält alle einstellbaren Parameter, Defaults und Schutzmechanismen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MetaProjectionStabilityConfig:
    """
    Zentrale Konfigurationsklasse für das meta-projection-stability System.

    Enthält alle einstellbaren Parameter, Default-Werte und Schutzmechanismen
    gegen ungültige Werte (Clamping, Threshold-Ordering, Alias-Unterstützung).
    """

    # ─── Laufzeit & Simulation ────────────────────────────────────────
    n_steps: int = 300
    dt: float = 1.0
    seed: int = 42
    enable_plot: bool = True

    # ─── Glättung & Gedächtnis ────────────────────────────────────────
    ema_alpha: float = 0.15
    momentum_alpha: float = 0.25
    delta_history_len: int = 64
    coherence_normalizer: float = 1.0

    # ─── Risk-System (0..1) ────────────────────────────────────────────
    risk_critical: float = 0.85
    risk_recover: float = 0.55
    risk_warn: float = 0.65
    risk_floor: float = 0.00
    risk_ceiling: float = 1.00
    min_risk_for_decay: float = 0.05

    # ─── Trust & Human Significance ────────────────────────────────────
    trust_floor: float = 0.05
    trust_ceiling: float = 1.00
    trust_init: float = 0.75
    trust_decay: float = 0.01
    trust_gain: float = 0.02
    trust_flow: float = 0.10  # breiter Fallback für Experimente

    human_significance_init: float = 0.80
    human_significance_floor: float = 0.05
    human_significance_ceiling: float = 1.00

    impact_clip: float = 1.00
    impact_scale: float = 1.00

    # ─── Interestingness & Adaptation ──────────────────────────────────
    interestingness_critical: float = 0.30
    interestingness_warn: float = 0.45
    interestingness_target: float = 0.60

    # ─── Recovery / Reset / Anchoring ──────────────────────────────────
    human_recovery_base: float = 0.02
    human_recovery_gain: float = 0.015
    reset_human_to: float = 0.70

    anchoring_strength: float = 0.20
    significance_anchor_weight: float = 0.25

    # ─── Governor-Thresholds (Kompatibilität mit governor.py) ──────────
    collaboration_threshold: float = 0.70
    safety_threshold: float = 0.40

    # Diese dürfen > 1.0 sein (asymmetrische Penalty-Logik!)
    risk_penalty_factor: float = 1.80
    trust_penalty_factor: float = 1.20

    autonomy_floor: float = 0.05
    autonomy_ceiling: float = 1.00

    # ─── Optionale Globale Einflüsse ───────────────────────────────────
    use_global_sense: bool = False
    worldbank_indicator: str = "NY.GDP.MKTP.KD.ZG"  # Beispiel: GDP growth
    globalsense_scale: float = 1.0

    # ─── Debugging & Kompatibilität ────────────────────────────────────
    verbose: bool = False
    debug: bool = False
    profile_name: str = "default"

    # Optionales Metadaten-Feld (nicht zwingend genutzt, aber praktisch)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialisiert Aliase und führt Sicherheits-Clamping durch."""
        self._clamp_and_normalize()
        self._create_aliases()

    def _clamp_and_normalize(self) -> None:
        """Sichert Wertebereiche und stellt konsistente Reihenfolge her."""
        fields_0_to_1 = [
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

        for fname in fields_0_to_1:
            v = getattr(self, fname, None)
            if isinstance(v, (int, float)):
                setattr(self, fname, max(0.0, min(1.0, float(v))))

        # Spezielle positive Werte (dürfen >1 sein)
        self.coherence_normalizer = max(1e-9, float(self.coherence_normalizer or 1.0))
        self.risk_penalty_factor = max(0.0, float(self.risk_penalty_factor or 1.0))
        self.trust_penalty_factor = max(0.0, float(self.trust_penalty_factor or 1.0))

        # Integer / positive Werte
        self.n_steps = max(1, int(self.n_steps))
        self.delta_history_len = max(2, int(self.delta_history_len or 64))
        self.dt = max(1e-6, float(self.dt))
        self.seed = max(0, int(self.seed))

        # Ceiling/Floor-Konsistenz
        for pair in ["risk", "trust", "human_significance", "autonomy"]:
            floor_v = float(getattr(self, f"{pair}_floor", 0.0))
            ceiling_v = float(getattr(self, f"{pair}_ceiling", 1.0))
            low = min(floor_v, ceiling_v)
            high = max(floor_v, ceiling_v)
            setattr(self, f"{pair}_floor", low)
            setattr(self, f"{pair}_ceiling", high)

        # Threshold-Logik (defensive ordering)
        if self.safety_threshold > self.collaboration_threshold:
            self.safety_threshold, self.collaboration_threshold = (
                self.collaboration_threshold,
                self.safety_threshold,
            )

        # Risk-Reihenfolge recover ≤ warn ≤ critical
        thresholds = sorted(
            [
                float(self.risk_recover),
                float(self.risk_warn),
                float(self.risk_critical),
            ]
        )
        self.risk_recover, self.risk_warn, self.risk_critical = thresholds

        # Risk-Thresholds zusätzlich in [risk_floor, risk_ceiling] einhegen
        self.risk_recover = min(max(self.risk_recover, self.risk_floor), self.risk_ceiling)
        self.risk_warn = min(max(self.risk_warn, self.risk_floor), self.risk_ceiling)
        self.risk_critical = min(max(self.risk_critical, self.risk_floor), self.risk_ceiling)

        # Initialwerte innerhalb Floor/Ceiling halten
        self.trust_init = min(max(self.trust_init, self.trust_floor), self.trust_ceiling)
        self.human_significance_init = min(
            max(self.human_significance_init, self.human_significance_floor),
            self.human_significance_ceiling,
        )

    def _create_aliases(self) -> None:
        """Legt gängige Alias-Attribute an (für Kompatibilität)."""
        aliases = {
            # Risk
            "risk_min": self.risk_floor,
            "risk_max": self.risk_ceiling,
            # Trust
            "trust_min": self.trust_floor,
            "trust_max": self.trust_ceiling,
            # Governor
            "collab_th": self.collaboration_threshold,
            "collab_threshold": self.collaboration_threshold,
            "safety_th": self.safety_threshold,
            "autonomy_min": self.autonomy_floor,
            "autonomy_max": self.autonomy_ceiling,
            # Penalty
            "risk_p": self.risk_penalty_factor,
            "trust_p": self.trust_penalty_factor,
            "risk_penalty": self.risk_penalty_factor,
            "trust_penalty": self.trust_penalty_factor,
            # Sonstiges
            "interest_critical": self.interestingness_critical,
            "interest_warn": self.interestingness_warn,
            "interest_target": self.interestingness_target,
            "human_reset_to": self.reset_human_to,
            "recovery_base": self.human_recovery_base,
            "plot": self.enable_plot,
            "steps": self.n_steps,
        }

        for name, value in aliases.items():
            setattr(self, name, value)

    def to_dict(self) -> Dict[str, Any]:
        """Gibt die echten Dataclass-Felder als dict zurück (keine Aliase)."""
        return asdict(self)

    def update(self, **kwargs: Any) -> "MetaProjectionStabilityConfig":
        """
        Aktualisiert Felder chainable und sicher.
        Ungültige Keys werden ignoriert (kein Fehler).
        """
        valid_fields = {f.name for f in fields(self)}
        for key, value in kwargs.items():
            if key in valid_fields:
                setattr(self, key, value)

        # Nach Update normalisieren + Aliase neu erzeugen
        self.__post_init__()
        return self


# ─── Convenience-Funktionen ─────────────────────────────────────────────

def get_default_config() -> MetaProjectionStabilityConfig:
    """Erzeugt eine saubere Standard-Konfiguration."""
    return MetaProjectionStabilityConfig()


def print_config(config: MetaProjectionStabilityConfig | None = None) -> None:
    """
    Druckt die Konfiguration lesbar aus (hauptsächlich für Debugging/CLI).
    """
    cfg = config or get_default_config()
    data = cfg.to_dict()

    print("MetaProjectionStabilityConfig")
    print("─" * 50)
    for key in sorted(data):
        print(f"{key:<28}: {data[key]}")
    print("─" * 50)


if __name__ == "__main__":
    # Quick-Test
    cfg = get_default_config()
    print_config(cfg)

    # Chainable Update-Beispiel
    cfg.update(n_steps=500, debug=True)
    print("\nNach Update:")
    print_config(cfg)
