from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np


BarometerStatus = Literal["GREEN", "YELLOW", "RED"]


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


@dataclass
class IntegrityBarometerConfig:
    """
    Konfiguration für das Integritäts-/Risikobarometer.

    Ziel:
    - stabile, geglättete Anzeige (EMA)
    - robuste Aggregation aus Kohärenz, Trust, Root-Stability, Somatic Anchor
    - klare Status-Ampel (GREEN / YELLOW / RED)
    """
    # Gewichte (werden intern normalisiert, falls Summe != 1)
    weight_root: float = 0.20
    weight_somatic: float = 0.20
    weight_coherence: float = 0.40
    weight_trust: float = 0.20

    # Glättung
    ema_alpha_risk: float = 0.15

    # Schwellwerte
    yellow_threshold: float = 0.40
    red_threshold: float = 0.70

    # Sicherheitsgrenzen / Bias
    risk_floor: float = 0.0
    risk_ceiling: float = 1.0

    # Optional: etwas Risiko-Abmilderung bei sehr hoher Kohärenz + Trust
    synergy_bonus: float = 0.10

    def __post_init__(self) -> None:
        self.weight_root = max(0.0, float(self.weight_root))
        self.weight_somatic = max(0.0, float(self.weight_somatic))
        self.weight_coherence = max(0.0, float(self.weight_coherence))
        self.weight_trust = max(0.0, float(self.weight_trust))

        self.ema_alpha_risk = _clip01(self.ema_alpha_risk)
        self.yellow_threshold = _clip01(self.yellow_threshold)
        self.red_threshold = _clip01(self.red_threshold)

        # defensive ordering
        if self.yellow_threshold > self.red_threshold:
            self.yellow_threshold, self.red_threshold = self.red_threshold, self.yellow_threshold

        self.risk_floor = _clip01(self.risk_floor)
        self.risk_ceiling = _clip01(self.risk_ceiling)
        if self.risk_floor > self.risk_ceiling:
            self.risk_floor, self.risk_ceiling = self.risk_ceiling, self.risk_floor

        self.synergy_bonus = _clip01(self.synergy_bonus)

    def normalized_weights(self) -> tuple[float, float, float, float]:
        w = np.array(
            [self.weight_root, self.weight_somatic, self.weight_coherence, self.weight_trust],
            dtype=float,
        )
        s = float(w.sum())
        if s <= 1e-12:
            # Fallback auf sinnvolle Default-Verteilung
            return (0.20, 0.20, 0.40, 0.20)
        w /= s
        return float(w[0]), float(w[1]), float(w[2]), float(w[3])


@dataclass
class IntegrityBarometerState:
    """
    Zustand des Barometers.
    """
    manip_risk: float = 0.0
    raw_risk: float = 0.0
    integrity_score: float = 1.0
    status: BarometerStatus = "GREEN"
    step_count: int = 0

    # Input-echo (hilfreich fürs Debugging / Plotting)
    root_stability: float = 1.0
    somatic_anchor: float = 1.0
    coherence_level: float = 1.0
    trust: float = 1.0


def _status_from_risk(risk: float, cfg: IntegrityBarometerConfig) -> BarometerStatus:
    if risk >= cfg.red_threshold:
        return "RED"
    if risk >= cfg.yellow_threshold:
        return "YELLOW"
    return "GREEN"


def _compute_integrity_score(
    root_stability: float,
    somatic_anchor: float,
    coherence_level: float,
    trust: float,
    cfg: IntegrityBarometerConfig,
) -> float:
    wr, ws, wc, wt = cfg.normalized_weights()

    root_stability = _clip01(root_stability)
    somatic_anchor = _clip01(somatic_anchor)
    coherence_level = _clip01(coherence_level)
    trust = _clip01(trust)

    # gewichtete Integrität
    base = (
        wr * root_stability
        + ws * somatic_anchor
        + wc * coherence_level
        + wt * trust
    )

    # Synergiebonus bei gleichzeitig guter Kohärenz + Trust
    synergy = cfg.synergy_bonus * max(0.0, min(coherence_level, trust) - 0.5) * 2.0
    score = _clip01(base + max(0.0, synergy))

    return score


def barometer_step(
    *,
    root_stability: float,
    somatic_anchor: float,
    coherence_level: float,
    trust: float,
    prev_state: Optional[IntegrityBarometerState] = None,
    config: Optional[IntegrityBarometerConfig] = None,
    dt: float = 0.1,
) -> IntegrityBarometerState:
    """
    Ein einzelner, stateless testbarer Integritäts-Barometer-Schritt.

    Inputs:
    - root_stability: stabiler Systemanker (0..1)
    - somatic_anchor: externer/sensorischer Anchor (0..1)
    - coherence_level: MPS-Kohärenz/Stabilitätswert (0..1)
    - trust: Trust-Wert (0..1)
    - prev_state: vorheriger Zustand (für EMA)
    - config: Konfiguration
    - dt: derzeit nicht zwingend genutzt, bleibt für API-Stabilität erhalten
    """
    _ = dt  # API-kompatibel, aktuell nicht verwendet
    cfg = config or IntegrityBarometerConfig()

    root_stability = _clip01(root_stability)
    somatic_anchor = _clip01(somatic_anchor)
    coherence_level = _clip01(coherence_level)
    trust = _clip01(trust)

    integrity_score = _compute_integrity_score(
        root_stability=root_stability,
        somatic_anchor=somatic_anchor,
        coherence_level=coherence_level,
        trust=trust,
        cfg=cfg,
    )

    # Risiko = inverse Integrität
    raw_risk = 1.0 - integrity_score
    raw_risk = float(np.clip(raw_risk, cfg.risk_floor, cfg.risk_ceiling))

    if prev_state is None:
        smoothed_risk = raw_risk
        step_count = 1
    else:
        alpha = cfg.ema_alpha_risk
        smoothed_risk = alpha * raw_risk + (1.0 - alpha) * float(prev_state.manip_risk)
        smoothed_risk = float(np.clip(smoothed_risk, cfg.risk_floor, cfg.risk_ceiling))
        step_count = int(prev_state.step_count) + 1

    status = _status_from_risk(smoothed_risk, cfg)

    return IntegrityBarometerState(
        manip_risk=smoothed_risk,
        raw_risk=raw_risk,
        integrity_score=integrity_score,
        status=status,
        step_count=step_count,
        root_stability=root_stability,
        somatic_anchor=somatic_anchor,
        coherence_level=coherence_level,
        trust=trust,
    )


if __name__ == "__main__":
    cfg = IntegrityBarometerConfig(weight_coherence=0.60, ema_alpha_risk=0.15)
    s = None

    demo_inputs = [
        (1.0, 0.9, 0.95, 0.9),
        (1.0, 0.8, 0.70, 0.8),
        (1.0, 0.7, 0.45, 0.7),
        (1.0, 0.6, 0.25, 0.5),
    ]

    for i, (r0, sa, coh, tr) in enumerate(demo_inputs, start=1):
        s = barometer_step(
            root_stability=r0,
            somatic_anchor=sa,
            coherence_level=coh,
            trust=tr,
            prev_state=s,
            config=cfg,
            dt=0.1,
        )
        print(
            f"[{i}] integrity={s.integrity_score:.3f} raw_risk={s.raw_risk:.3f} "
            f"risk={s.manip_risk:.3f} status={s.status}"
        )
