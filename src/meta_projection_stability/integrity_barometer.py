from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


@dataclass
class IntegrityBarometerConfig:
    """
    Konfiguration für den Integritäts-/Manipulations-Barometer.

    Ziel:
    - Aus mehreren Faktoren einen stabilen Integritätsscore (0..1) ableiten
    - Daraus ein Manipulationsrisiko (0..1) berechnen
    - Einfache Ampel-Klassifikation (GREEN/YELLOW/RED)
    - Optional geglättetes Risiko (EMA), damit nicht jeder Spike sofort eskaliert
    """

    # Schwellen für die Ampel auf Basis des RISIKOS
    green_risk_threshold: float = 0.20   # risk <= 0.20 => GREEN
    yellow_risk_threshold: float = 0.50  # risk <= 0.50 => YELLOW, sonst RED

    # Integritäts-Baseline: Werte unterhalb der Baseline erhöhen Risiko stärker
    integrity_floor: float = 0.10

    # EMA-Glättung für Risiko
    ema_alpha_risk: float = 0.20

    # Grace-Period (optional) für aufrufende Systeme; hier nur mitgeführt
    grace_period_s: float = 0.0

    # Gewichtung der Integritätsfaktoren (werden normalisiert)
    weight_root: float = 0.40
    weight_somatic: float = 0.20
    weight_coherence: float = 0.25
    weight_trust: float = 0.15

    def __post_init__(self):
        self._clamp()

    def _clamp(self) -> None:
        self.green_risk_threshold = _clip01(self.green_risk_threshold)
        self.yellow_risk_threshold = _clip01(self.yellow_risk_threshold)

        # defensive ordering: green <= yellow
        if self.green_risk_threshold > self.yellow_risk_threshold:
            self.green_risk_threshold, self.yellow_risk_threshold = (
                self.yellow_risk_threshold,
                self.green_risk_threshold,
            )

        self.integrity_floor = _clip01(self.integrity_floor)
        self.ema_alpha_risk = _clip01(self.ema_alpha_risk)
        self.grace_period_s = max(0.0, _safe_float(self.grace_period_s, 0.0))

        # normalize positive weights (fallback to defaults if degenerate)
        weights = np.array(
            [
                max(0.0, _safe_float(self.weight_root, 0.40)),
                max(0.0, _safe_float(self.weight_somatic, 0.20)),
                max(0.0, _safe_float(self.weight_coherence, 0.25)),
                max(0.0, _safe_float(self.weight_trust, 0.15)),
            ],
            dtype=float,
        )
        s = float(np.sum(weights))
        if s <= 0.0:
            weights = np.array([0.40, 0.20, 0.25, 0.15], dtype=float)
            s = float(np.sum(weights))

        weights = weights / s
        self.weight_root = float(weights[0])
        self.weight_somatic = float(weights[1])
        self.weight_coherence = float(weights[2])
        self.weight_trust = float(weights[3])


@dataclass
class IntegrityBarometerState:
    """
    Zustand des Integritäts-Barometers für einen Simulationsschritt.
    """
    # Rohinputs (sichtbar für Debug/Plots)
    root_stability: float = 0.0
    somatic_anchor: float = 0.0
    coherence_level: float = 0.0
    trust: float = 0.0

    # Abgeleitete Größen
    integrity_score: float = 0.0          # 0..1
    manip_risk_raw: float = 0.0           # 0..1 (vor EMA)
    manip_risk: float = 0.0               # 0..1 (nach EMA)
    status: str = "IDLE"                  # GREEN / YELLOW / RED / IDLE
    time_s: float = 0.0

    # Sichtbare Schwellen / Metadaten (praktisch für Plot/Debug)
    green_risk_threshold: float = 0.20
    yellow_risk_threshold: float = 0.50
    integrity_floor: float = 0.10
    ema_alpha_risk: float = 0.20

    notes: list[str] = field(default_factory=list)


def classify_barometer_status(
    manip_risk: float,
    green_risk_threshold: float = 0.20,
    yellow_risk_threshold: float = 0.50,
) -> str:
    """
    Klassifiziert die Ampel anhand des Manipulationsrisikos.

    - GREEN:  risk <= green_threshold
    - YELLOW: risk <= yellow_threshold
    - RED:    risk > yellow_threshold
    """
    r = _clip01(manip_risk)
    g = _clip01(green_risk_threshold)
    y = _clip01(yellow_risk_threshold)

    if g > y:
        g, y = y, g

    if r <= g:
        return "GREEN"
    if r <= y:
        return "YELLOW"
    return "RED"


def compute_integrity_score(
    root_stability: float,
    somatic_anchor: float,
    coherence_level: float,
    trust: float,
    config: Optional[IntegrityBarometerConfig] = None,
) -> float:
    """
    Berechnet einen gewichteten Integritätsscore (0..1).

    Inputs:
    - root_stability: Stabilität des Root-/Baseline-Layers
    - somatic_anchor: somatische/operative Verankerung (oder Ersatzsignal)
    - coherence_level: Kohärenz-/Signalqualität
    - trust: externer oder interner Trust-Wert

    Die Gewichte kommen aus IntegrityBarometerConfig und werden dort normalisiert.
    """
    cfg = config or IntegrityBarometerConfig()

    r = _clip01(root_stability)
    s = _clip01(somatic_anchor)
    c = _clip01(coherence_level)
    t = _clip01(trust)

    score = (
        cfg.weight_root * r
        + cfg.weight_somatic * s
        + cfg.weight_coherence * c
        + cfg.weight_trust * t
    )
    return _clip01(score)


def compute_manipulation_risk(
    integrity_score: float,
    integrity_floor: float = 0.10,
) -> float:
    """
    Leitet ein Manipulationsrisiko (0..1) aus dem Integritätsscore ab.

    Design:
    - Risiko sinkt bei hoher Integrität
    - integrity_floor wirkt als Baseline: unterhalb der Floor steigt Risiko stärker
    """
    score = _clip01(integrity_score)
    floor = _clip01(integrity_floor)

    # Baseline-adjusted integrity
    baseline_adjusted = max(0.0, score - floor)
    return _clip01(1.0 - baseline_adjusted)


def barometer_step(
    root_stability: float,
    somatic_anchor: float,
    coherence_level: float,
    trust: float,
    prev_state: Optional[IntegrityBarometerState] = None,
    config: Optional[IntegrityBarometerConfig] = None,
    dt: float = 0.0,
) -> IntegrityBarometerState:
    """
    Ein Simulationsschritt des Integritäts-Barometers.

    Ablauf:
    1) Integritätsscore berechnen
    2) Roh-Risiko ableiten
    3) Risiko via EMA glätten (mit prev_state)
    4) Ampelstatus klassifizieren
    """
    cfg = config or IntegrityBarometerConfig()
    dt = max(0.0, _safe_float(dt, 0.0))

    score = compute_integrity_score(
        root_stability=root_stability,
        somatic_anchor=somatic_anchor,
        coherence_level=coherence_level,
        trust=trust,
        config=cfg,
    )
    risk_raw = compute_manipulation_risk(
        integrity_score=score,
        integrity_floor=cfg.integrity_floor,
    )

    prev_risk = 0.0 if prev_state is None else _clip01(prev_state.manip_risk)
    alpha = cfg.ema_alpha_risk
    risk = _clip01(alpha * risk_raw + (1.0 - alpha) * prev_risk)

    status = classify_barometer_status(
        manip_risk=risk,
        green_risk_threshold=cfg.green_risk_threshold,
        yellow_risk_threshold=cfg.yellow_risk_threshold,
    )

    time_s = dt if prev_state is None else float(prev_state.time_s + dt)

    state = IntegrityBarometerState(
        root_stability=_clip01(root_stability),
        somatic_anchor=_clip01(somatic_anchor),
        coherence_level=_clip01(coherence_level),
        trust=_clip01(trust),
        integrity_score=score,
        manip_risk_raw=risk_raw,
        manip_risk=risk,
        status=status,
        time_s=time_s,
        green_risk_threshold=cfg.green_risk_threshold,
        yellow_risk_threshold=cfg.yellow_risk_threshold,
        integrity_floor=cfg.integrity_floor,
        ema_alpha_risk=cfg.ema_alpha_risk,
    )

    # optional notes for debugging transitions
    if prev_state is not None and prev_state.status != status:
        state.notes.append(f"status_change: {prev_state.status} -> {status}")

    return state


def summarize_barometer_history(history: list[IntegrityBarometerState]) -> Dict[str, float | int]:
    """
    Kompakte Auswertung für CLI / Debugging:
    - Anzahl Statuswechsel
    - Anteil GREEN/YELLOW/RED
    - erste RED-Stelle
    - mittlere Integrität / mittleres Risiko
    """
    n = len(history)
    if n == 0:
        return {
            "n_steps": 0,
            "status_switches": 0,
            "green_ratio": 0.0,
            "yellow_ratio": 0.0,
            "red_ratio": 0.0,
            "first_red_idx": -1,
            "mean_integrity_score": 0.0,
            "mean_manip_risk": 0.0,
        }

    statuses = [h.status for h in history]
    scores = [float(h.integrity_score) for h in history]
    risks = [float(h.manip_risk) for h in history]

    status_switches = 0
    for i in range(1, n):
        if statuses[i] != statuses[i - 1]:
            status_switches += 1

    green_count = sum(1 for s in statuses if s == "GREEN")
    yellow_count = sum(1 for s in statuses if s == "YELLOW")
    red_count = sum(1 for s in statuses if s == "RED")

    first_red = next((i for i, s in enumerate(statuses) if s == "RED"), -1)

    return {
        "n_steps": n,
        "status_switches": status_switches,
        "green_ratio": green_count / n,
        "yellow_ratio": yellow_count / n,
        "red_ratio": red_count / n,
        "first_red_idx": first_red,
        "mean_integrity_score": float(np.mean(scores)) if scores else 0.0,
        "mean_manip_risk": float(np.mean(risks)) if risks else 0.0,
    }


def map_barometer_to_governor_inputs(
    state: IntegrityBarometerState,
    trust_override: Optional[float] = None,
) -> Dict[str, float]:
    """
    Kopplungsschritt Barometer -> Governor (neutral gehalten):

    Governor erwartet typischerweise:
    - trust (0..1)
    - risk  (0..1)

    Standard-Strategie:
    - trust = Integrity-Score (oder override)
    - risk  = manip_risk
    """
    trust = _clip01(state.integrity_score if trust_override is None else trust_override)
    risk = _clip01(state.manip_risk)
    return {"trust": trust, "risk": risk}


def compute_barometer_history(
    root_hist,
    somatic_hist,
    coherence_hist,
    trust_hist,
    config: Optional[IntegrityBarometerConfig] = None,
    dt: float = 0.0,
) -> list[IntegrityBarometerState]:
    """
    Hilfsfunktion für Zeitreihen:
    Berechnet die Barometer-Historie aus vier gleich langen Input-Serien.
    """
    cfg = config or IntegrityBarometerConfig()

    r = np.asarray(root_hist, dtype=float)
    s = np.asarray(somatic_hist, dtype=float)
    c = np.asarray(coherence_hist, dtype=float)
    t = np.asarray(trust_hist, dtype=float)

    if not (r.shape == s.shape == c.shape == t.shape):
        raise ValueError("All histories must have the same shape")

    out: list[IntegrityBarometerState] = []
    prev: Optional[IntegrityBarometerState] = None

    for rv, sv, cv, tv in zip(r, s, c, t):
        step_state = barometer_step(
            root_stability=float(rv),
            somatic_anchor=float(sv),
            coherence_level=float(cv),
            trust=float(tv),
            prev_state=prev,
            config=cfg,
            dt=dt,
        )
        out.append(step_state)
        prev = step_state

    return out
