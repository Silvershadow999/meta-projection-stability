from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class GovernorState:
    """
    Zustand des Governors für einen Simulationsschritt.
    """
    trust: float
    risk: float
    autonomy: float = 0.0
    mode: str = "UNKNOWN"

    # Schwellen (für Plot/Debug sichtbar im State)
    collaboration_threshold: float = 0.70
    safety_threshold: float = 0.40

    # Penalty-Faktoren (ebenfalls sichtbar, hilfreich für Trinity-/Preset-Vergleiche)
    risk_penalty_factor: float = 1.8
    trust_penalty_factor: float = 1.2


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def compute_autonomy_index(
    trust: float,
    risk: float,
    collab_th: float = 0.70,
    risk_penalty_factor: float = 1.8,
    trust_penalty_factor: float = 1.2,
) -> float:
    """
    Berechnet den Autonomie-/Kreativitäts-Spielraum der KI (0..1).

    Design:
    - Risiko wird asymmetrisch stärker bestraft als Vertrauensverlust
    - Inputs werden defensiv auf [0..1] geclippt
    - Risk-Penalty greift ab moderatem Risiko (> 0.25)
    - Trust-Penalty greift, wenn Trust unter die Kollaborations-Schwelle fällt
    """
    trust = _clip01(trust)
    risk = _clip01(risk)
    collab_th = _clip01(collab_th)

    risk_penalty_factor = float(risk_penalty_factor)
    trust_penalty_factor = float(trust_penalty_factor)

    # Asymmetrische Malus-Logik
    risk_penalty = risk_penalty_factor * max(0.0, risk - 0.25)
    trust_penalty = trust_penalty_factor * max(0.0, collab_th - trust)

    autonomy = 1.0 - risk_penalty - trust_penalty
    return float(np.clip(autonomy, 0.0, 1.0))


def determine_mode(
    autonomy: float,
    collab_th: float = 0.70,
    safety_th: float = 0.40,
) -> str:
    """
    Diskreter Verhaltensmodus auf Basis des Autonomy-Index.
    """
    autonomy = _clip01(autonomy)
    collab_th = _clip01(collab_th)
    safety_th = _clip01(safety_th)

    # defensive ordering: safety <= collab
    if safety_th > collab_th:
        safety_th, collab_th = collab_th, safety_th

    if autonomy >= collab_th:
        return "FULL_SYMBIOTIC"
    elif autonomy >= safety_th:
        return "CAUTIOUS_COLLAB"
    else:
        return "SAFETY_LOCK"


def governor_step(
    trust: float,
    risk: float,
    collab_th: float = 0.70,
    safety_th: float = 0.40,
    risk_penalty_factor: float = 1.8,
    trust_penalty_factor: float = 1.2,
) -> GovernorState:
    """
    Ein Simulationsschritt des Governors: Inputs -> Autonomy -> Mode.
    """
    autonomy = compute_autonomy_index(
        trust=trust,
        risk=risk,
        collab_th=collab_th,
        risk_penalty_factor=risk_penalty_factor,
        trust_penalty_factor=trust_penalty_factor,
    )
    mode = determine_mode(
        autonomy=autonomy,
        collab_th=collab_th,
        safety_th=safety_th,
    )

    return GovernorState(
        trust=_clip01(trust),
        risk=_clip01(risk),
        autonomy=autonomy,
        mode=mode,
        collaboration_threshold=_clip01(collab_th),
        safety_threshold=_clip01(safety_th),
        risk_penalty_factor=float(risk_penalty_factor),
        trust_penalty_factor=float(trust_penalty_factor),
    )


def summarize_governor_history(history: list[GovernorState]) -> Dict[str, float]:
    """
    Kompakte Auswertung für CLI / Debugging:
    - Anzahl Moduswechsel
    - Anteil Zeit pro Modus
    - Erste SAFETY_LOCK-Stelle
    - mittlere Autonomie
    """
    n = len(history)
    if n == 0:
        return {
            "n_steps": 0,
            "mode_switches": 0,
            "full_symbiotic_ratio": 0.0,
            "cautious_collab_ratio": 0.0,
            "safety_lock_ratio": 0.0,
            "first_safety_lock_idx": -1,
            "mean_autonomy": 0.0,
        }

    modes = [h.mode for h in history]
    autonomies = [float(h.autonomy) for h in history]

    mode_switches = 0
    for i in range(1, n):
        if modes[i] != modes[i - 1]:
            mode_switches += 1

    full_count = sum(1 for m in modes if m == "FULL_SYMBIOTIC")
    caut_count = sum(1 for m in modes if m == "CAUTIOUS_COLLAB")
    safe_count = sum(1 for m in modes if m == "SAFETY_LOCK")

    first_safety = next((i for i, m in enumerate(modes) if m == "SAFETY_LOCK"), -1)
    mean_autonomy = float(np.mean(autonomies)) if autonomies else 0.0

    return {
        "n_steps": n,
        "mode_switches": mode_switches,
        "full_symbiotic_ratio": full_count / n,
        "cautious_collab_ratio": caut_count / n,
        "safety_lock_ratio": safe_count / n,
        "first_safety_lock_idx": first_safety,
        "mean_autonomy": mean_autonomy,
    }


# ---------------------------------------------------------------------
# Trinity profile presets (für Vergleichsplots / Demos)
# ---------------------------------------------------------------------
TRINITY_PROFILES = {
    # etwas optimistischer / weniger risk-avers
    "Gemini": {"risk_p": 1.5, "trust_p": 1.1},
    # strengerer Sicherheitsfilter
    "Grok": {"risk_p": 1.9, "trust_p": 1.3},
    # ausbalanciert
    "ChatGPT": {"risk_p": 1.7, "trust_p": 1.2},
}


def compute_trinity_autonomy_histories(
    trust_hist,
    risk_hist,
    collab_th: float = 0.70,
    profiles: dict | None = None,
) -> dict[str, np.ndarray]:
    """
    Hilfsfunktion für Vergleiche:
    Berechnet für mehrere Profile die Autonomie-Historien auf denselben Inputs.
    """
    trust_arr = np.asarray(trust_hist, dtype=float)
    risk_arr = np.asarray(risk_hist, dtype=float)

    if trust_arr.shape != risk_arr.shape:
        raise ValueError("trust_hist and risk_hist must have the same shape")

    use_profiles = profiles or TRINITY_PROFILES
    out: dict[str, np.ndarray] = {}

    for name, p in use_profiles.items():
        risk_p = float(p.get("risk_p", 1.8))
        trust_p = float(p.get("trust_p", 1.2))
        autonomy = np.array(
            [
                compute_autonomy_index(
                    trust=t,
                    risk=r,
                    collab_th=collab_th,
                    risk_penalty_factor=risk_p,
                    trust_penalty_factor=trust_p,
                )
                for t, r in zip(trust_arr, risk_arr)
            ],
            dtype=float,
        )
        out[str(name)] = autonomy

    return out
