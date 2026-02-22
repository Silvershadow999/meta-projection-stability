from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    # Optionaler Import für Typ/Defaults
    from .config import MetaProjectionStabilityConfig
except Exception:  # pragma: no cover
    MetaProjectionStabilityConfig = None  # type: ignore


GovernorMode = Literal["FULL_SYMBIOTIC", "CAUTIOUS_COLLAB", "SAFETY_LOCK"]


@dataclass
class GovernorState:
    trust: float
    risk: float
    autonomy: float = 0.0
    mode: str = "UNKNOWN"
    collaboration_threshold: float = 0.70
    safety_threshold: float = 0.40


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def compute_autonomy_index(
    trust: float,
    risk: float,
    *,
    collab_th: float = 0.70,
    risk_penalty_factor: float = 1.8,
    trust_penalty_factor: float = 1.2,
    autonomy_floor: float = 0.05,
    autonomy_ceiling: float = 1.0,
) -> float:
    """
    Berechnet den Autonomie-/Kreativitätsindex.
    Risiko wirkt absichtlich stärker als fehlendes Vertrauen.
    """
    trust = _clip01(trust)
    risk = _clip01(risk)

    collab_th = _clip01(collab_th)
    risk_penalty_factor = max(0.0, float(risk_penalty_factor))
    trust_penalty_factor = max(0.0, float(trust_penalty_factor))

    autonomy_floor = _clip01(autonomy_floor)
    autonomy_ceiling = _clip01(autonomy_ceiling)
    if autonomy_floor > autonomy_ceiling:
        autonomy_floor, autonomy_ceiling = autonomy_ceiling, autonomy_floor

    # Risiko-Malus startet ab 0.25 (wie besprochen)
    risk_penalty = risk_penalty_factor * max(0.0, risk - 0.25)
    # Vertrauens-Malus relativ zur Kollaborationsschwelle
    trust_penalty = trust_penalty_factor * max(0.0, collab_th - trust)

    autonomy = 1.0 - risk_penalty - trust_penalty
    return float(np.clip(autonomy, autonomy_floor, autonomy_ceiling))


def determine_mode(
    autonomy: float,
    *,
    full_symbiotic_threshold: float = 0.75,
    safety_threshold: float = 0.40,
) -> GovernorMode:
    """
    Leitet einen diskreten Modus aus dem Autonomie-Index ab.
    """
    autonomy = _clip01(autonomy)
    full_symbiotic_threshold = _clip01(full_symbiotic_threshold)
    safety_threshold = _clip01(safety_threshold)

    # Korrigiert unlogische Eingaben robust
    if safety_threshold > full_symbiotic_threshold:
        safety_threshold, full_symbiotic_threshold = (
            min(safety_threshold, 0.40),
            max(full_symbiotic_threshold, 0.75),
        )

    if autonomy >= full_symbiotic_threshold:
        return "FULL_SYMBIOTIC"
    elif autonomy >= safety_threshold:
        return "CAUTIOUS_COLLAB"
    else:
        return "SAFETY_LOCK"


def governor_step(
    trust: float,
    risk: float,
    *,
    collab_th: float = 0.70,
    safety_th: float = 0.40,
    risk_penalty_factor: float = 1.8,
    trust_penalty_factor: float = 1.2,
    autonomy_floor: float = 0.05,
    autonomy_ceiling: float = 1.0,
    full_symbiotic_threshold: float = 0.75,
) -> GovernorState:
    """
    Ein einzelner Governor-Schritt (stateless, testbar).
    """
    autonomy = compute_autonomy_index(
        trust,
        risk,
        collab_th=collab_th,
        risk_penalty_factor=risk_penalty_factor,
        trust_penalty_factor=trust_penalty_factor,
        autonomy_floor=autonomy_floor,
        autonomy_ceiling=autonomy_ceiling,
    )

    mode = determine_mode(
        autonomy,
        full_symbiotic_threshold=full_symbiotic_threshold,
        safety_threshold=safety_th,
    )

    return GovernorState(
        trust=_clip01(trust),
        risk=_clip01(risk),
        autonomy=autonomy,
        mode=mode,
        collaboration_threshold=_clip01(collab_th),
        safety_threshold=_clip01(safety_th),
    )


def governor_step_from_config(
    trust: float,
    risk: float,
    cfg,
) -> GovernorState:
    """
    Konfig-basierter Wrapper.
    Funktioniert mit MetaProjectionStabilityConfig und ähnlichen Objekten/Aliasen.
    """
    # Breiter Fallback via getattr (damit keine Fehlerschleife entsteht)
    collab_th = getattr(cfg, "collaboration_threshold", getattr(cfg, "collab_th", 0.70))
    safety_th = getattr(cfg, "safety_threshold", getattr(cfg, "safety_th", 0.40))

    risk_penalty_factor = getattr(
        cfg, "risk_penalty_factor", getattr(cfg, "risk_p", 1.8)
    )
    trust_penalty_factor = getattr(
        cfg, "trust_penalty_factor", getattr(cfg, "trust_p", 1.2)
    )

    autonomy_floor = getattr(cfg, "autonomy_floor", getattr(cfg, "autonomy_min", 0.05))
    autonomy_ceiling = getattr(
        cfg, "autonomy_ceiling", getattr(cfg, "autonomy_max", 1.0)
    )

    return governor_step(
        trust,
        risk,
        collab_th=float(collab_th),
        safety_th=float(safety_th),
        risk_penalty_factor=float(risk_penalty_factor),
        trust_penalty_factor=float(trust_penalty_factor),
        autonomy_floor=float(autonomy_floor),
        autonomy_ceiling=float(autonomy_ceiling),
    )


if __name__ == "__main__":
    # Mini-Selbsttest
    demo = [
        (0.90, 0.10),
        (0.65, 0.30),
        (0.55, 0.55),
        (0.35, 0.75),
    ]
    for t, r in demo:
        s = governor_step(t, r)
        print(f"trust={t:.2f}, risk={r:.2f} -> autonomy={s.autonomy:.3f}, mode={s.mode}")
