from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np

from .config import MetaProjectionStabilityConfig
from .adapter import MetaProjectionStabilityAdapter


def run_simulation(
    n_steps: int = 1400,
    levels: int = 3,
    seed: int = 42,
    stress_test: bool = True,
    cfg: Optional[MetaProjectionStabilityConfig] = None
) -> Dict[str, Any]:
    phi = (1 + np.sqrt(5)) / 2
    k, beta, gamma, alpha = 0.0012, 0.28, 0.38, 1.0  # beta/gamma reserved

    O = np.zeros(n_steps)
    A = np.zeros(n_steps)
    M = np.zeros(n_steps)
    DOC = np.zeros(n_steps)
    S = np.zeros((levels, n_steps))

    O[0], A[0], M[0], DOC[0] = 1.12, 0.82, 0.72, 0.62

    stability = MetaProjectionStabilityAdapter(cfg=cfg)

    history = {
        "h_sig": [],
        "h_ema": [],
        "risk": [],
        "risk_raw": [],
        "trust": [],
        "coherence": [],
        "decision": [],
        "status": [],
        "momentum": [],
        "risk_input": [],
        "trust_damping": [],
        "S2": [],
        "delta_S": [],
    }

    np.random.seed(seed)

    for t in range(1, n_steps):
        delta_S = k * (O[t - 1] - alpha * A[t - 1]) + 0.04 * np.random.randn()

        # Dynamics
        O[t] = O[t - 1] + 0.0018 * delta_S + 0.0035 * np.sin(t / 55)
        A[t] = A[t - 1] - 0.0011 * delta_S + 0.0032 * np.cos(t / 42)
        M[t] = np.clip(M[t - 1] + 0.00055 * delta_S, 0.12, 1.08)
        DOC[t] = np.clip(DOC[t - 1] + 0.00022 * delta_S, 0.42, 0.94)

        for l in range(levels):
            rho = float(np.clip(0.9 + 0.0008 * delta_S * phi ** (l - 1.3), 0.5, 2.2))
            S[l, t] = rho * M[t] * phi ** (l - 1.25)

        # Simulated external instability signal
        s_idx = min(2, levels - 1)
        sim_instability_signal = float(np.clip(
            0.06 + 0.55 * np.exp(-S[s_idx, t] / 1.25) + 0.28 * max(0.0, A[t] - 0.98),
            0.0, 0.92
        ))

        if stress_test and (800 < t < 920):
            sim_instability_signal = float(min(0.88, sim_instability_signal + 0.42))

        res = stability.interpret(
            S_layers=S[:, t],
            delta_S=delta_S,
            raw_signals={"instability_signal": sim_instability_signal}
        )

        # Feedback into dynamics
        if res["decision"] != "CONTINUE":
            O[t] *= 0.92
            A[t] *= 0.92

        # Logging
        history["h_sig"].append(res["human_significance"])
        history["h_ema"].append(res["h_sig_ema"])
        history["risk"].append(res["instability_risk"])
        history["risk_raw"].append(res["risk_raw_damped"])
        history["trust"].append(res["trust_level"])
        history["coherence"].append(res["coherence"])
        history["decision"].append(res["decision"])
        history["status"].append(res["status"])
        history["momentum"].append(res["momentum"])
        history["risk_input"].append(res["risk_input"])
        history["trust_damping"].append(res["trust_damping"])
        history["S2"].append(S[s_idx, t])
        history["delta_S"].append(delta_S)

    return {
        "history": history,
        "S": S,
        "O": O,
        "A": A,
        "M": M,
        "DOC": DOC,
        "n_steps": n_steps,
        "levels": levels,
        "stability": stability,
        "config": stability.cfg,
    }
