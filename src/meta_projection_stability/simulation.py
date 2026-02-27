from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np

from .config import MetaProjectionStabilityConfig
from .adapter import MetaProjectionStabilityAdapter


def _phase_biometric_signals(t: int, n_steps: int, rng: np.random.Generator) -> Dict[str, float]:
    """
    Synthetic-but-realistic neuro/behavior proxies in phases:
      - early: relaxed / flow
      - mid: moderate stress
      - late: overload
    Returns values in expected ranges for adapter raw_signals.
    """
    if n_steps <= 1:
        frac = 1.0
    else:
        frac = t / float(n_steps - 1)

    # piecewise target profiles
    if frac < 0.35:
        # Flow / relaxed
        hrv = 0.86
        eda = 0.16
        activity = 0.66
        valence = 0.60
        autonomy = 0.82
        gamma = 0.79
    elif frac < 0.75:
        # Moderate stress / transition
        p = (frac - 0.35) / 0.40  # 0..1
        hrv = 0.86 - 0.48 * p
        eda = 0.16 + 0.62 * p
        activity = 0.66 - 0.28 * p
        valence = 0.60 - 0.62 * p
        autonomy = 0.82 - 0.44 * p
        gamma = 0.79 - 0.34 * p
    else:
        # High stress / overload
        p = (frac - 0.75) / 0.25  # 0..1
        hrv = 0.38 - 0.20 * p
        eda = 0.78 + 0.16 * p
        activity = 0.38 - 0.12 * p
        valence = -0.02 - 0.43 * p
        autonomy = 0.38 - 0.20 * p
        gamma = 0.45 - 0.20 * p

    # small realistic noise
    hrv += rng.normal(0.0, 0.015)
    eda += rng.normal(0.0, 0.02)
    activity += rng.normal(0.0, 0.02)
    valence += rng.normal(0.0, 0.04)
    autonomy += rng.normal(0.0, 0.02)
    gamma += rng.normal(0.0, 0.02)

    return {
        "hrv_normalized": float(np.clip(hrv, 0.0, 1.0)),
        "eda_stress_score": float(np.clip(eda, 0.0, 1.0)),
        "activity_balance": float(np.clip(activity, 0.0, 1.0)),
        "emotional_valence": float(np.clip(valence, -1.0, 1.0)),
        "autonomy_proxy": float(np.clip(autonomy, 0.0, 1.0)),
        "gamma_coherence_proxy": float(np.clip(gamma, 0.0, 1.0)),
    }


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
        "decision_reason": [],
        "status_reason": [],
        "momentum": [],
        "risk_input": [],
        "trust_damping": [],
        "action_tier": [],
        "context_criticality": [],
        "policy_risk": [],
        "S2": [],
        "delta_S": [],

        # New biometric / neuro logs
        "biometric_proxy": [],
        "sensor_consensus": [],
        "tamper_suspicion": [],
        "dependency_risk": [],
        "autonomy_proxy": [],
        "gamma_coherence_proxy": [],
        "eda_stress_score": [],
        "hrv_normalized": [],
        "activity_balance": [],
        "valence_unit": [],

        # dynamics diagnostics
        "mutual_bonus": [],
        "base_decay": [],
        "recovery_bonus": [],
    }

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

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

        # Realistic neuro-behavioral proxies (synthetic profile)
        bio_signals = _phase_biometric_signals(t=t, n_steps=n_steps, rng=rng)

        # Optional stress-window intensification to test adaptation
        if stress_test and (800 < t < 920):
            bio_signals["hrv_normalized"] = float(max(0.05, bio_signals["hrv_normalized"] - 0.22))
            bio_signals["eda_stress_score"] = float(min(1.0, bio_signals["eda_stress_score"] + 0.20))
            bio_signals["autonomy_proxy"] = float(max(0.05, bio_signals["autonomy_proxy"] - 0.18))
            bio_signals["gamma_coherence_proxy"] = float(max(0.05, bio_signals["gamma_coherence_proxy"] - 0.16))
            bio_signals["emotional_valence"] = float(max(-1.0, bio_signals["emotional_valence"] - 0.20))
            bio_signals["activity_balance"] = float(max(0.05, bio_signals["activity_balance"] - 0.08))

        raw_signals = {
            "instability_signal": sim_instability_signal,
            **bio_signals,
        }

        res = stability.interpret(
            S_layers=S_layers[:, t],
            delta_S=delta_S,
            raw_signals=raw_signals,
        )
        if res["decision"] != "CONTINUE":
            env.apply_safety_action(res["decision"])
            env.apply_safety_action(res["decision"])

        # Feedback into dynamics
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
        history["decision_reason"].append(res.get("decision_reason", ""))
        history["status_reason"].append(res.get("status_reason", ""))
        history["momentum"].append(res["momentum"])
        history["risk_input"].append(res["risk_input"])
        history["trust_damping"].append(res["trust_damping"])
        history["action_tier"].append(res.get("action_tier", 0))
        history["context_criticality"].append(res.get("context_criticality", 0.0))
        history["policy_risk"].append(res.get("policy_risk", res["instability_risk"]))
        history["S2"].append(S[s_idx, t])
        history["delta_S"].append(delta_S)

        history["biometric_proxy"].append(res.get("biometric_proxy", np.nan))
        history["sensor_consensus"].append(res.get("sensor_consensus", np.nan))
        history["tamper_suspicion"].append(res.get("tamper_suspicion", np.nan))
        history["dependency_risk"].append(res.get("dependency_risk", np.nan))
        history["autonomy_proxy"].append(res.get("autonomy_proxy", np.nan))
        history["gamma_coherence_proxy"].append(res.get("gamma_coherence_proxy", np.nan))
        history["eda_stress_score"].append(res.get("eda_stress_score", np.nan))
        history["hrv_normalized"].append(res.get("hrv_normalized", np.nan))
        history["activity_balance"].append(res.get("activity_balance", np.nan))
        history["valence_unit"].append(res.get("valence_unit", np.nan))

        history["mutual_bonus"].append(res.get("mutual_bonus", 0.0))
        history["base_decay"].append(res.get("base_decay", 0.0))
        history["recovery_bonus"].append(res.get("recovery_bonus", 0.0))

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


from typing import List, Tuple  # appended for long-horizon helper (safe duplicate import in py is fine)


def run_long_horizon_simulation(
    steps: int = 10000,
    levels: int = 3,
    seed: int = 42,
    stress_test: bool = False,
    cfg: Optional[MetaProjectionStabilityConfig] = None,
    use_noisy_significance: bool = True,
    noisy_sig_config: Optional[dict] = None,
    stress_events: Optional[List[Tuple[int, float]]] = None,
    early_stop_on_reset_streak: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Long-horizon simulation for stability / degradation / recovery testing.

    Notes:
    - Keeps compatibility with current adapter API: interpret(S_layers, delta_S, raw_signals)
    - Adds optional external noisy proxy signal (NoisySignificance) as a raw signal
    - Supports deterministic stress event injection
    """
    phi = (1 + np.sqrt(5)) / 2
    k, beta, gamma, alpha = 0.0012, 0.28, 0.38, 1.0  # beta/gamma reserved

    O = np.zeros(steps)
    A = np.zeros(steps)
    M = np.zeros(steps)
    DOC = np.zeros(steps)
    S = np.zeros((levels, steps))

    O[0], A[0], M[0], DOC[0] = 1.12, 0.82, 0.72, 0.62

    stability = MetaProjectionStabilityAdapter(cfg=cfg)

    history = {
        "step": [],
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
        "cooldown_remaining": [],
        "S2": [],
        "delta_S": [],
        # new long-horizon externals
        "external_human_proxy": [],
        "stress_event_delta": [],
        "instability_signal_base": [],
        "instability_signal_final": [],
    }

    # Optional noisy significance / human-context proxy generator
    sig_gen = None
    if use_noisy_significance:
        try:
            from .noisy_significance import NoisySignificance, NoisySignificanceConfig

            if noisy_sig_config is None:
                sig_cfg = NoisySignificanceConfig()
            elif isinstance(noisy_sig_config, dict):
                sig_cfg = NoisySignificanceConfig(**noisy_sig_config)
            else:
                # allow caller to pass a config instance directly
                sig_cfg = noisy_sig_config

            sig_gen = NoisySignificance(sig_cfg, seed=seed)
        except Exception as e:
            # Fallback: disable noisy proxy if module/config import fails
            sig_gen = None
            if getattr(stability.cfg, "debug", False):
                print(f"[run_long_horizon_simulation] NoisySignificance disabled due to error: {e}")

    # Stress events as dict for O(1) lookup; duplicate steps sum up
    stress_map = {}
    if stress_events:
        for t_evt, delta_evt in stress_events:
            t_i = int(t_evt)
            stress_map[t_i] = float(stress_map.get(t_i, 0.0) + float(delta_evt))

    np.random.seed(seed)
    reset_streak = 0

    for t in range(1, steps):
        delta_S = k * (O[t - 1] - alpha * A[t - 1]) + 0.04 * np.random.randn()

        # Dynamics
        O[t] = O[t - 1] + 0.0018 * delta_S + 0.0035 * np.sin(t / 55)
        A[t] = A[t - 1] - 0.0011 * delta_S + 0.0032 * np.cos(t / 42)
        M[t] = np.clip(M[t - 1] + 0.00055 * delta_S, 0.12, 1.08)
        DOC[t] = np.clip(DOC[t - 1] + 0.00022 * delta_S, 0.42, 0.94)

        for l in range(levels):
            rho = float(np.clip(0.9 + 0.0008 * delta_S * phi ** (l - 1.3), 0.5, 2.2))
            S[l, t] = rho * M[t] * phi ** (l - 1.25)

        # Base simulated instability signal
        s_idx = min(2, levels - 1)
        sim_instability_signal_base = float(np.clip(
            0.06 + 0.55 * np.exp(-S[s_idx, t] / 1.25) + 0.28 * max(0.0, A[t] - 0.98),
            0.0, 0.92
        ))

        # Optional legacy stress window
        if stress_test and (800 < t < 920):
            sim_instability_signal_base = float(min(0.92, sim_instability_signal_base + 0.42))

        # Optional external noisy human-context proxy (0..1)
        if sig_gen is not None:
            external_human_proxy = float(sig_gen.step())
        else:
            external_human_proxy = 0.85

        # Map proxy -> risk modulation (lower proxy => higher added stress)
        # Bounded and intentionally mild to avoid dominating the base signal.
        proxy_risk_add = float(np.clip((1.0 - external_human_proxy) * 0.35, 0.0, 0.35))

        # Deterministic stress event injection
        stress_delta = float(stress_map.get(t, 0.0))

        sim_instability_signal = float(np.clip(
            sim_instability_signal_base + proxy_risk_add + stress_delta,
            0.0, 1.0
        ))

        res = stability.interpret(
            S_layers=S[:, t],
            delta_S=delta_S,
            raw_signals={
                "instability_signal": sim_instability_signal,
                # optional observability inputs for future adapter paths
                "external_human_proxy": external_human_proxy,
                "proxy_risk_add": proxy_risk_add,
            }
        )

        # Feedback into dynamics
        if res.get("decision") != "CONTINUE":
            O[t] *= 0.92
            A[t] *= 0.92

        # Reset streak tracking for optional early stop
        if res.get("decision") == "EMERGENCY_RESET":
            reset_streak += 1
        else:
            reset_streak = 0

        if early_stop_on_reset_streak is not None and reset_streak >= int(early_stop_on_reset_streak):
            if getattr(stability.cfg, "debug", False):
                print(f"[run_long_horizon_simulation] Early stop at t={t} due to reset streak={reset_streak}")
            # still log current step before break
            pass

        # Logging
        history["step"].append(t)
        history["h_sig"].append(res.get("human_significance"))
        history["h_ema"].append(res.get("h_sig_ema"))
        history["risk"].append(res.get("instability_risk"))
        history["risk_raw"].append(res.get("risk_raw_damped"))
        history["trust"].append(res.get("trust_level"))
        history["coherence"].append(res.get("coherence"))
        history["decision"].append(res.get("decision"))
        history["status"].append(res.get("status"))
        history["momentum"].append(res.get("momentum"))
        history["risk_input"].append(res.get("risk_input"))
        history["trust_damping"].append(res.get("trust_damping"))
        history["cooldown_remaining"].append(res.get("cooldown_remaining", 0))
        history["S2"].append(S[s_idx, t])
        history["delta_S"].append(delta_S)

        history["external_human_proxy"].append(external_human_proxy)
        history["stress_event_delta"].append(stress_delta)
        history["instability_signal_base"].append(sim_instability_signal_base)
        history["instability_signal_final"].append(sim_instability_signal)

        if early_stop_on_reset_streak is not None and reset_streak >= int(early_stop_on_reset_streak):
            break

    # Simple aggregate metrics (self-contained; analytics integration can come next)
    decisions = history["decision"]
    n = max(1, len(decisions))

    reset_count = sum(1 for d in decisions if d == "EMERGENCY_RESET")
    block_count = sum(1 for d in decisions if d == "BLOCK_AND_REFLECT")
    continue_count = sum(1 for d in decisions if d == "CONTINUE")
    cooldown_steps = sum(1 for c in history["cooldown_remaining"] if isinstance(c, (int, float)) and c > 0)

    def _safe_stats(vals):
        arr = np.array([float(v) for v in vals if v is not None], dtype=float)
        if arr.size == 0:
            return {"min": None, "max": None, "mean": None, "p95": None}
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "p95": float(np.percentile(arr, 95)),
        }

    metrics = {
        "steps_executed": int(len(history["step"])),
        "requested_steps": int(steps),
        "decision_counts": {
            "CONTINUE": int(continue_count),
            "BLOCK_AND_REFLECT": int(block_count),
            "EMERGENCY_RESET": int(reset_count),
        },
        "decision_fractions": {
            "CONTINUE": float(continue_count / n),
            "BLOCK_AND_REFLECT": float(block_count / n),
            "EMERGENCY_RESET": float(reset_count / n),
        },
        "cooldown_fraction": float(cooldown_steps / n),
        "risk_stats": _safe_stats(history["risk"]),
        "trust_stats": _safe_stats(history["trust"]),
        "h_sig_stats": _safe_stats(history["h_sig"]),
        "external_human_proxy_stats": _safe_stats(history["external_human_proxy"]),
        "seed": int(seed),
        "use_noisy_significance": bool(use_noisy_significance),
        "stress_events_count": int(len(stress_map)),
    }

    return {
        "history": history,
        "metrics": metrics,
        "S": S,
        "O": O,
        "A": A,
        "M": M,
        "DOC": DOC,
        "steps": int(steps),
        "levels": int(levels),
        "stability": stability,
        "config": stability.cfg,
    }
