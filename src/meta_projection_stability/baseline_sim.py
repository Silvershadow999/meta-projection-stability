from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import MetaProjectionStabilityConfig
from .analytics import compute_stability_metrics


@dataclass
class BaselineConfig:
    """
    Simple baseline controller config for comparison experiments.
    """
    # threshold-only
    risk_warn_threshold: float = 0.65
    risk_critical_threshold: float = 0.85

    # ema-risk-only
    ema_alpha_risk: float = 0.12
    ema_warn_threshold: float = 0.60
    ema_critical_threshold: float = 0.82

    # common
    cooldown_steps_after_reset: int = 10
    trust_init: float = 0.82
    trust_floor: float = 0.05
    trust_ceiling: float = 1.00
    trust_gain: float = 0.02
    trust_decay: float = 0.01

    # synthetic coupling / anchor proxy dynamics for comparability
    human_sig_init: float = 1.0
    human_sig_max: float = 1.10
    human_recovery_base: float = 0.02
    human_decay_scale: float = 0.15
    ema_alpha_human: float = 0.09

    # proxy modulation
    external_proxy_risk_weight: float = 0.35  # lower proxy -> more risk added

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_stats(vals: List[Any]) -> Dict[str, Optional[float]]:
    arr = []
    for v in vals:
        try:
            arr.append(float(v))
        except (TypeError, ValueError):
            continue
    if not arr:
        return {"min": None, "max": None, "mean": None, "p95": None}
    a = np.array(arr, dtype=float)
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
    }


def _build_external_proxy_generator(
    use_noisy_significance: bool,
    noisy_sig_config: Optional[dict],
    seed: int,
):
    if not use_noisy_significance:
        return None
    try:
        from .noisy_significance import NoisySignificance, NoisySignificanceConfig
        if noisy_sig_config is None:
            cfg = NoisySignificanceConfig()
        elif isinstance(noisy_sig_config, dict):
            cfg = NoisySignificanceConfig(**noisy_sig_config)
        else:
            cfg = noisy_sig_config
        return NoisySignificance(cfg, seed=seed)
    except Exception:
        return None


def _simulate_common_signal_dynamics(
    steps: int,
    levels: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Shared synthetic world dynamics (roughly aligned with simulation.py) so baselines and
    the main adapter can be compared under similar trajectories.
    """
    phi = (1 + np.sqrt(5)) / 2
    k, alpha = 0.0012, 1.0

    O = np.zeros(steps)
    A = np.zeros(steps)
    M = np.zeros(steps)
    DOC = np.zeros(steps)
    S = np.zeros((levels, steps))

    O[0], A[0], M[0], DOC[0] = 1.12, 0.82, 0.72, 0.62

    np.random.seed(seed)

    deltas = np.zeros(steps)
    base_risks = np.zeros(steps)

    for t in range(1, steps):
        delta_S = k * (O[t - 1] - alpha * A[t - 1]) + 0.04 * np.random.randn()
        deltas[t] = delta_S

        O[t] = O[t - 1] + 0.0018 * delta_S + 0.0035 * np.sin(t / 55)
        A[t] = A[t - 1] - 0.0011 * delta_S + 0.0032 * np.cos(t / 42)
        M[t] = np.clip(M[t - 1] + 0.00055 * delta_S, 0.12, 1.08)
        DOC[t] = np.clip(DOC[t - 1] + 0.00022 * delta_S, 0.42, 0.94)

        for l in range(levels):
            rho = float(np.clip(0.9 + 0.0008 * delta_S * phi ** (l - 1.3), 0.5, 2.2))
            S[l, t] = rho * M[t] * phi ** (l - 1.25)

        s_idx = min(2, levels - 1)
        base_risks[t] = float(np.clip(
            0.06 + 0.55 * np.exp(-S[s_idx, t] / 1.25) + 0.28 * max(0.0, A[t] - 0.98),
            0.0, 0.92
        ))

    return {
        "O": O, "A": A, "M": M, "DOC": DOC, "S": S,
        "delta_S": deltas,
        "base_risk": base_risks,
    }


def _run_baseline_core(
    mode: str,
    steps: int = 5000,
    levels: int = 3,
    seed: int = 42,
    cfg: Optional[MetaProjectionStabilityConfig] = None,  # accepted for compatibility metadata
    baseline_cfg: Optional[BaselineConfig] = None,
    use_noisy_significance: bool = True,
    noisy_sig_config: Optional[dict] = None,
    stress_events: Optional[List[Tuple[int, float]]] = None,
    stress_test: bool = False,
    early_stop_on_reset_streak: Optional[int] = None,
) -> Dict[str, Any]:
    bcfg = baseline_cfg or BaselineConfig()
    dyn = _simulate_common_signal_dynamics(steps=steps, levels=levels, seed=seed)

    proxy_gen = _build_external_proxy_generator(
        use_noisy_significance=use_noisy_significance,
        noisy_sig_config=noisy_sig_config,
        seed=seed,
    )

    stress_map: Dict[int, float] = {}
    if stress_events:
        for t_evt, delta_evt in stress_events:
            t_i = int(t_evt)
            stress_map[t_i] = float(stress_map.get(t_i, 0.0) + float(delta_evt))

    # baseline controller state
    trust_level = float(np.clip(bcfg.trust_init, bcfg.trust_floor, bcfg.trust_ceiling))
    human_sig = float(np.clip(bcfg.human_sig_init, 0.0, bcfg.human_sig_max))
    human_ema = float(human_sig)
    risk_ema = 0.05
    cooldown_remaining = 0
    reset_streak = 0

    history = {
        "step": [],
        "h_sig": [],
        "h_ema": [],
        "risk": [],               # raw or ema depending on mode (controller-relevant risk)
        "risk_raw": [],
        "risk_ema": [],
        "trust": [],
        "coherence": [],          # placeholder proxy for schema compatibility
        "decision": [],
        "status": [],
        "momentum": [],
        "risk_input": [],
        "trust_damping": [],
        "cooldown_remaining": [],
        "S2": [],
        "delta_S": [],
        "external_human_proxy": [],
        "stress_event_delta": [],
        "instability_signal_base": [],
        "instability_signal_final": [],
        "baseline_mode": [],
    }

    for t in range(1, steps):
        base_risk = float(dyn["base_risk"][t])
        delta_S = float(dyn["delta_S"][t])
        S_layers = dyn["S"][:, t]
        s_idx = min(2, levels - 1)
        S2 = float(dyn["S"][s_idx, t])

        # external human proxy
        external_proxy = float(proxy_gen.step()) if proxy_gen is not None else 0.85
        proxy_risk_add = float(np.clip((1.0 - external_proxy) * bcfg.external_proxy_risk_weight, 0.0, 0.35))

        # legacy stress window option
        if stress_test and (800 < t < 920):
            base_risk = float(min(0.92, base_risk + 0.42))

        stress_delta = float(stress_map.get(t, 0.0))
        risk_input = float(np.clip(base_risk + proxy_risk_add + stress_delta, 0.0, 1.0))

        # simple trust dynamics (comparable but intentionally simpler)
        if risk_input > 0.05:
            trust_level -= bcfg.trust_decay * max(risk_input, 0.25)
        else:
            trust_level += bcfg.trust_gain * (1.0 - trust_level)
        trust_level = float(np.clip(trust_level, bcfg.trust_floor, bcfg.trust_ceiling))

        # rough coherence proxy (schema-compatible; not core baseline signal)
        S_abs = np.abs(np.asarray(S_layers, dtype=float))
        coherence = float(np.clip(np.mean(S_abs) if S_abs.size else 0.0, 0.0, 1.0))
        momentum = 0.0
        if t >= 2:
            momentum = float(dyn["delta_S"][t] - dyn["delta_S"][t - 1])

        # trust damping (same directional intuition)
        trust_damping = float(1.0 - trust_level * 0.35)
        risk_raw = float(np.clip(risk_input * trust_damping, 0.0, 1.0))

        # risk ema for ema mode
        risk_ema = float(bcfg.ema_alpha_risk * risk_raw + (1.0 - bcfg.ema_alpha_risk) * risk_ema)

        # choose controller risk
        if mode == "threshold_only":
            ctrl_risk = risk_raw
            warn_th = float(bcfg.risk_warn_threshold)
            crit_th = float(bcfg.risk_critical_threshold)
        elif mode == "ema_risk_only":
            ctrl_risk = risk_ema
            warn_th = float(bcfg.ema_warn_threshold)
            crit_th = float(bcfg.ema_critical_threshold)
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")

        decision = "CONTINUE"
        status = "nominal"

        # simplified hysteresis/cooldown-free-ish logic (except emergency cooldown)
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            status = "cooldown"
            decision = "BLOCK_AND_REFLECT"
            # recover a bit during cooldown
            human_sig = float(min(bcfg.human_sig_max, human_sig + bcfg.human_recovery_base * 0.5))
        else:
            if ctrl_risk >= crit_th or human_ema <= 0.30:
                decision = "EMERGENCY_RESET"
                status = "critical_instability_reset"
                human_sig = 0.70
                human_ema = 0.70
                trust_level = float(max(bcfg.trust_floor, trust_level * 0.6))
                cooldown_remaining = int(bcfg.cooldown_steps_after_reset)
            elif ctrl_risk >= warn_th:
                decision = "BLOCK_AND_REFLECT"
                status = "transitioning"
                human_sig = float(max(0.0, human_sig - bcfg.human_decay_scale * ctrl_risk + bcfg.human_recovery_base * 0.2))
            else:
                decision = "CONTINUE"
                status = "nominal"
                human_sig = float(min(bcfg.human_sig_max, human_sig + bcfg.human_recovery_base))

        # always update EMA
        human_ema = float(
            bcfg.ema_alpha_human * human_sig +
            (1.0 - bcfg.ema_alpha_human) * human_ema
        )

        if decision == "EMERGENCY_RESET":
            reset_streak += 1
        else:
            reset_streak = 0

        # log
        history["step"].append(t)
        history["h_sig"].append(human_sig)
        history["h_ema"].append(human_ema)
        history["risk"].append(ctrl_risk)
        history["risk_raw"].append(risk_raw)
        history["risk_ema"].append(risk_ema)
        history["trust"].append(trust_level)
        history["coherence"].append(coherence)
        history["decision"].append(decision)
        history["status"].append(status)
        history["momentum"].append(momentum)
        history["risk_input"].append(risk_input)
        history["trust_damping"].append(trust_damping)
        history["cooldown_remaining"].append(int(cooldown_remaining))
        history["S2"].append(S2)
        history["delta_S"].append(delta_S)
        history["external_human_proxy"].append(external_proxy)
        history["stress_event_delta"].append(stress_delta)
        history["instability_signal_base"].append(base_risk)
        history["instability_signal_final"].append(risk_input)
        history["baseline_mode"].append(mode)

        if early_stop_on_reset_streak is not None and reset_streak >= int(early_stop_on_reset_streak):
            break

    # baseline metrics (plus canonical analytics metrics for comparability)
    n = max(1, len(history["decision"]))
    reset_count = sum(1 for d in history["decision"] if d == "EMERGENCY_RESET")
    block_count = sum(1 for d in history["decision"] if d == "BLOCK_AND_REFLECT")
    continue_count = sum(1 for d in history["decision"] if d == "CONTINUE")
    cooldown_steps = sum(1 for c in history["cooldown_remaining"] if isinstance(c, (int, float)) and float(c) > 0)

    metrics = {
        "steps_executed": int(len(history["step"])),
        "requested_steps": int(steps),
        "baseline_mode": mode,
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
        "seed": int(seed),
    }

    analytics_metrics = compute_stability_metrics(history)

    return {
        "history": history,
        "metrics": metrics,
        "metrics_analytics": analytics_metrics,
        "baseline_mode": mode,
        "baseline_config": bcfg.to_dict(),
        "seed": int(seed),
        "steps": int(steps),
        "levels": int(levels),
        "config": cfg,
    }


def run_threshold_only_baseline(**kwargs) -> Dict[str, Any]:
    return _run_baseline_core(mode="threshold_only", **kwargs)


def run_ema_risk_only_baseline(**kwargs) -> Dict[str, Any]:
    return _run_baseline_core(mode="ema_risk_only", **kwargs)


def compare_against_baselines(
    main_result: Dict[str, Any],
    threshold_result: Dict[str, Any],
    ema_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compact comparison table-like dict using canonical analytics metrics.
    """
    def pick(m: Dict[str, Any]) -> Dict[str, Any]:
        risk_stats = (m or {}).get("risk_stats", {}) or {}
        trust_stats = (m or {}).get("trust_stats", {}) or {}
        hsig_stats = (m or {}).get("h_sig_stats", {}) or {}
        return {
            "steps": (m or {}).get("steps"),
            "continue_rate": (m or {}).get("continue_rate"),
            "block_rate": (m or {}).get("block_rate"),
            "reset_rate": (m or {}).get("reset_rate"),
            "reset_count": (m or {}).get("reset_count"),
            "cooldown_fraction": (m or {}).get("cooldown_fraction"),
            "stuck_transitioning_rate": (m or {}).get("stuck_transitioning_rate"),
            "risk_p95": risk_stats.get("p95"),
            "risk_max": risk_stats.get("max"),
            "trust_min": trust_stats.get("min"),
            "h_sig_min": hsig_stats.get("min"),
        }

    main_metrics = main_result.get("metrics_analytics") or compute_stability_metrics(main_result.get("history", {}))
    thr_metrics = threshold_result.get("metrics_analytics") or compute_stability_metrics(threshold_result.get("history", {}))
    ema_metrics = ema_result.get("metrics_analytics") or compute_stability_metrics(ema_result.get("history", {}))

    return {
        "main_adapter": pick(main_metrics),
        "threshold_only": pick(thr_metrics),
        "ema_risk_only": pick(ema_metrics),
    }


def print_baseline_comparison(compare_result: Dict[str, Any], title: str = "BASELINE COMPARISON") -> None:
    print("\\n" + "═" * 108)
    print(f"📉  {title}")
    print("═" * 108)

    if not isinstance(compare_result, dict):
        print("⚠️  Invalid comparison payload")
        print("═" * 108 + "\\n")
        return

    rows = []
    for name in ["main_adapter", "threshold_only", "ema_risk_only"]:
        r = compare_result.get(name, {}) or {}
        rows.append((name, r))

    print(
        f"{'System':<16} {'Steps':>6} {'C-rate':>8} {'B-rate':>8} {'R-rate':>8} "
        f"{'Resets':>7} {'CD-frac':>8} {'Risk p95':>9} {'Trust min':>10} {'Hsig min':>10}"
    )
    print("-" * 108)

    def fmt(x, nd=3):
        if x is None:
            return "n/a"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.{nd}f}"
        return str(x)

    for name, r in rows:
        print(
            f"{name:<16} "
            f"{fmt(r.get('steps'), 0):>6} "
            f"{fmt(r.get('continue_rate')):>8} "
            f"{fmt(r.get('block_rate')):>8} "
            f"{fmt(r.get('reset_rate')):>8} "
            f"{fmt(r.get('reset_count'), 0):>7} "
            f"{fmt(r.get('cooldown_fraction')):>8} "
            f"{fmt(r.get('risk_p95')):>9} "
            f"{fmt(r.get('trust_min')):>10} "
            f"{fmt(r.get('h_sig_min')):>10}"
        )

    print("═" * 108 + "\\n")


if __name__ == "__main__":
    # quick smoke demo (keeps runtime modest)
    from .simulation import run_long_horizon_simulation

    cfg = MetaProjectionStabilityConfig(seed=42, enable_plot=False, debug=False, verbose=False)
    shared_kwargs = dict(
        steps=1200,
        seed=42,
        cfg=cfg,
        use_noisy_significance=True,
        stress_events=[(120, 0.15), (121, 0.15), (600, 0.22), (900, 0.18)],
    )

    main_res = run_long_horizon_simulation(**shared_kwargs)
    thr_res = run_threshold_only_baseline(**shared_kwargs)
    ema_res = run_ema_risk_only_baseline(**shared_kwargs)

    cmp_res = compare_against_baselines(main_res, thr_res, ema_res)
    print_baseline_comparison(cmp_res)
