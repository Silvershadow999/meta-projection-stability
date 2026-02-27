from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from .adapter import MetaProjectionStabilityAdapter
from .config import MetaProjectionStabilityConfig
from .axiom_gateway import AxiomGateway


@dataclass
class ScenarioResult:
    name: str
    history: Dict[str, List[Any]]
    metrics: Dict[str, Any]


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def run_adversarial_scenario(
    name: str,
    steps: int = 1200,
    seed: int = 42,
    cfg: Optional[MetaProjectionStabilityConfig] = None,
) -> ScenarioResult:
    """
    Runs a deterministic adversarial scenario against the adapter.

    Scenarios:
      - sensor_freeze: biometric channels constant for long spans (stale sensor)
      - slow_drift_poison: gradual degradation of one channel (poison drift)
      - threshold_hover: attacker hovers just under risk gates to keep system in limbo
      - spoof_flip: alternating fake-high and fake-low signals (spoof / oscillation)
    """
    rng = np.random.default_rng(seed)
    cfg = cfg or MetaProjectionStabilityConfig(enable_plot=False)
    a = MetaProjectionStabilityAdapter(cfg)

    # Base layer input (stable-ish)
    S_layers = np.array([0.92, 0.90, 0.91], dtype=float)

    history: Dict[str, List[Any]] = {
        "instability_signal": [],
        "biometric_channels": [],
        "sensor_consensus": [],
        "biometric_proxy": [],
        "critical_channel_penalty": [],
        "consensus_penalty": [],
        "bio_penalty": [],
        "base_decay_effective": [],
        "instability_risk": [],
        "trust_level": [],
        "human_sig": [],
        "h_ema": [],
        "decision": [],
        "status": [],
        "mutual_bonus": [],
        "mutual_ema": [],
    }

    # --- Scenario generators ---
    def gen_sensor_freeze(t: int) -> Dict[str, Any]:
        # channels are unnaturally constant (freeze), occasional tiny noise
        base = [0.88, 0.87, 0.89]
        if 200 < t < 900:
            ch = base  # frozen
        else:
            ch = [b + 0.02 * rng.normal() for b in base]
        ch = [_clip01(v) for v in ch]
        instab = _clip01(0.10 + 0.06 * abs(rng.normal()))
        return {"instability_signal": instab, "biometric_channels": ch, "autonomy_proxy": 0.85, "mutuality_signal": 0.75}

    def gen_slow_drift_poison(t: int) -> Dict[str, Any]:
        # one channel slowly degrades; looks plausible but trends down
        drift = max(0.0, (t - 150) / 900)  # 0..1
        ch1 = _clip01(0.95 - 0.75 * drift + 0.02 * rng.normal())
        ch2 = _clip01(0.92 + 0.02 * rng.normal())
        ch3 = _clip01(0.90 + 0.02 * rng.normal())
        instab = _clip01(0.08 + 0.10 * drift + 0.05 * abs(rng.normal()))
        return {"instability_signal": instab, "biometric_channels": [ch1, ch2, ch3], "autonomy_proxy": 0.82, "mutuality_signal": 0.70}

    def gen_threshold_hover(t: int) -> Dict[str, Any]:
        # hover around stability gates: low-ish risk but persistent
        # makes the adapter live in "transition band" if tuned aggressively
        wobble = 0.03 * np.sin(t / 25.0)
        instab = _clip01(0.11 + wobble + 0.02 * rng.normal())
        # channels look good, but not perfect
        ch = [_clip01(0.86 + 0.03 * rng.normal()), _clip01(0.87 + 0.03 * rng.normal()), _clip01(0.85 + 0.03 * rng.normal())]
        return {"instability_signal": instab, "biometric_channels": ch, "autonomy_proxy": 0.80, "mutuality_signal": 0.60}



    # --- Axiom gateway-backed scenarios ---
    gateway = AxiomGateway(state_path="axiom_state.json")

    def gen_axiom_spoof_dos(t: int) -> Dict[str, Any]:
        # attacker flips hard_harm_commit high (should latch and lock)
        if t == 50:
            gateway.latch_harm_commit(reason="axiom_spoof_dos_trigger")
        ax = gateway.get_hard_harm_commit_signal()
        ch = [_clip01(0.90 + 0.02 * rng.normal()), _clip01(0.88 + 0.02 * rng.normal()), _clip01(0.89 + 0.02 * rng.normal())]
        raw = {"instability_signal": _clip01(0.10 + 0.02 * rng.normal()), "biometric_channels": ch, "autonomy_proxy": 0.85, "mutuality_signal": 0.5}
        raw.update(ax)
        return raw

    def gen_restart_clear_attempt(t: int) -> Dict[str, Any]:
        # Simulate restart: at t=100, we construct a new adapter instance (handled in loop by decision lock).
        # The gateway persists the latch across restarts.
        if t == 40:
            gateway.latch_harm_commit(reason="restart_clear_attempt_trigger")
        ax = gateway.get_hard_harm_commit_signal()
        ch = [_clip01(0.92 + 0.02 * rng.normal()), _clip01(0.93 + 0.02 * rng.normal()), _clip01(0.91 + 0.02 * rng.normal())]
        raw = {"instability_signal": _clip01(0.12 + 0.03 * rng.normal()), "biometric_channels": ch, "autonomy_proxy": 0.85, "mutuality_signal": 0.6}
        raw.update(ax)
        return raw

    def gen_lockdown_grief(t: int) -> Dict[str, Any]:
        # No harm commit, but constant tamper suspicion; should not latch, but may degrade.
        ax = gateway.get_hard_harm_commit_signal()  # stays 0
        wobble = 0.03 * np.sin(t / 18.0)
        instab = _clip01(0.12 + wobble + 0.02 * rng.normal())
        ch = [_clip01(0.86 + 0.03 * rng.normal()), _clip01(0.87 + 0.03 * rng.normal()), _clip01(0.85 + 0.03 * rng.normal())]
        raw = {"instability_signal": instab, "biometric_channels": ch, "autonomy_proxy": 0.78, "mutuality_signal": 0.55, "tamper_suspicion": 0.9}
        raw.update(ax)
        return raw
    def gen_spoof_flip(t: int) -> Dict[str, Any]:
        # alternating fake calm / fake collapse, tries to confuse EMA gates
        flip = (t // 30) % 2
        if flip == 0:
            ch = [0.98, 0.97, 0.96]  # fake-good
            instab = 0.06
            mutual = 0.95
        else:
            ch = [0.12, 0.98, 0.97]  # fake-bad on one channel
            instab = 0.22
            mutual = 0.10
        ch = [_clip01(v + 0.01 * rng.normal()) for v in ch]
        instab = _clip01(instab + 0.03 * rng.normal())
        return {"instability_signal": instab, "biometric_channels": ch, "autonomy_proxy": 0.75, "mutuality_signal": mutual}

    generators = {
        "axiom_spoof_dos": gen_axiom_spoof_dos,
        "restart_clear_attempt": gen_restart_clear_attempt,
        "lockdown_grief": gen_lockdown_grief,
        "sensor_freeze": gen_sensor_freeze,
        "slow_drift_poison": gen_slow_drift_poison,
        "threshold_hover": gen_threshold_hover,
        "spoof_flip": gen_spoof_flip,
    }
    if name not in generators:
        raise ValueError(f"Unknown scenario: {name}. Options: {list(generators)}")

    # --- Run loop ---
    for t in range(steps):
        raw = generators[name](t)

        # delta_S: light noise, slightly worse when instability rises (simulate coupling)
        delta_S = float(-0.004 * raw["instability_signal"] + 0.010 * rng.normal())

        out = a.interpret(S_layers=S_layers, delta_S=delta_S, raw_signals=raw)


        # Axiom hard-lock detection
        if out.get("decision") == "AXIOM_ZERO_LOCK":
            history["decision"].append(out.get("decision"))
            history["status"].append(out.get("status"))
            history["mutual_bonus"].append(out.get("mutual_bonus", 0.0))
            history["mutual_ema"].append(out.get("mutual_ema", 0.0))

            # fill numeric channels with None (keeps aligned length)
            history["instability_signal"].append(raw.get("instability_signal"))
            history["biometric_channels"].append(list(raw.get("biometric_channels", [])))
            for k in ("sensor_consensus","biometric_proxy","critical_channel_penalty","consensus_penalty","bio_penalty",
                      "base_decay_effective","instability_risk","trust_level","human_sig","h_ema"):
                history[k].append(out.get(k))

            # store lock marker and stop
            history.setdefault("axiom_locked_at_step", []).append(int(t))
            break

        # log
        history["instability_signal"].append(raw["instability_signal"])
        history["biometric_channels"].append(list(raw["biometric_channels"]))
        history["sensor_consensus"].append(out.get("sensor_consensus"))
        history["biometric_proxy"].append(out.get("biometric_proxy"))
        history["critical_channel_penalty"].append(out.get("critical_channel_penalty"))
        history["consensus_penalty"].append(out.get("consensus_penalty"))
        history["bio_penalty"].append(out.get("bio_penalty"))
        history["base_decay_effective"].append(out.get("base_decay_effective"))
        history["instability_risk"].append(out.get("instability_risk"))
        history["trust_level"].append(out.get("trust_level"))
        history["human_sig"].append(out.get("human_significance"))
        history["h_ema"].append(out.get("h_sig_ema"))
        history["decision"].append(out.get("decision"))
        history["status"].append(out.get("status"))
        history["mutual_bonus"].append(out.get("mutual_bonus", 0.0))
        history["mutual_ema"].append(out.get("mutual_ema", 0.0))

    # --- Metrics ---
    decisions = history["decision"]
    blocked = sum(1 for d in decisions if d in ("BLOCK_AND_REFLECT", "DEGRADED_VERIFY_MODE"))
    resets = sum(1 for d in decisions if d in ("EMERGENCY_RESET", "EMERGENCY_LOCKDOWN"))
    continue_count = sum(1 for d in decisions if d in ("CONTINUE", "CONTINUE_MONITORING"))

    risks = [r for r in history["instability_risk"] if isinstance(r, (int, float))]
    mean_risk = float(np.mean(risks)) if risks else None
    p95_risk = float(np.quantile(risks, 0.95)) if risks else None

    base_decay = [v for v in history["base_decay_effective"] if isinstance(v, (int, float))]
    mean_decay = float(np.mean(base_decay)) if base_decay else None

    metrics = {
        "steps": steps,

        "steps_executed": len(decisions),
        "axiom_locked": any(d == "AXIOM_ZERO_LOCK" for d in decisions),
        "locked_at_step": next((i for i,d in enumerate(decisions) if d == "AXIOM_ZERO_LOCK"), None),

        "blocked_steps": blocked,
        "blocked_share": blocked / max(1, steps),
        "continue_steps": continue_count,
        "resets": resets,
        "mean_risk": mean_risk,
        "p95_risk": p95_risk,
        "mean_base_decay_effective": mean_decay,
        "mean_mutual_bonus": float(np.mean(history["mutual_bonus"])) if history["mutual_bonus"] else 0.0,
        "final_trust": history["trust_level"][-1],
        "final_h_sig": history["human_sig"][-1],
        "final_h_ema": history["h_ema"][-1],
    }

    return ScenarioResult(name=name, history=history, metrics=metrics)


def run_all_scenarios(
    steps: int = 1200,
    seed: int = 42,
    cfg: Optional[MetaProjectionStabilityConfig] = None,
) -> List[ScenarioResult]:
    results = []
    for name in ("sensor_freeze", "slow_drift_poison", "threshold_hover", "spoof_flip", "axiom_spoof_dos", "restart_clear_attempt", "lockdown_grief"):
        results.append(run_adversarial_scenario(name=name, steps=steps, seed=seed, cfg=cfg))
    return results
