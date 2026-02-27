from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import MetaProjectionStabilityConfig
from .simulation import run_long_horizon_simulation
from .analytics import compute_stability_metrics


@dataclass
class AdversarialScenario:
    """
    Declarative scenario definition for long-horizon adversarial/stress simulations.
    """
    name: str
    steps: int = 5000
    stress_events: Optional[List[Tuple[int, float]]] = None
    noisy_sig_config: Optional[Dict[str, Any]] = None
    use_noisy_significance: bool = True
    stress_test: bool = False
    early_stop_on_reset_streak: Optional[int] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _generate_periodic_spikes(
    start: int,
    stop: int,
    every: int,
    amplitude: float,
    jitter: int = 0,
    seed: int = 42,
) -> List[Tuple[int, float]]:
    rng = np.random.default_rng(seed)
    events: List[Tuple[int, float]] = []
    for t in range(start, stop, max(1, every)):
        jt = int(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0
        events.append((max(1, t + jt), float(amplitude)))
    return events


def get_default_adversarial_scenarios() -> Dict[str, AdversarialScenario]:
    """
    Curated first-pass scenarios targeting your current architecture:
    - spikes / oscillation
    - drift/noise amplification
    - cooldown pressure
    """
    return {
        "spike_storm": AdversarialScenario(
            name="spike_storm",
            steps=6000,
            stress_events=[(i, 0.28) for i in range(400, 2600, 120)],
            noisy_sig_config={
                "base_value": 0.84,
                "ou_theta": 0.05,
                "ou_sigma": 0.08,
                "spike_prob": 0.01,
                "spike_mag": (-0.25, 0.12),
            },
            description="Frequent medium spikes to test block/reset resilience under repeated shocks.",
        ),
        "slow_drift": AdversarialScenario(
            name="slow_drift",
            steps=8000,
            stress_events=[],
            noisy_sig_config={
                "base_value": 0.82,
                "ou_theta": 0.015,   # weak mean reversion
                "ou_sigma": 0.13,    # noisy enough to drift
                "spike_prob": 0.002,
                "spike_mag": (-0.15, 0.08),
            },
            description="Slow degradation / drift pressure to expose hidden long-horizon fragility.",
        ),
        "oscillation_attack": AdversarialScenario(
            name="oscillation_attack",
            steps=7000,
            stress_events=(
                [(t, 0.22) for t in range(500, 3500, 90)] +
                [(t, -0.08) for t in range(545, 3545, 90)]
            ),
            noisy_sig_config={
                "base_value": 0.85,
                "ou_theta": 0.07,
                "ou_sigma": 0.06,
                "spike_prob": 0.004,
                "spike_mag": (-0.18, 0.10),
            },
            description="Alternating spikes and relief to stress hysteresis and chattering resistance.",
        ),
        "cooldown_pin": AdversarialScenario(
            name="cooldown_pin",
            steps=6500,
            stress_events=_generate_periodic_spikes(
                start=300, stop=4200, every=75, amplitude=0.32, jitter=8, seed=123
            ),
            noisy_sig_config={
                "base_value": 0.78,
                "ou_theta": 0.03,
                "ou_sigma": 0.10,
                "spike_prob": 0.006,
                "spike_mag": (-0.22, 0.08),
            },
            early_stop_on_reset_streak=8,
            description="Pushes system toward repeated resets/cooldowns to detect cooldown pinning.",
        ),
    }


def run_adversarial_scenario(
    scenario_name: str,
    cfg: Optional[MetaProjectionStabilityConfig] = None,
    seed: int = 42,
    steps_override: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run one named adversarial scenario and return simulation result + scenario metadata + metrics.
    """
    scenarios = get_default_adversarial_scenarios()
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {sorted(scenarios.keys())}")

    scenario = scenarios[scenario_name]
    steps = int(steps_override) if steps_override is not None else int(scenario.steps)

    # Defensive copy of config (or construct a fresh one)
    if cfg is None:
        cfg_local = MetaProjectionStabilityConfig(seed=seed, enable_plot=False, debug=debug, verbose=False)
    else:
        cfg_local = cfg
        # best effort to keep deterministic metadata coherent
        if hasattr(cfg_local, "seed"):
            try:
                cfg_local.seed = int(seed)
            except Exception:
                pass
        if hasattr(cfg_local, "debug"):
            try:
                cfg_local.debug = bool(debug)
            except Exception:
                pass

    result = run_long_horizon_simulation(
        steps=steps,
        levels=3,
        seed=seed,
        stress_test=bool(scenario.stress_test),
        cfg=cfg_local,
        use_noisy_significance=bool(scenario.use_noisy_significance),
        noisy_sig_config=scenario.noisy_sig_config,
        stress_events=scenario.stress_events,
        early_stop_on_reset_streak=scenario.early_stop_on_reset_streak,
    )

    # Recompute with canonical analytics function for consistent schema
    metrics = compute_stability_metrics(result.get("history", {}))
    result["metrics_analytics"] = metrics
    result["scenario"] = scenario.to_dict()
    result["scenario_name"] = scenario.name
    result["seed"] = int(seed)

    return result


def run_adversarial_suite(
    scenario_names: Optional[List[str]] = None,
    cfg: Optional[MetaProjectionStabilityConfig] = None,
    seed: int = 42,
    steps_override: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run multiple adversarial scenarios and return a compact summary + per-scenario results.
    """
    scenarios = get_default_adversarial_scenarios()
    names = scenario_names or list(scenarios.keys())

    per_scenario: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for name in names:
        res = run_adversarial_scenario(
            scenario_name=name,
            cfg=cfg,
            seed=seed,
            steps_override=steps_override,
            debug=debug,
        )
        per_scenario[name] = res

        m = res.get("metrics_analytics", {}) or {}
        risk_stats = m.get("risk_stats", {}) or {}
        trust_stats = m.get("trust_stats", {}) or {}
        hsig_stats = m.get("h_sig_stats", {}) or {}

        summary_rows.append({
            "scenario": name,
            "steps": m.get("steps"),
            "continue_rate": m.get("continue_rate"),
            "block_rate": m.get("block_rate"),
            "reset_rate": m.get("reset_rate"),
            "reset_count": m.get("reset_count"),
            "cooldown_fraction": m.get("cooldown_fraction"),
            "stuck_transitioning_rate": m.get("stuck_transitioning_rate"),
            "risk_p95": risk_stats.get("p95"),
            "risk_max": risk_stats.get("max"),
            "trust_min": trust_stats.get("min"),
            "h_sig_min": hsig_stats.get("min"),
            "time_to_first_reset": m.get("time_to_first_reset"),
        })

    return {
        "suite_valid": True,
        "seed": int(seed),
        "scenario_names": names,
        "summary": summary_rows,
        "results": per_scenario,
    }


def print_adversarial_suite_summary(suite_result: Dict[str, Any], title: str = "ADVERSARIAL SUITE SUMMARY") -> None:
    """
    Compact terminal summary without requiring pandas.
    """
    print("\n" + "═" * 92)
    print(f"⚔️  {title}")
    print("═" * 92)

    if not isinstance(suite_result, dict) or not suite_result.get("suite_valid", False):
        print("⚠️  Invalid suite result payload")
        print("═" * 92 + "\n")
        return

    rows = suite_result.get("summary", []) or []
    if not rows:
        print("⚠️  No summary rows")
        print("═" * 92 + "\n")
        return

    print(
        f"{'Scenario':<20} {'Steps':>6} {'C-rate':>8} {'B-rate':>8} {'R-rate':>8} "
        f"{'Resets':>7} {'CD-frac':>8} {'Risk p95':>9} {'Trust min':>10} {'Hsig min':>10}"
    )
    print("-" * 92)

    for r in rows:
        def fmt(x, nd=3):
            if x is None:
                return "n/a"
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            if isinstance(x, (float, np.floating)):
                return f"{float(x):.{nd}f}"
            return str(x)

        print(
            f"{str(r.get('scenario', 'n/a')):<20} "
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

    print("═" * 92 + "\n")


if __name__ == "__main__":
    cfg = MetaProjectionStabilityConfig(seed=42, enable_plot=False, debug=False, verbose=False)
    suite = run_adversarial_suite(cfg=cfg, seed=42, steps_override=1500)
    print_adversarial_suite_summary(suite)


def run_adversarial_suite_multi_seed(
    scenario_names: Optional[List[str]] = None,
    cfg: Optional[MetaProjectionStabilityConfig] = None,
    seeds: Optional[List[int]] = None,
    steps_override: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run adversarial suite across multiple seeds and aggregate key metrics per scenario.

    Returns:
      {
        "multi_seed_valid": True,
        "seeds": [...],
        "scenario_names": [...],
        "per_seed": {seed: suite_result, ...},
        "aggregate": [
           {
             "scenario": ...,
             "runs": N,
             "continue_rate_mean": ...,
             "continue_rate_std": ...,
             ...
           }, ...
        ]
      }
    """
    import math
    import statistics

    if seeds is None or len(seeds) == 0:
        seeds = [42, 43, 44, 45, 46]

    scenarios = get_default_adversarial_scenarios()
    names = scenario_names or list(scenarios.keys())

    per_seed: Dict[str, Any] = {}
    rows_by_scenario: Dict[str, List[Dict[str, Any]]] = {name: [] for name in names}

    for seed in seeds:
        suite = run_adversarial_suite(
            scenario_names=names,
            cfg=cfg,
            seed=int(seed),
            steps_override=steps_override,
            debug=debug,
        )
        per_seed[str(int(seed))] = suite

        for row in suite.get("summary", []) or []:
            sc = row.get("scenario")
            if sc in rows_by_scenario:
                rows_by_scenario[sc].append(row)

    def _to_float_list(values):
        out = []
        for v in values:
            try:
                fv = float(v)
                if math.isfinite(fv):
                    out.append(fv)
            except (TypeError, ValueError):
                continue
        return out

    def _agg(vals):
        vals = _to_float_list(vals)
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        if len(vals) == 1:
            return {
                "mean": float(vals[0]),
                "std": 0.0,
                "min": float(vals[0]),
                "max": float(vals[0]),
            }
        return {
            "mean": float(sum(vals) / len(vals)),
            "std": float(statistics.pstdev(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    aggregate: List[Dict[str, Any]] = []

    metric_fields = [
        "continue_rate",
        "block_rate",
        "reset_rate",
        "reset_count",
        "cooldown_fraction",
        "stuck_transitioning_rate",
        "risk_p95",
        "risk_max",
        "trust_min",
        "h_sig_min",
        "time_to_first_reset",
    ]

    for name in names:
        rows = rows_by_scenario.get(name, [])
        agg_row: Dict[str, Any] = {
            "scenario": name,
            "runs": int(len(rows)),
        }

        # steps can vary if early stop is active; aggregate too
        step_vals = [r.get("steps") for r in rows]
        step_stats = _agg(step_vals)
        agg_row["steps_mean"] = step_stats["mean"]
        agg_row["steps_std"] = step_stats["std"]
        agg_row["steps_min"] = step_stats["min"]
        agg_row["steps_max"] = step_stats["max"]

        for field in metric_fields:
            stats = _agg([r.get(field) for r in rows])
            agg_row[f"{field}_mean"] = stats["mean"]
            agg_row[f"{field}_std"] = stats["std"]
            agg_row[f"{field}_min"] = stats["min"]
            agg_row[f"{field}_max"] = stats["max"]

        aggregate.append(agg_row)

    return {
        "multi_seed_valid": True,
        "seeds": [int(s) for s in seeds],
        "scenario_names": names,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def print_adversarial_multi_seed_summary(
    multi_result: Dict[str, Any],
    title: str = "ADVERSARIAL MULTI-SEED SUMMARY",
) -> None:
    """
    Compact summary of aggregated scenario metrics over multiple seeds.
    """
    print("\n" + "═" * 116)
    print(f"🧪  {title}")
    print("═" * 116)

    if not isinstance(multi_result, dict) or not multi_result.get("multi_seed_valid", False):
        print("⚠️  Invalid multi-seed result payload")
        print("═" * 116 + "\n")
        return

    seeds = multi_result.get("seeds", [])
    print(f"  Seeds: {seeds}")
    print()

    rows = multi_result.get("aggregate", []) or []
    if not rows:
        print("⚠️  No aggregate rows")
        print("═" * 116 + "\n")
        return

    print(
        f"{'Scenario':<20} {'Runs':>4} "
        f"{'C-rate μ±σ':>18} {'B-rate μ±σ':>18} {'R-rate μ±σ':>18} "
        f"{'Risk p95 μ':>10} {'Trust min μ':>12} {'Hsig min μ':>11}"
    )
    print("-" * 116)

    def _fmt_mu_sigma(mu, sd, nd=3):
        if mu is None:
            return "n/a"
        if sd is None:
            sd = 0.0
        return f"{float(mu):.{nd}f}±{float(sd):.{nd}f}"

    def _fmt(x, nd=3):
        if x is None:
            return "n/a"
        return f"{float(x):.{nd}f}"

    for r in rows:
        print(
            f"{str(r.get('scenario', 'n/a')):<20} "
            f"{int(r.get('runs', 0)):>4} "
            f"{_fmt_mu_sigma(r.get('continue_rate_mean'), r.get('continue_rate_std')):>18} "
            f"{_fmt_mu_sigma(r.get('block_rate_mean'), r.get('block_rate_std')):>18} "
            f"{_fmt_mu_sigma(r.get('reset_rate_mean'), r.get('reset_rate_std')):>18} "
            f"{_fmt(r.get('risk_p95_mean')):>10} "
            f"{_fmt(r.get('trust_min_mean')):>12} "
            f"{_fmt(r.get('h_sig_min_mean')):>11}"
        )

    print("═" * 116 + "\n")
