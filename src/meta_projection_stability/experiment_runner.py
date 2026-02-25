from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import json

from .config import MetaProjectionStabilityConfig
from .simulation import run_long_horizon_simulation
from .baseline_sim import (
    run_threshold_only_baseline,
    run_ema_risk_only_baseline,
)
from .analytics import compute_stability_metrics
from .adversarial_sim import get_default_adversarial_scenarios


@dataclass
class ExperimentBatchConfig:
    scenario_names: Optional[List[str]] = None
    seeds: Optional[List[int]] = None
    systems: Optional[List[str]] = None  # ["main_adapter", "threshold_only", "ema_risk_only"]
    steps_override: Optional[int] = None
    debug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _default_systems() -> List[str]:
    return ["main_adapter", "threshold_only", "ema_risk_only"]


def _default_seeds() -> List[int]:
    return [41, 42, 43]


def _extract_scenario_run_kwargs(
    scenario_name: str,
    seed: int,
    cfg: MetaProjectionStabilityConfig,
    steps_override: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    scenarios = get_default_adversarial_scenarios()
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {sorted(scenarios.keys())}")

    sc = scenarios[scenario_name]
    steps = int(steps_override) if steps_override is not None else int(sc.steps)

    # defensive config update
    try:
        cfg.seed = int(seed)
    except Exception:
        pass
    try:
        cfg.debug = bool(debug)
    except Exception:
        pass

    return {
        "steps": steps,
        "levels": 3,
        "seed": int(seed),
        "stress_test": bool(sc.stress_test),
        "cfg": cfg,
        "use_noisy_significance": bool(sc.use_noisy_significance),
        "noisy_sig_config": sc.noisy_sig_config,
        "stress_events": sc.stress_events,
        "early_stop_on_reset_streak": sc.early_stop_on_reset_streak,
    }


def _run_system(system_name: str, run_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if system_name == "main_adapter":
        return run_long_horizon_simulation(**run_kwargs)
    if system_name == "threshold_only":
        return run_threshold_only_baseline(**run_kwargs)
    if system_name == "ema_risk_only":
        return run_ema_risk_only_baseline(**run_kwargs)
    raise ValueError(f"Unknown system '{system_name}'")


def _flatten_row(
    system_name: str,
    scenario_name: str,
    seed: int,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    m = result.get("metrics_analytics")
    if not isinstance(m, dict) or not m:
        m = compute_stability_metrics(result.get("history", {}))

    risk_stats = (m.get("risk_stats") or {}) if isinstance(m, dict) else {}
    trust_stats = (m.get("trust_stats") or {}) if isinstance(m, dict) else {}
    hsig_stats = (m.get("h_sig_stats") or {}) if isinstance(m, dict) else {}

    row = {
        "system": system_name,
        "scenario": scenario_name,
        "seed": int(seed),
        "valid": bool(m.get("valid", False)) if isinstance(m, dict) else False,
        "steps": m.get("steps") if isinstance(m, dict) else None,

        "continue_rate": m.get("continue_rate") if isinstance(m, dict) else None,
        "block_rate": m.get("block_rate") if isinstance(m, dict) else None,
        "reset_rate": m.get("reset_rate") if isinstance(m, dict) else None,

        "continue_count": m.get("continue_count") if isinstance(m, dict) else None,
        "block_count": m.get("block_count") if isinstance(m, dict) else None,
        "reset_count": m.get("reset_count") if isinstance(m, dict) else None,

        "nominal_fraction": m.get("nominal_fraction") if isinstance(m, dict) else None,
        "transitioning_fraction": m.get("transitioning_fraction") if isinstance(m, dict) else None,
        "cooldown_fraction": m.get("cooldown_fraction") if isinstance(m, dict) else None,

        "max_block_streak": m.get("max_block_streak") if isinstance(m, dict) else None,
        "max_reset_streak": m.get("max_reset_streak") if isinstance(m, dict) else None,
        "max_cooldown_streak": m.get("max_cooldown_streak") if isinstance(m, dict) else None,

        "avg_block_streak": m.get("avg_block_streak") if isinstance(m, dict) else None,
        "avg_cooldown_streak": m.get("avg_cooldown_streak") if isinstance(m, dict) else None,

        "false_positive_block_rate": m.get("false_positive_block_rate") if isinstance(m, dict) else None,
        "stuck_transitioning_rate": m.get("stuck_transitioning_rate") if isinstance(m, dict) else None,

        "time_to_first_block": m.get("time_to_first_block") if isinstance(m, dict) else None,
        "time_to_first_reset": m.get("time_to_first_reset") if isinstance(m, dict) else None,

        "risk_min": risk_stats.get("min"),
        "risk_mean": risk_stats.get("mean"),
        "risk_p95": risk_stats.get("p95"),
        "risk_max": risk_stats.get("max"),

        "trust_min": trust_stats.get("min"),
        "trust_mean": trust_stats.get("mean"),
        "trust_p95": trust_stats.get("p95"),
        "trust_max": trust_stats.get("max"),

        "h_sig_min": hsig_stats.get("min"),
        "h_sig_mean": hsig_stats.get("mean"),
        "h_sig_p95": hsig_stats.get("p95"),
        "h_sig_max": hsig_stats.get("max"),
    }
    return row


def _group_mean_std(rows: List[Dict[str, Any]], group_keys: List[str], metric_keys: List[str]) -> List[Dict[str, Any]]:
    import math
    import statistics

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        key = tuple(r.get(k) for k in group_keys)
        grouped.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []

    def _num_list(vals):
        arr = []
        for v in vals:
            try:
                fv = float(v)
                if math.isfinite(fv):
                    arr.append(fv)
            except (TypeError, ValueError):
                continue
        return arr

    for key, grp in grouped.items():
        row: Dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        row["runs"] = len(grp)

        for mk in metric_keys:
            vals = _num_list([g.get(mk) for g in grp])
            if not vals:
                row[f"{mk}_mean"] = None
                row[f"{mk}_std"] = None
                row[f"{mk}_min"] = None
                row[f"{mk}_max"] = None
            else:
                row[f"{mk}_mean"] = float(sum(vals) / len(vals))
                row[f"{mk}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
                row[f"{mk}_min"] = float(min(vals))
                row[f"{mk}_max"] = float(max(vals))

        out.append(row)

    # stable sort
    out.sort(key=lambda r: tuple(str(r.get(k)) for k in group_keys))
    return out


def run_experiment_batch(
    batch_cfg: Optional[ExperimentBatchConfig] = None,
    cfg: Optional[MetaProjectionStabilityConfig] = None,
) -> Dict[str, Any]:
    """
    Run a batch over (scenario x seed x system), collect flat rows, and aggregate.
    """
    batch_cfg = batch_cfg or ExperimentBatchConfig()
    systems = batch_cfg.systems or _default_systems()
    seeds = batch_cfg.seeds or _default_seeds()
    scenarios_map = get_default_adversarial_scenarios()
    scenario_names = batch_cfg.scenario_names or list(scenarios_map.keys())

    rows: List[Dict[str, Any]] = []
    run_errors: List[Dict[str, Any]] = []

    for scenario_name in scenario_names:
        for seed in seeds:
            # fresh config per run to reduce accidental state leakage
            cfg_run = cfg or MetaProjectionStabilityConfig(seed=int(seed), enable_plot=False, debug=batch_cfg.debug, verbose=False)
            if cfg is not None:
                # best effort shallow reset-ish values
                try:
                    cfg_run.seed = int(seed)
                except Exception:
                    pass
                try:
                    cfg_run.debug = bool(batch_cfg.debug)
                except Exception:
                    pass

            run_kwargs = _extract_scenario_run_kwargs(
                scenario_name=scenario_name,
                seed=int(seed),
                cfg=cfg_run,
                steps_override=batch_cfg.steps_override,
                debug=batch_cfg.debug,
            )

            for system_name in systems:
                try:
                    result = _run_system(system_name, run_kwargs)
                    row = _flatten_row(system_name, scenario_name, int(seed), result)
                    rows.append(row)
                except Exception as e:
                    run_errors.append({
                        "system": system_name,
                        "scenario": scenario_name,
                        "seed": int(seed),
                        "error": repr(e),
                    })

    metric_keys = [
        "continue_rate",
        "block_rate",
        "reset_rate",
        "cooldown_fraction",
        "stuck_transitioning_rate",
        "false_positive_block_rate",
        "risk_p95",
        "risk_max",
        "trust_min",
        "trust_mean",
        "h_sig_min",
        "h_sig_mean",
        "time_to_first_reset",
    ]

    agg_by_system_scenario = _group_mean_std(
        rows=rows,
        group_keys=["system", "scenario"],
        metric_keys=metric_keys,
    )
    agg_by_system = _group_mean_std(
        rows=rows,
        group_keys=["system"],
        metric_keys=metric_keys,
    )
    agg_by_scenario = _group_mean_std(
        rows=rows,
        group_keys=["scenario"],
        metric_keys=metric_keys,
    )

    return {
        "batch_valid": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_config": batch_cfg.to_dict(),
        "rows": rows,
        "run_errors": run_errors,
        "aggregates": {
            "by_system_scenario": agg_by_system_scenario,
            "by_system": agg_by_system,
            "by_scenario": agg_by_scenario,
        },
    }


def save_experiment_batch_json(batch_result: Dict[str, Any], path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(batch_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def save_experiment_batch_csv(batch_result: Dict[str, Any], path: str) -> str:
    """
    Saves flat per-run rows to CSV (best for quick spreadsheet analysis).
    """
    rows = batch_result.get("rows", []) if isinstance(batch_result, dict) else []
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        p.write_text("", encoding="utf-8")
        return str(p)

    # union of keys for stable schema
    fieldnames = sorted({k for r in rows if isinstance(r, dict) for k in r.keys()})
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if isinstance(r, dict):
                w.writerow(r)
    return str(p)


def print_experiment_batch_summary(batch_result: Dict[str, Any], title: str = "EXPERIMENT BATCH SUMMARY") -> None:
    print("\n" + "═" * 120)
    print(f"🧾  {title}")
    print("═" * 120)

    if not isinstance(batch_result, dict) or not batch_result.get("batch_valid", False):
        print("⚠️  Invalid batch result payload")
        print("═" * 120 + "\n")
        return

    rows = batch_result.get("rows", []) or []
    errs = batch_result.get("run_errors", []) or []
    print(f"  Runs completed: {len(rows)}")
    print(f"  Errors:         {len(errs)}")
    if errs:
        print("  First error:", errs[0])

    agg = (((batch_result.get("aggregates") or {}).get("by_system_scenario")) or [])
    if not agg:
        print("⚠️  No aggregate rows")
        print("═" * 120 + "\n")
        return

    print()
    print(
        f"{'System':<16} {'Scenario':<20} {'Runs':>4} "
        f"{'C-rate μ±σ':>18} {'B-rate μ±σ':>18} {'R-rate μ±σ':>18} "
        f"{'Risk p95 μ':>10} {'Trust min μ':>12} {'Hsig min μ':>11}"
    )
    print("-" * 120)

    def _fmt(x, nd=3):
        if x is None:
            return "n/a"
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    def _fmt_mu_sigma(mu, sd, nd=3):
        if mu is None:
            return "n/a"
        if sd is None:
            sd = 0.0
        return f"{float(mu):.{nd}f}±{float(sd):.{nd}f}"

    for r in agg:
        print(
            f"{str(r.get('system', 'n/a')):<16} "
            f"{str(r.get('scenario', 'n/a')):<20} "
            f"{int(r.get('runs', 0)):>4} "
            f"{_fmt_mu_sigma(r.get('continue_rate_mean'), r.get('continue_rate_std')):>18} "
            f"{_fmt_mu_sigma(r.get('block_rate_mean'), r.get('block_rate_std')):>18} "
            f"{_fmt_mu_sigma(r.get('reset_rate_mean'), r.get('reset_rate_std')):>18} "
            f"{_fmt(r.get('risk_p95_mean')):>10} "
            f"{_fmt(r.get('trust_min_mean')):>12} "
            f"{_fmt(r.get('h_sig_min_mean')):>11}"
        )

    print("═" * 120 + "\n")


if __name__ == "__main__":
    cfg = MetaProjectionStabilityConfig(seed=42, enable_plot=False, debug=False, verbose=False)
    batch_cfg = ExperimentBatchConfig(
        scenario_names=["spike_storm", "slow_drift"],
        seeds=[41, 42],
        systems=["main_adapter", "threshold_only", "ema_risk_only"],
        steps_override=800,
        debug=False,
    )
    result = run_experiment_batch(batch_cfg=batch_cfg, cfg=cfg)
    print_experiment_batch_summary(result, title="Quick Batch Demo")
    save_experiment_batch_json(result, "artifacts/experiment_batch_demo.json")
    save_experiment_batch_csv(result, "artifacts/experiment_batch_demo.csv")
