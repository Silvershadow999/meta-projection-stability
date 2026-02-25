from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import MetaProjectionStabilityConfig
from .experiment_runner import (
    ExperimentBatchConfig,
    run_experiment_batch,
    print_experiment_batch_summary,
    save_experiment_batch_json,
    save_experiment_batch_csv,
)
from .adversarial_sim import get_default_adversarial_scenarios


def _parse_args() -> argparse.Namespace:
    scenarios = sorted(get_default_adversarial_scenarios().keys())

    p = argparse.ArgumentParser(
        prog="meta_projection_stability.cli_experiments",
        description="Run batch experiments across scenarios / seeds / systems and optionally export CSV/JSON.",
    )

    p.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help=f"Scenario names (default: all). Available: {', '.join(scenarios)}",
    )
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seeds list, e.g. --seeds 41 42 43 (default internal preset)",
    )
    p.add_argument(
        "--systems",
        nargs="*",
        default=None,
        choices=["main_adapter", "threshold_only", "ema_risk_only"],
        help="Systems to run (default: all 3)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override scenario steps for all runs (recommended for quick tests)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode in config where supported",
    )

    # convenience presets
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke preset (small steps + few seeds/scenarios) unless explicitly overridden",
    )

    # export options
    p.add_argument(
        "--export",
        action="store_true",
        help="Export JSON and CSV to artifacts/ with timestamped filenames",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="artifacts",
        help="Output directory for exports (default: artifacts)",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="experiment_batch",
        help="Filename prefix for exports",
    )

    # silent-ish mode (summary still prints)
    p.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip terminal summary print",
    )

    return p.parse_args()


def _validate_scenarios(selected: Optional[List[str]]) -> List[str]:
    available = set(get_default_adversarial_scenarios().keys())
    if not selected:
        return sorted(list(available))

    invalid = [s for s in selected if s not in available]
    if invalid:
        raise SystemExit(
            f"Unknown scenarios: {invalid}. Available: {sorted(available)}"
        )
    return selected


def main() -> None:
    args = _parse_args()

    scenarios = _validate_scenarios(args.scenarios)
    seeds = args.seeds
    systems = args.systems
    steps = args.steps

    if args.quick:
        # only fill defaults if user did not explicitly provide them
        if args.scenarios is None:
            scenarios = ["spike_storm", "slow_drift"]
        if seeds is None:
            seeds = [41, 42]
        if systems is None:
            systems = ["main_adapter", "threshold_only", "ema_risk_only"]
        if steps is None:
            steps = 500

    cfg = MetaProjectionStabilityConfig(
        seed=(seeds[0] if seeds else 42),
        enable_plot=False,
        debug=bool(args.debug),
        verbose=False,
    )

    batch_cfg = ExperimentBatchConfig(
        scenario_names=scenarios,
        seeds=seeds,
        systems=systems,
        steps_override=steps,
        debug=bool(args.debug),
    )

    result = run_experiment_batch(batch_cfg=batch_cfg, cfg=cfg)

    if not args.no_summary:
        print_experiment_batch_summary(result, title="CLI EXPERIMENT BATCH")

    rows = result.get("rows", []) if isinstance(result, dict) else []
    errs = result.get("run_errors", []) if isinstance(result, dict) else []

    print(f"Completed runs: {len(rows)}")
    print(f"Errors: {len(errs)}")

    if errs:
        print("First error:", errs[0])

    if args.export:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        json_path = outdir / f"{args.prefix}_{ts}.json"
        csv_path = outdir / f"{args.prefix}_{ts}.csv"

        save_experiment_batch_json(result, str(json_path))
        save_experiment_batch_csv(result, str(csv_path))

        print(f"JSON saved: {json_path}")
        print(f"CSV  saved: {csv_path}")


if __name__ == "__main__":
    main()
