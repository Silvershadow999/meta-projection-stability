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
from .ranking import (
    ScoreWeights,
    rank_batch_rows,
    aggregate_ranked_rows,
    print_top_rankings,
    save_ranked_rows_csv,
    save_aggregate_ranking_csv,
)
from .profiles import list_profiles, describe_profiles, apply_profile


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
        "--profiles",
        nargs="*",
        default=None,
        help="Profile list for sweeps, e.g. --profiles balanced protective strict_safety",
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
        "--profile",
        type=str,
        default="balanced",
        help="Config profile for main adapter runs (and shared cfg defaults): balanced | protective | aggressive_recovery | strict_safety",
    )
    p.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available config profiles and exit",
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
        "--rank",
        action="store_true",
        help="Compute and print ranking for batch rows",
    )
    p.add_argument(
        "--rank-group-by",
        nargs="*",
        default=None,
        help="Grouping keys for aggregate ranking (default: profile system scenario)",
    )
    p.add_argument(
        "--rank-top",
        type=int,
        default=12,
        help="Top N rows/groups to print in ranking view (default: 12)",
    )
    p.add_argument(
        "--rank-export",
        action="store_true",
        help="Export ranking CSVs (ranked rows + aggregate groups)",
    )
    p.add_argument(
        "--rank-preset",
        type=str,
        default="balanced",
        help="Ranking weight preset: balanced | safety_first | throughput",
    )
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


def _validate_profiles(selected: Optional[List[str]]) -> List[str]:
    available = set(list_profiles())
    if not selected:
        return ["balanced"]

    invalid = [p for p in selected if p not in available]
    if invalid:
        raise SystemExit(
            f"Unknown profiles: {invalid}. Available: {sorted(available)}"
        )
    return selected



def _make_rank_weights_from_preset(preset: str) -> ScoreWeights:
    """
    Presets for ranking priorities.
    """
    preset = (preset or "balanced").strip().lower()

    if preset == "balanced":
        return ScoreWeights()

    if preset == "safety_first":
        return ScoreWeights(
            continue_rate=0.85,
            block_rate=-0.10,
            reset_rate=-1.80,
            cooldown_fraction=-0.45,
            stuck_transitioning_rate=-1.10,
            false_positive_block_rate=-0.55,
            risk_p95=-1.20,
            risk_max=-0.80,
            trust_min=0.70,
            trust_mean=0.25,
            h_sig_min=0.90,
            h_sig_mean=0.25,
            time_to_first_reset=0.20,
            time_to_first_block=0.02,
        )

    if preset == "throughput":
        return ScoreWeights(
            continue_rate=1.25,
            block_rate=-0.08,
            reset_rate=-1.20,
            cooldown_fraction=-0.20,
            stuck_transitioning_rate=-0.65,
            false_positive_block_rate=-0.95,
            risk_p95=-0.70,
            risk_max=-0.35,
            trust_min=0.45,
            trust_mean=0.20,
            h_sig_min=0.50,
            h_sig_mean=0.18,
            time_to_first_reset=0.10,
            time_to_first_block=0.08,
        )

    raise SystemExit(
        "Unknown --rank-preset. Use one of: balanced, safety_first, throughput"
    )


def main() -> None:
    args = _parse_args()

    if args.list_profiles:
        desc = describe_profiles()
        print("Available profiles:")
        for name in list_profiles():
            print(f" - {name}: {desc.get(name, '')}")
        return

    scenarios = _validate_scenarios(args.scenarios)
    profiles = _validate_profiles(args.profiles)
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
        if args.profiles is None:
            profiles = ["balanced", "protective"]
        if steps is None:
            steps = 500

    cfg = MetaProjectionStabilityConfig(
        seed=(seeds[0] if seeds else 42),
        enable_plot=False,
        debug=bool(args.debug),
        verbose=False,
    )

    apply_profile(cfg, profile_name=args.profile)

    batch_cfg = ExperimentBatchConfig(
        scenario_names=scenarios,
        seeds=seeds,
        systems=systems,
        profiles=profiles,
        steps_override=steps,
        debug=bool(args.debug),
    )

    result = run_experiment_batch(batch_cfg=batch_cfg, cfg=cfg)

    if not args.no_summary:
        print_experiment_batch_summary(result, title="CLI EXPERIMENT BATCH")

    rows = result.get("rows", []) if isinstance(result, dict) else []
    errs = result.get("run_errors", []) if isinstance(result, dict) else []

    print(f"Profile: {args.profile}")
    print(f"Profiles: {profiles}")
    print(f"Completed runs: {len(rows)}")
    print(f"Errors: {len(errs)}")

    if errs:
        print("First error:", errs[0])


    # -------------------------
    # Optional ranking layer
    # -------------------------
    if args.rank:
        group_keys = args.rank_group_by or ["profile", "system", "scenario"]

        # defensive cleanup
        group_keys = [g for g in group_keys if isinstance(g, str) and g.strip()]
        if not group_keys:
            group_keys = ["profile", "system", "scenario"]

        rank_weights = _make_rank_weights_from_preset(args.rank_preset)

        ranked = rank_batch_rows(result, weights=rank_weights)
        agg_rank = aggregate_ranked_rows(ranked, group_keys=tuple(group_keys))

        print_top_rankings(
            ranked,
            agg_rank,
            top_n_rows=max(1, int(args.rank_top)),
            top_n_groups=max(1, int(args.rank_top)),
            title=f"CLI BATCH RANKING ({args.rank_preset})",
        )

        if args.rank_export:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)

            ranked_csv = outdir / f"{args.prefix}_{ts}_ranked_rows.csv"
            grouped_csv = outdir / f"{args.prefix}_{ts}_rank_groups.csv"

            save_ranked_rows_csv(ranked, str(ranked_csv))
            save_aggregate_ranking_csv(agg_rank, str(grouped_csv))

            print(f"Ranking CSV saved: {ranked_csv}")
            print(f"Group   CSV saved: {grouped_csv}")

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
