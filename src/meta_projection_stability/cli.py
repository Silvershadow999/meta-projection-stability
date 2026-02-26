from __future__ import annotations

import argparse
from typing import Any, Dict

from .config import MetaProjectionStabilityConfig
from .simulation import run_simulation

try:
    from .plotting import plot_results, print_summary
except Exception:
    plot_results = None
    print_summary = None

try:
    from .globalsense import GlobalSense
except Exception:
    GlobalSense = None

try:
    from .adversarial import run_adversarial_scenario, run_all_scenarios
except Exception:
    run_adversarial_scenario = None
    run_all_scenarios = None


def _print_kv_block(title: str, data: Dict[str, Any]) -> None:
    print(f"\n== {title} ==")
    for key in sorted(data.keys()):
        print(f"{key}: {data[key]}")


def _safe_print_summary(result: Any) -> None:
    if callable(print_summary):
        try:
            print_summary(result)
            return
        except Exception as exc:
            print(f"[CLI] print_summary failed: {exc}")

    if isinstance(result, dict):
        _print_kv_block("summary", result)
    else:
        print(result)


def _safe_plot(result: Any) -> None:
    if callable(plot_results):
        try:
            plot_results(result)
        except Exception as exc:
            print(f"[CLI] plotting skipped due to error: {exc}")
    else:
        print("[CLI] plotting unavailable: plot_results not found")


def _apply_global_sense(cfg: MetaProjectionStabilityConfig, args: argparse.Namespace) -> MetaProjectionStabilityConfig:
    if not getattr(args, "global_sense", False):
        return cfg

    if GlobalSense is None:
        print("[CLI] GlobalSense not available in this build; continuing without it.")
        return cfg

    try:
        gs = GlobalSense()
        return gs.map_to_config(
            cfg,
            use_stress_index=True,
            force_refresh=getattr(args, "gs_force_refresh", False),
            debug=getattr(args, "gs_debug", False),
        )
    except Exception as exc:
        print(f"[CLI] GlobalSense disabled due to error: {exc}")
        return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meta_projection_stability.cli",
        description="Meta Projection Stability CLI",
    )

    subparsers = parser.add_subparsers(dest="command")

    sim = subparsers.add_parser("simulate", help="Run the main simulation")
    sim.add_argument("--steps", type=int, default=1400, help="Number of simulation steps")
    sim.add_argument("--levels", type=int, default=3, help="Number of layers")
    sim.add_argument("--seed", type=int, default=42, help="Random seed")
    sim.add_argument("--stress-test", action="store_true", help="Enable built-in stress window")
    sim.add_argument("--no-plot", action="store_true", help="Skip plotting")
    sim.add_argument("--global-sense", action="store_true", help="Enable real-world macro modulation before simulation")
    sim.add_argument("--gs-debug", action="store_true", help="Print GlobalSense debug output")
    sim.add_argument("--gs-force-refresh", action="store_true", help="Ignore GlobalSense cache and fetch fresh values")

    adv = subparsers.add_parser("adversarial", help="Run a named adversarial scenario")
    adv.add_argument("--scenario", type=str, required=True, help="Scenario name")
    adv.add_argument("--steps", type=int, default=120, help="Scenario steps")
    adv.add_argument("--seed", type=int, default=42, help="Random seed")

    all_adv = subparsers.add_parser("all-scenarios", help="Run all adversarial scenarios")
    all_adv.add_argument("--steps", type=int, default=120, help="Scenario steps")
    all_adv.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser


def _default_cfg() -> MetaProjectionStabilityConfig:
    return MetaProjectionStabilityConfig(
        risk_critical_threshold=0.58,
        risk_recovery_threshold=0.36,
        risk_warning_threshold=0.45,
        momentum_risk_weight=0.09,
        ema_alpha_risk=0.12,
    )


def _run_simulate(args: argparse.Namespace) -> None:
    cfg = _default_cfg()
    cfg = _apply_global_sense(cfg, args)

    result = run_simulation(
        n_steps=args.steps,
        levels=args.levels,
        seed=args.seed,
        stress_test=args.stress_test,
        cfg=cfg,
    )

    if not args.no_plot:
        _safe_plot(result)

    _safe_print_summary(result)


def _run_adversarial(args: argparse.Namespace) -> None:
    if run_adversarial_scenario is None:
        raise RuntimeError("adversarial module is not available")

    result = run_adversarial_scenario(
        name=args.scenario,
        steps=args.steps,
        seed=args.seed,
    )

    if isinstance(result, dict):
        _print_kv_block(args.scenario, result)
    else:
        print(result)


def _run_all_scenarios_cmd(args: argparse.Namespace) -> None:
    if run_all_scenarios is None:
        raise RuntimeError("adversarial module is not available")

    result = run_all_scenarios(
        steps=args.steps,
        seed=args.seed,
    )

    if isinstance(result, dict):
        for scenario_name, summary in result.items():
            if isinstance(summary, dict):
                _print_kv_block(str(scenario_name), summary)
            else:
                print(f"\n== {scenario_name} ==")
                print(summary)
    else:
        print(result)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        args.command = "simulate"
        if not hasattr(args, "steps"):
            args.steps = 1400
        if not hasattr(args, "levels"):
            args.levels = 3
        if not hasattr(args, "seed"):
            args.seed = 42
        if not hasattr(args, "stress_test"):
            args.stress_test = False
        if not hasattr(args, "no_plot"):
            args.no_plot = False
        if not hasattr(args, "global_sense"):
            args.global_sense = False
        if not hasattr(args, "gs_debug"):
            args.gs_debug = False
        if not hasattr(args, "gs_force_refresh"):
            args.gs_force_refresh = False

    if args.command == "simulate":
        _run_simulate(args)
    elif args.command == "adversarial":
        _run_adversarial(args)
    elif args.command == "all-scenarios":
        _run_all_scenarios_cmd(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
