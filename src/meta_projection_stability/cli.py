from __future__ import annotations

import argparse

from .config import MetaProjectionStabilityConfig
from .simulation import run_simulation
from .plotting import plot_results, print_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meta Projection Stability simulation CLI")
    parser.add_argument("--steps", type=int, default=1400, help="Number of simulation steps")
    parser.add_argument("--levels", type=int, default=3, help="Number of layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stress-test", action="store_true", help="Enable built-in stress window")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = MetaProjectionStabilityConfig(
        # your tuned configuration can be placed here directly if desired
        risk_critical_threshold=0.58,
        risk_recovery_threshold=0.36,
        risk_warning_threshold=0.45,
        momentum_risk_weight=0.09,
        ema_alpha_risk=0.12,
    )

    result = run_simulation(
        n_steps=args.steps,
        levels=args.levels,
        seed=args.seed,
        stress_test=args.stress_test,
        cfg=cfg,
    )

    if not args.no_plot:
        plot_results(result)

    print_summary(result)


if __name__ == "__main__":
    main()
