from __future__ import annotations

import argparse

from .config import MetaProjectionStabilityConfig
from .simulation import run_simulation
from .plotting import plot_results, print_summary
from .globalsense import GlobalSense


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Meta Projection Stability simulation CLI"
    )
    parser.add_argument("--steps", type=int, default=1400, help="Number of simulation steps")
    parser.add_argument("--levels", type=int, default=3, help="Number of layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stress-test", action="store_true", help="Enable built-in stress window")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")

    # New: optional real-world modulation via World Bank API
    parser.add_argument(
        "--global-sense",
        action="store_true",
        help="Enable real-world macro modulation (World Bank API) before simulation",
    )
    parser.add_argument(
        "--gs-debug",
        action="store_true",
        help="Print GlobalSense snapshot / stress index debug output",
    )
    parser.add_argument(
        "--gs-force-refresh",
        action="store_true",
        help="Ignore GlobalSense cache and fetch fresh values",
    )

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

    # Optional: map real-world macro conditions onto config
    if args.global_sense:
        try:
            gs = GlobalSense()
            cfg = gs.map_to_config(
                cfg,
                use_stress_index=True,
                force_refresh=args.gs_force_refresh,
                debug=args.gs_debug,
            )
        except Exception as e:
            # CLI should remain usable even if external API fails
            print(f"[CLI] GlobalSense disabled due to error: {e}")

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
