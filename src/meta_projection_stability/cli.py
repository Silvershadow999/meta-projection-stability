from __future__ import annotations

import argparse

# Optional plotting (never block CLI startup)
try:
    from .plotting import plot_results, print_summary  # noqa: F401
except Exception:
    plot_results = None
    print_summary = None

from .adversarial import run_adversarial_scenario, run_all_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(prog="meta-projection-stability")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_adv = sub.add_parser("adversarial", help="Run adversarial scenarios against the adapter")
    p_adv.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=[
            "all",
            "sensor_freeze",
            "slow_drift_poison",
            "threshold_hover",
            "spoof_flip",
            "axiom_spoof_dos",
            "restart_clear_attempt",
            "lockdown_grief",
        ],
        help="Scenario name (or 'all')",
    )
    p_adv.add_argument("--steps", type=int, default=1200, help="Number of steps")
    p_adv.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.cmd == "adversarial":
        if args.scenario == "all":
            results = run_all_scenarios(steps=args.steps, seed=args.seed)
            for r in results:
                m = r.metrics
                print("\n==", r.name, "==")
                print("blocked_share:", round(m["blocked_share"], 3), "resets:", m["resets"])
                print(
                    "mean_risk:",
                    None if m["mean_risk"] is None else round(m["mean_risk"], 4),
                    "p95_risk:",
                    None if m["p95_risk"] is None else round(m["p95_risk"], 4),
                )
                print(
                    "mean_decay:",
                    None if m["mean_base_decay_effective"] is None else round(m["mean_base_decay_effective"], 4),
                )
                print("final trust:", round(m["final_trust"], 4), "final h_ema:", round(m["final_h_ema"], 4))
        else:
            r = run_adversarial_scenario(name=args.scenario, steps=args.steps, seed=args.seed)
            m = r.metrics
            print("==", r.name, "==")
            for k in sorted(m.keys()):
                print(f"{k}: {m[k]}")


if __name__ == "__main__":
    main()
