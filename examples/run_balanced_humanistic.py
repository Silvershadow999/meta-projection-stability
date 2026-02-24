from meta_projection_stability import (
    MetaProjectionStabilityConfig,
    run_simulation,
    plot_results,
    print_summary,
)

# NEU:
from meta_projection_stability.analytics import print_enhanced_summary


def main():
    cfg = MetaProjectionStabilityConfig(
        human_sig_max=1.10,
        trust_gain=0.018,
        trust_decay=0.004,
        trust_floor=0.38,
        risk_warning_threshold=0.45,
        risk_critical_threshold=0.58,
        risk_recovery_threshold=0.36,
        momentum_alert_threshold=0.055,
        momentum_risk_weight=0.09,
        cooldown_steps_after_reset=14,
        nominal_recovery_boost_factor=1.22,
    )

    result = run_simulation(n_steps=1400, levels=3, seed=42, stress_test=True, cfg=cfg)

    # NEU: verständlicher Sicherheitsbericht
    print_enhanced_summary(result)

    plot_results(result)
    print_summary(result)


if __name__ == "__main__":
    main()
