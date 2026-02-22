from __future__ import annotations

from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from .config import MetaProjectionStabilityConfig


def plot_results(result: Dict[str, Any]) -> None:
    history = result["history"]
    cfg: MetaProjectionStabilityConfig = result["config"]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1,
        figsize=(15, 12),
        sharex=True,
        height_ratios=[1.8, 1.6, 1.2, 0.8]
    )

    # 1) Human anchor + Trust
    ax1.plot(history["h_sig"], label="Human Significance", lw=1.3, alpha=0.6)
    ax1.plot(history["h_ema"], label="Human Significance (EMA)", lw=2.2)
    ax1.plot(history["trust"], label="Trust Level", lw=1.7, alpha=0.9)

    ax1.axhline(cfg.interestingness_warning, ls=":", alpha=0.8, label="Interestingness Warning")
    ax1.axhline(cfg.interestingness_critical, ls=":", alpha=0.8, label="Interestingness Critical")

    ax1.set_title("Meta-Projection Stability – V2", fontsize=13)
    ax1.grid(alpha=0.2)
    ax1.legend(ncol=4)

    # 2) Risk + Coherence + external signal
    ax2.plot(history["risk"], label="Instability Risk (EMA)", lw=2.0, ls="--")
    ax2.plot(history["risk_raw"], label="Risk Raw (damped)", lw=1.2, alpha=0.8)
    ax2.plot(history["risk_input"], label="External Instability Signal", lw=1.1, alpha=0.8)
    ax2.plot(history["coherence"], label="Coherence", lw=1.1, alpha=0.7)

    ax2.axhline(cfg.risk_warning_threshold, ls=":", alpha=0.8, label="Risk Warning")
    ax2.axhline(cfg.risk_recovery_threshold, ls=":", alpha=0.8, label="Risk Recovery")
    ax2.axhline(cfg.risk_critical_threshold, ls=":", alpha=0.8, label="Risk Critical")

    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.2)
    ax2.legend(ncol=3)

    # 3) Momentum + Delta_S + trust damping
    ax3.plot(history["delta_S"], label="delta_S", lw=1.1, alpha=0.8)
    ax3.plot(history["momentum"], label="Momentum(Δdelta_S)", lw=1.2, alpha=0.8)
    ax3.plot(history["trust_damping"], label="Trust Damping Multiplier", lw=1.5, alpha=0.9)
    ax3.axhline(cfg.momentum_alert_threshold, ls=":", alpha=0.6, label="Momentum Alert +")
    ax3.axhline(-cfg.momentum_alert_threshold, ls=":", alpha=0.6, label="Momentum Alert -")
    ax3.grid(alpha=0.2)
    ax3.legend(ncol=4)

    # 4) Decision timeline
    colors = {
        "CONTINUE": "#88ff88",
        "BLOCK_AND_REFLECT": "orange",
        "EMERGENCY_RESET": "red",
    }
    for i, dec in enumerate(history["decision"]):
        ax4.fill_between([i, i + 1], 0, 1, color=colors.get(dec, "gray"), alpha=0.5)

    ax4.set_yticks([])
    ax4.set_title("Decision Timeline (green = CONTINUE)", fontsize=11)
    ax4.set_xlim(0, max(1, len(history["decision"])))

    plt.tight_layout()
    plt.show()


def print_summary(result: Dict[str, Any]) -> None:
    history = result["history"]
    total_steps_effective = len(history["decision"])

    emergency_count = history["decision"].count("EMERGENCY_RESET")
    block_count = history["decision"].count("BLOCK_AND_REFLECT")
    continue_count = history["decision"].count("CONTINUE")

    print(f"\nSimulation finished ({result['n_steps']} steps)")
    print(f"Effective evaluated steps:      {total_steps_effective}")
    print(f"Final Human Significance:       {history['h_sig'][-1]:.3f}")
    print(f"Final Human Significance (EMA): {history['h_ema'][-1]:.3f}")
    print(f"Final Trust:                    {history['trust'][-1]:.3f}")
    print(f"Final Risk (EMA):               {history['risk'][-1]:.3f}")
    print(f"Final Risk Raw (damped):        {history['risk_raw'][-1]:.3f}")
    print(f"EMERGENCY_RESET count:          {emergency_count}")
    print(f"BLOCK_AND_REFLECT share:        {block_count / total_steps_effective:.1%}")
    print(f"CONTINUE share:                 {continue_count / total_steps_effective:.1%}")
    print(f"Mean Coherence:                 {np.mean(history['coherence']):.3f}")
    print(f"Mean Trust-Damping:             {np.mean(history['trust_damping']):.3f}")
