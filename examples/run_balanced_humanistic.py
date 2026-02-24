from __future__ import annotations

# --- Core imports aus deinem Paket ---
from meta_projection_stability import (
    MetaProjectionStabilityConfig,
    run_simulation,
    plot_results,
    print_summary,
)

# --- Analytics import (mit Fallbacks für unterschiedliche Projektstrukturen) ---
try:
    # Empfohlen, wenn analytics.py unter src/meta_projection_stability/analytics.py liegt
    from meta_projection_stability.analytics import print_enhanced_summary
except Exception:
    try:
        # Falls du die Funktion so benannt hast
        from meta_projection_stability.analytics import print_risk_summary as print_enhanced_summary
    except Exception:
        try:
            # Falls analytics.py im Projektroot liegt
            from analytics import print_enhanced_summary
        except Exception:
            # Letzter Fallback: auf robusten Funktionsnamen mappen
            from analytics import print_risk_summary as print_enhanced_summary


def main() -> None:
    """
    Balanced / humanistic demo run + Standard-Plot + Standard-Summary + Enhanced Safety Summary
    """

    # HINWEIS:
    # Wir verwenden hier bewusst nur Felder, die je nach v2/v3 typischerweise existieren können.
    # Falls einzelne Parameter in deiner Config anders heißen, entferne sie einfach gezielt.
    cfg = MetaProjectionStabilityConfig(
        # Basiskonfig
        n_steps=1400,
        seed=42,
        enable_plot=True,
        debug=False,
        verbose=False,

        # Human / Trust / Risk (nur wenn in deiner Config vorhanden)
        trust_gain=0.018,
        trust_decay=0.004,
        trust_floor=0.38,

        # Falls deine Version diese Felder unterstützt:
        # (wenn nicht, einfach auskommentieren)
        # human_sig_max=1.10,
        # risk_warning_threshold=0.45,
        # risk_critical_threshold=0.58,
        # risk_recovery_threshold=0.36,
        # momentum_alert_threshold=0.055,
        # momentum_risk_weight=0.09,
        # cooldown_steps_after_reset=14,
        # nominal_recovery_boost_factor=1.22,
    )

    # Simulation laufen lassen
    # Falls deine run_simulation-Signatur anders ist, n_steps/seed/stress_test ggf. dort rausnehmen
    result = run_simulation(
        n_steps=1400,
        levels=3,
        seed=42,
        stress_test=True,
        cfg=cfg,
    )

    # Standard-Ausgaben
    plot_results(result)
    print_summary(result)

    # Enhanced / management-taugliche Sicherheits-Zusammenfassung
    # (funktionsname via Fallback auf print_risk_summary gemappt)
    print_enhanced_summary(result)


if __name__ == "__main__":
    main()
