from __future__ import annotations

from meta_projection_stability.config import MetaProjectionStabilityConfig
from meta_projection_stability.simulation import run_simulation
from meta_projection_stability.analytics import print_risk_summary


def main() -> None:
    # Nur Parameter setzen, die deine config.py sicher kennt
    cfg = MetaProjectionStabilityConfig(
        n_steps=1400,
        seed=42,
        enable_plot=False,   # erstmal aus, damit nichts wegen Plot-Funktionen crasht
        debug=True,
        verbose=True,
    )

    # WICHTIG:
    # Wir rufen run_simulation robust auf.
    # Falls deine Funktion andere Parameter erwartet, testen wir danach Schritt für Schritt.
    result = run_simulation(cfg=cfg)

    # Neue Analytics-Ausgabe (deine polished Version)
    print_risk_summary(result, critical_threshold=0.80)

    # Optional: Rohdaten-Typ anzeigen für Debug
    print(f"Result type: {type(result).__name__}")
    if isinstance(result, dict):
        print("Result keys:", list(result.keys())[:20])


if __name__ == "__main__":
    main()
