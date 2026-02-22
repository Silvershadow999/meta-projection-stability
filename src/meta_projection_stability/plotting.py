from __future__ import annotations

from typing import Any
import numpy as np
import matplotlib.pyplot as plt

# Governor-Import tolerant halten (falls Datei noch nicht vorhanden ist)
try:
    from .governor import GovernorState, compute_autonomy_index, TRINITY_PROFILES
except Exception:  # pragma: no cover
    GovernorState = Any  # type: ignore

    def compute_autonomy_index(
        trust: float,
        risk: float,
        collab_th: float = 0.70,
        risk_penalty_factor: float = 1.8,
        trust_penalty_factor: float = 1.2,
    ) -> float:
        trust = float(np.clip(trust, 0.0, 1.0))
        risk = float(np.clip(risk, 0.0, 1.0))
        risk_penalty = risk_penalty_factor * max(0.0, risk - 0.25)
        trust_penalty = trust_penalty_factor * max(0.0, collab_th - trust)
        return float(np.clip(1.0 - risk_penalty - trust_penalty, 0.0, 1.0))

    TRINITY_PROFILES = {
        "Gemini": {"risk_p": 1.5, "trust_p": 1.1},
        "Grok": {"risk_p": 1.9, "trust_p": 1.3},
        "ChatGPT": {"risk_p": 1.7, "trust_p": 1.2},
    }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _as_np(x, default=None):
    if x is None:
        if default is None:
            return None
        return np.asarray(default, dtype=float)
    try:
        return np.asarray(x, dtype=float)
    except Exception:
        return np.asarray(default if default is not None else [], dtype=float)


def _find_first(result: dict, *keys, default=None):
    for k in keys:
        if k in result:
            return result[k]
    return default


def _make_steps_from_length(n: int):
    return np.arange(int(n), dtype=int)


def _mode_to_numeric(mode: str) -> float:
    if mode == "FULL_SYMBIOTIC":
        return 1.0
    if mode == "CAUTIOUS_COLLAB":
        return 0.6
    if mode == "SAFETY_LOCK":
        return 0.2
    return 0.0


# ---------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------
def print_summary(result: dict):
    """
    Tolerantes Summary für CLI:
    - gibt Standardmetriken aus, wenn vorhanden
    - ignoriert fehlende Felder
    """
    if result is None:
        print("[summary] No result provided.")
        return

    if not isinstance(result, dict):
        print(f"[summary] Unsupported result type: {type(result).__name__}")
        return

    print("\n=== Simulation Summary ===")

    # Standardfelder
    candidate_keys = [
        "seed",
        "n_steps",
        "levels",
        "final_trust",
        "final_risk",
        "final_significance",
        "final_coherence",
        "mean_trust",
        "mean_risk",
        "mean_coherence",
        "mean_significance",
        "global_stress_index",
        "globalsense_enabled",
    ]
    for k in candidate_keys:
        if k in result:
            v = result[k]
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    # Governor summary (falls vorhanden)
    gov_summary = result.get("governor_summary")
    if isinstance(gov_summary, dict):
        print("\n--- Governor Summary ---")
        for k, v in gov_summary.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    # Wenn nichts gefunden wurde
    if not any(k in result for k in candidate_keys) and "governor_summary" not in result:
        print("(No recognized summary fields found; result keys:)")
        print(", ".join(sorted(result.keys())))


# ---------------------------------------------------------------------
# Generic simulation plotting
# ---------------------------------------------------------------------
def plot_results(result: dict):
    """
    Generischer Plot für Simulationsergebnisse (robust gegen unterschiedliche Key-Namen).

    Erwartet idealerweise dict mit einigen dieser Keys:
    - steps / time
    - trust_history / trust
    - risk_history / risk
    - coherence_history / coherence
    - significance_history / human_sig_history / significance
    - autonomy_history (optional)
    """
    if not isinstance(result, dict):
        raise ValueError("plot_results expects a dict result")

    # Häufige Varianten abfangen
    steps = _find_first(result, "steps", "time", "t")
    trust = _find_first(result, "trust_history", "trust", "h_trust", "trust_hist")
    risk = _find_first(result, "risk_history", "risk", "h_risk", "risk_hist")
    coherence = _find_first(result, "coherence_history", "coherence", "h_coherence", "coherence_hist")
    significance = _find_first(
        result,
        "significance_history",
        "human_sig_history",
        "human_significance_history",
        "significance",
        "h_significance",
        "significance_hist",
    )
    autonomy = _find_first(result, "autonomy_history", "autonomy", "autonomy_hist")

    # Governor-History kann Autonomy liefern
    governor_history = result.get("governor_history")
    if autonomy is None and governor_history:
        try:
            autonomy = [float(s.autonomy) for s in governor_history]
        except Exception:
            autonomy = None

    # Länge bestimmen aus verfügbaren Reihen
    series_candidates = [s for s in [trust, risk, coherence, significance, autonomy] if s is not None]
    if not series_candidates:
        raise ValueError("plot_results: no plottable time series found in result")

    n = min(len(s) for s in series_candidates)
    if n <= 0:
        raise ValueError("plot_results: found empty time series")

    if steps is None:
        steps = _make_steps_from_length(n)
    else:
        steps = np.asarray(steps)
        if len(steps) != n:
            steps = steps[:n]

    trust = _as_np(trust, default=np.full(n, np.nan))[:n]
    risk = _as_np(risk, default=np.full(n, np.nan))[:n]
    coherence = _as_np(coherence, default=np.full(n, np.nan))[:n]
    significance = _as_np(significance, default=np.full(n, np.nan))[:n]
    autonomy = _as_np(autonomy, default=np.full(n, np.nan))[:n]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Trust / Risk
    ax = axes[0]
    if not np.isnan(trust).all():
        ax.plot(steps, trust, lw=2, color="#1f77b4", label="Trust")
    if not np.isnan(risk).all():
        ax.plot(steps, risk, lw=2, ls="--", color="crimson", label="Risk")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Value (0–1)")
    ax.set_title("Trust / Risk")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    # Panel 2: Coherence / Significance
    ax = axes[1]
    if not np.isnan(coherence).all():
        ax.plot(steps, coherence, lw=2, color="purple", label="Coherence")
    if not np.isnan(significance).all():
        ax.plot(steps, significance, lw=2, color="teal", label="Human Significance")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Value (0–1)")
    ax.set_title("Coherence / Human Significance")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    # Panel 3: Autonomy (if available)
    ax = axes[2]
    if not np.isnan(autonomy).all():
        ax.fill_between(steps, 0, autonomy, color="green", alpha=0.25, label="Autonomy")
        ax.plot(steps, autonomy, color="darkgreen", lw=2)
        ax.axhline(0.70, color="orange", ls=":", alpha=0.7, label="Collaboration Threshold")
        ax.axhline(0.40, color="red", ls=":", alpha=0.7, label="Safety Threshold")
        ax.set_title("Autonomy / Creativity Space")
        ax.legend(loc="upper right")
    else:
        ax.text(
            0.5, 0.5,
            "No autonomy history found",
            transform=ax.transAxes,
            ha="center", va="center"
        )
        ax.set_title("Autonomy / Creativity Space")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Value (0–1)")
    ax.set_xlabel("Step")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Governor panel (Trust/Risk + Autonomy/Mode)
# ---------------------------------------------------------------------
def plot_governor_panel(steps, trust_history, risk_history, governor_states):
    """
    Zwei Panels:
    1) Inputs (Trust / Risk)
    2) Governor Output (Autonomy + Mode) inkl. Mode-Switch-Markern
    """
    if len(steps) == 0:
        raise ValueError("steps is empty")
    if not (len(steps) == len(trust_history) == len(risk_history) == len(governor_states)):
        raise ValueError("steps, trust_history, risk_history, governor_states must have equal length")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Farben konsistent
    TRUST_COLOR = "#1f77b4"   # blau
    RISK_COLOR = "crimson"
    AUTO_COLOR = "green"
    MODE_COLOR = "darkgreen"

    trust_history = np.asarray(trust_history, dtype=float)
    risk_history = np.asarray(risk_history, dtype=float)

    # --- Panel 1: Input-Signale ---
    ax1.plot(steps, trust_history, color=TRUST_COLOR, lw=2, label="Trust")
    ax1.plot(steps, risk_history, color=RISK_COLOR, ls="--", lw=2, label="Instability Risk")

    ax1.axhline(0.70, color="orange", ls=":", alpha=0.8, label="Collaboration Th")
    ax1.axhline(0.40, color="red", ls=":", alpha=0.8, label="Safety Th")

    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Normalized Value (0–1)")
    ax1.set_title("Input Signals: Trust & Risk")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.2)

    # --- Panel 2: Governor Output ---
    autonomy_hist = [float(getattr(s, "autonomy", np.nan)) for s in governor_states]
    mode_hist = [str(getattr(s, "mode", "UNKNOWN")) for s in governor_states]
    mode_num = [_mode_to_numeric(m) for m in mode_hist]

    ax2.fill_between(
        steps, 0, autonomy_hist,
        color=AUTO_COLOR, alpha=0.25,
        label="Autonomy / Creativity Space"
    )
    ax2.step(
        steps, mode_num, where="post",
        color=MODE_COLOR, lw=2,
        label="Behavior Mode"
    )

    # Vertikale Linien bei Mode-Wechsel
    for i in range(1, len(governor_states)):
        if mode_hist[i] != mode_hist[i - 1]:
            ax2.axvline(steps[i], color="gray", ls="--", alpha=0.5, lw=1)

    # Schwellen aus State (falls vorhanden), sonst Defaults
    if governor_states:
        collab_th = float(getattr(governor_states[0], "collaboration_threshold", 0.70))
        safety_th = float(getattr(governor_states[0], "safety_threshold", 0.40))
    else:
        collab_th, safety_th = 0.70, 0.40

    ax2.axhline(collab_th, color="orange", ls=":", alpha=0.7, label="Collab Threshold")
    ax2.axhline(safety_th, color="red", ls=":", alpha=0.7, label="Safety Threshold")

    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Autonomy & Mode Level")
    ax2.set_title("Governor Output: Adaptive Autonomy & Behavior Mode")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Trinity comparison plot
# ---------------------------------------------------------------------
def plot_trinity_comparison(
    steps,
    trust_hist,
    risk_hist,
    collab_th: float = 0.70,
    safety_th: float = 0.40,
    profile_colors: dict | None = None,
):
    """
    Trinity-Vergleich:
    Vergleich der Autonomy-Kurven mehrerer KI-Profile (Gemini / Grok / ChatGPT)
    auf denselben Trust-/Risk-Eingängen.
    """
    if len(steps) == 0:
        raise ValueError("steps is empty")
    if not (len(steps) == len(trust_hist) == len(risk_hist)):
        raise ValueError("steps, trust_hist, risk_hist must have equal length")

    trust_hist = np.asarray(trust_hist, dtype=float)
    risk_hist = np.asarray(risk_hist, dtype=float)

    colors = profile_colors or {
        "Gemini": "cyan",
        "Grok": "orange",
        "ChatGPT": "green",
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    for name, p in TRINITY_PROFILES.items():
        risk_p = float(p.get("risk_p", 1.8))
        trust_p = float(p.get("trust_p", 1.2))

        autonomy = [
            compute_autonomy_index(
                trust=t,
                risk=r,
                collab_th=collab_th,
                risk_penalty_factor=risk_p,
                trust_penalty_factor=trust_p,
            )
            for t, r in zip(trust_hist, risk_hist)
        ]

        color = colors.get(name, None)
        ax.plot(
            steps, autonomy,
            label=f"{name} Autonomy (r={risk_p:.1f}, t={trust_p:.1f})",
            color=color, lw=2, alpha=0.9
        )
        ax.fill_between(steps, 0, autonomy, color=color, alpha=0.10)

    ax.axhline(collab_th, color="orange", ls=":", alpha=0.8, label="Collaboration Threshold")
    ax.axhline(safety_th, color="red", ls=":", alpha=0.8, label="Safety Threshold")

    ax.set_ylim(0, 1.1)
    ax.set_title("Trinity AI Comparison: Autonomous Action Space", fontsize=14)
    ax.set_xlabel("Step")
    ax.set_ylabel("Autonomy Index (0.0 – 1.0)")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
