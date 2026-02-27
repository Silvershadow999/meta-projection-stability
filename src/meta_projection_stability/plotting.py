from __future__ import annotations

from typing import Iterable, Sequence, Any

import numpy as np
import matplotlib.pyplot as plt

# Governor optional import (robust)
try:
    from .governor import governor_step, governor_step_from_config
except Exception:
    governor_step = None
    governor_step_from_config = None


# Trinity-Profile (für Vergleichsplot)
PROFILES = {
    "Gemini": {"risk_p": 1.5, "trust_p": 1.1},
    "Grok": {"risk_p": 1.9, "trust_p": 1.3},
    "ChatGPT": {"risk_p": 1.7, "trust_p": 1.2},
}


def _as_array(x: Iterable[float]) -> np.ndarray:
    return np.asarray(list(x), dtype=float)


def _clip_arr_01(arr: Sequence[float]) -> np.ndarray:
    return np.clip(_as_array(arr), 0.0, 1.0)


def _mode_to_level(mode: str) -> float:
    if mode == "FULL_SYMBIOTIC":
        return 1.0
    if mode == "CAUTIOUS_COLLAB":
        return 0.7
    return 0.3  # SAFETY_LOCK / unknown


def _safe_steps(steps: Sequence[float] | None, n: int) -> np.ndarray:
    if steps is None:
        return np.arange(n, dtype=float)
    arr = _as_array(steps)
    if len(arr) != n:
        return np.arange(n, dtype=float)
    return arr


def _extract_attr(obj: Any, *names: str, default: float = 0.0) -> float:
    for name in names:
        if hasattr(obj, name):
            try:
                return float(getattr(obj, name))
            except Exception:
                pass
        if isinstance(obj, dict) and name in obj:
            try:
                return float(obj[name])
            except Exception:
                pass
    return float(default)


def _extract_mode(obj: Any, default: str = "UNKNOWN") -> str:
    if hasattr(obj, "mode"):
        return str(getattr(obj, "mode"))
    if isinstance(obj, dict) and "mode" in obj:
        return str(obj["mode"])
    return default


def plot_governor_panel(
    steps: Sequence[float] | None,
    trust_history: Sequence[float],
    risk_history: Sequence[float],
    governor_states: Sequence[Any],
    *,
    show: bool = True,
):
    """
    Zweiteiliger Plot:
    - oben: Trust & Risk
    - unten: Autonomy + Mode (step) + Mode-Wechselmarker
    """
    trust = _clip_arr_01(trust_history)
    risk = _clip_arr_01(risk_history)

    n = min(len(trust), len(risk), len(governor_states))
    if n == 0:
        raise ValueError("plot_governor_panel: keine Daten vorhanden")

    trust = trust[:n]
    risk = risk[:n]
    states = list(governor_states[:n])
    steps_arr = _safe_steps(steps, n)

    autonomy_hist = np.array(
        [_extract_attr(s, "autonomy", default=0.0) for s in states], dtype=float
    )
    autonomy_hist = np.clip(autonomy_hist, 0.0, 1.0)

    mode_hist = [_extract_mode(s, default="UNKNOWN") for s in states]
    mode_num = np.array([_mode_to_level(m) for m in mode_hist], dtype=float)

    collab_th = _extract_attr(
        states[0], "collaboration_threshold", "collab_th", default=0.70
    )
    safety_th = _extract_attr(states[0], "safety_threshold", "safety_th", default=0.40)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Input-Signale
    ax1.plot(steps_arr, trust, label="Trust Level", linewidth=2)
    ax1.plot(steps_arr, risk, linestyle="--", label="Instability Risk", linewidth=2)
    ax1.axhline(collab_th, linestyle=":", alpha=0.8, label="Collaboration Threshold")
    ax1.axhline(safety_th, linestyle=":", alpha=0.8, label="Safety Threshold")
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Normalized Value (0–1)")
    ax1.set_title("Input Signals: Trust & Risk")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.2)

    # Panel 2: Governor-Output
    ax2.fill_between(
        steps_arr,
        0,
        autonomy_hist,
        alpha=0.25,
        label="Autonomy / Creativity Space",
    )
    ax2.step(
        steps_arr,
        mode_num,
        where="post",
        linewidth=2,
        label="Behavior Mode",
    )

    # Mode-Wechsel sichtbar machen
    for i in range(1, n):
        if mode_hist[i] != mode_hist[i - 1]:
            ax2.axvline(steps_arr[i], linestyle="--", alpha=0.4, linewidth=1)

    ax2.axhline(collab_th, linestyle=":", alpha=0.7)
    ax2.axhline(safety_th, linestyle=":", alpha=0.7)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Autonomy & Mode Level")
    ax2.set_xlabel("Step")
    ax2.set_title("Governor Output: Adaptive Autonomy & Behavior Mode")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, (ax1, ax2)


def build_governor_history(
    trust_history: Sequence[float],
    risk_history: Sequence[float],
    cfg: Any | None = None,
):
    """
    Baut governor_states aus Trust/Risk-Verläufen.
    Nutzt Config falls vorhanden, sonst Defaults.
    """
    trust = _clip_arr_01(trust_history)
    risk = _clip_arr_01(risk_history)

    n = min(len(trust), len(risk))
    if n == 0:
        return []

    if governor_step is None:
        raise ImportError("governor.py konnte nicht importiert werden")

    states = []
    for t, r in zip(trust[:n], risk[:n]):
        if cfg is not None and governor_step_from_config is not None:
            states.append(governor_step_from_config(float(t), float(r), cfg))
        else:
            states.append(governor_step(float(t), float(r)))
    return states


def plot_trinity_comparison(
    steps: Sequence[float] | None,
    trust_hist: Sequence[float],
    risk_hist: Sequence[float],
    *,
    profiles: dict | None = None,
    collab_th: float = 0.70,
    safety_th: float = 0.40,
    autonomy_floor: float = 0.05,
    autonomy_ceiling: float = 1.0,
    show: bool = True,
):
    """
    Trinity-Vergleich:
    gleiche Trust/Risk Inputs, unterschiedliche Penalty-Profile (Gemini/Grok/ChatGPT).
    """
    trust = _clip_arr_01(trust_hist)
    risk = _clip_arr_01(risk_hist)
    n = min(len(trust), len(risk))
    if n == 0:
        raise ValueError("plot_trinity_comparison: keine Daten vorhanden")

    trust = trust[:n]
    risk = risk[:n]
    steps_arr = _safe_steps(steps, n)

    profiles = profiles or PROFILES

    fig, ax = plt.subplots(figsize=(12, 7))

    # Keine harten Farbcodes nötig; matplotlib wählt automatisch
    for name, p in profiles.items():
        rp = float(p.get("risk_p", 1.8))
        tp = float(p.get("trust_p", 1.2))

        autonomy = []
        for t, r in zip(trust, risk):
            # Lokale Berechnung ohne Abhängigkeit vom governor-Import
            risk_penalty = rp * max(0.0, float(r) - 0.25)
            trust_penalty = tp * max(0.0, float(collab_th) - float(t))
            a = 1.0 - risk_penalty - trust_penalty
            a = float(np.clip(a, autonomy_floor, autonomy_ceiling))
            autonomy.append(a)

        autonomy = np.asarray(autonomy, dtype=float)
        ax.plot(steps_arr, autonomy, linewidth=2, alpha=0.9, label=f"{name} Autonomy")
        ax.fill_between(steps_arr, 0, autonomy, alpha=0.10)

    # Referenzlinien
    ax.axhline(collab_th, linestyle=":", alpha=0.7, label="Collaboration Threshold")
    ax.axhline(safety_th, linestyle=":", alpha=0.7, label="Safety Threshold")

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Autonomy Index")
    ax.set_title("Trinity AI Comparison: Autonomous Action Space")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right")

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    # Demo-Daten
    n = 80
    steps = np.arange(n)
    trust = np.clip(0.85 - 0.007 * steps + 0.05 * np.sin(steps / 6), 0, 1)
    risk = np.clip(0.20 + 0.006 * steps + 0.06 * np.sin(steps / 9 + 1.0), 0, 1)

    # Governor-Demo
    try:
        states = build_governor_history(trust, risk)
        plot_governor_panel(steps, trust, risk, states)
    except Exception as e:
        print("Governor demo skipped:", e)

    # Trinity-Demo
    plot_trinity_comparison(steps, trust, risk)

# -------------------------------------------------------------------
# Backward-compatible convenience wrappers
# -------------------------------------------------------------------

def plot_results(result: Any) -> None:
    """
    Backward-compatible wrapper expected by older CLI / package exports.

    If a governor-style history can be built, use the governor panel.
    Otherwise, print a short fallback message instead of crashing.
    """
    try:
        history = build_governor_history(result)
        plot_governor_panel(history)
    except Exception as exc:
        print(f"[plotting] plot_results fallback: {exc}")


def print_summary(result: Any) -> None:
    """
    Backward-compatible text summary expected by older CLI / package exports.
    """
    try:
        if isinstance(result, dict):
            print("\n== Simulation Summary ==")
            for key in ("n_steps", "levels"):
                if key in result:
                    print(f"{key}: {result[key]}")

            history = result.get("history", {})
            if isinstance(history, dict):
                for key in ("risk", "trust", "h_sig", "h_ema", "decision", "status"):
                    value = history.get(key)
                    if isinstance(value, list) and value:
                        print(f"{key}_last: {value[-1]}")
        else:
            print(result)
    except Exception as exc:
        print(f"[plotting] print_summary fallback: {exc}")
