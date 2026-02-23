"""
analytics.py

Zusätzliche Auswertung / KPI-Utilities für meta-projection-stability.

Ziel:
- Simulations-Resultate verständlich auswerten
- Safety Score berechnen
- Kritische / Warn-Zeitanteile reporten
- Präsentationsfähige Zusammenfassung liefern (CLI / Demo / Employer-ready)

Erwartetes result-Format (flexibel):
{
    "risks": [...],              # bevorzugt
    # oder alternative Keys:
    "risk": [...],
    "risk_history": [...],

    # optional:
    "trust": [...],
    "human_significance": [...],
    "integrity": [...],
}
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import math
import numpy as np


# ============================================================
# Helpers
# ============================================================

def _to_float_array(values: Any) -> np.ndarray:
    """
    Konvertiert Listen / Tuples / NumPy-Arrays robust in float64-Array.
    Nicht-finite Werte (inf, -inf) bleiben zunächst erhalten; NaNs werden später behandelt.
    """
    if values is None:
        return np.asarray([], dtype=np.float64)

    if isinstance(values, np.ndarray):
        return values.astype(np.float64, copy=False)

    if isinstance(values, (list, tuple)):
        try:
            return np.asarray(values, dtype=np.float64)
        except Exception:
            cleaned = []
            for v in values:
                try:
                    cleaned.append(float(v))
                except Exception:
                    cleaned.append(np.nan)
            return np.asarray(cleaned, dtype=np.float64)

    # Fallback: einzelner Wert
    try:
        return np.asarray([float(values)], dtype=np.float64)
    except Exception:
        return np.asarray([], dtype=np.float64)


def _finite_only(arr: np.ndarray) -> np.ndarray:
    """Gibt nur finite Werte zurück."""
    arr = np.asarray(arr, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _safe_mean(arr: np.ndarray, default: float = 0.0) -> float:
    arr = _finite_only(arr)
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))


def _safe_std(arr: np.ndarray, default: float = 0.0) -> float:
    arr = _finite_only(arr)
    if arr.size == 0:
        return float(default)
    return float(np.std(arr))


def _safe_min(arr: np.ndarray, default: float = 0.0) -> float:
    arr = _finite_only(arr)
    if arr.size == 0:
        return float(default)
    return float(np.min(arr))


def _safe_max(arr: np.ndarray, default: float = 0.0) -> float:
    arr = _finite_only(arr)
    if arr.size == 0:
        return float(default)
    return float(np.max(arr))


# ============================================================
# Result extraction
# ============================================================

def extract_risks(result: Dict[str, Any]) -> np.ndarray:
    """
    Extrahiert Risikoverlauf robust aus verschiedenen möglichen Keys.

    Bevorzugte Keys:
    - "risks"
    Alternativen:
    - "risk"
    - "risk_history"
    - "risk_hist"

    Returns:
        np.ndarray (float64), ggf. leer
    """
    if not isinstance(result, dict):
        return np.asarray([], dtype=np.float64)

    for key in ("risks", "risk", "risk_history", "risk_hist"):
        if key in result:
            return _to_float_array(result.get(key))

    return np.asarray([], dtype=np.float64)


def extract_optional_series(result: Dict[str, Any], *keys: str) -> np.ndarray:
    """
    Extrahiert beliebige optionale Zeitreihen (z. B. trust, human_significance).
    """
    if not isinstance(result, dict):
        return np.asarray([], dtype=np.float64)

    for key in keys:
        if key in result:
            return _to_float_array(result.get(key))
    return np.asarray([], dtype=np.float64)


# ============================================================
# Core analytics
# ============================================================

def calculate_safety_score(
    result: Dict[str, Any],
    critical_threshold: float = 0.8,
) -> float:
    """
    Berechnet einen einfachen Safety Score in Prozent.

    Definition:
      Safety = 100 * (1 - Anteil[risk > critical_threshold])

    Hinweise:
    - Nutzt nur finite Werte.
    - Wenn keine Risiken vorhanden sind -> 0.0 (explizit konservativ).
    """
    risks = extract_risks(result)
    risks = _finite_only(risks)

    if risks.size == 0:
        return 0.0

    critical_moments = int(np.sum(risks > float(critical_threshold)))
    total_steps = int(risks.size)

    # Schritt-für-Schritt (saubere Rechnung)
    ratio = critical_moments / total_steps
    safety_percent = (1.0 - ratio) * 100.0

    # Clamp auf [0, 100]
    safety_percent = max(0.0, min(100.0, safety_percent))
    return round(float(safety_percent), 2)


def summarize_safety(
    result: Dict[str, Any],
    *,
    critical_threshold: float = 0.8,
    warn_threshold: float = 0.65,
    tail_len: Optional[int] = None,
) -> Dict[str, float]:
    """
    Liefert eine strukturierte Zusammenfassung (KPI-Dict), z. B. für CLI/JSON/Reports.

    Enthaltene KPIs:
    - safety_score_percent
    - total_steps
    - valid_steps
    - critical_count / critical_fraction
    - warn_count / warn_fraction
    - risk_mean/std/min/max
    - risk_tail_mean/std (optional; bei tail_len)
    """
    risks_raw = extract_risks(result)
    valid_mask = np.isfinite(risks_raw)
    risks = risks_raw[valid_mask]

    total_steps = int(risks_raw.size)
    valid_steps = int(risks.size)
    invalid_steps = int(total_steps - valid_steps)

    if valid_steps == 0:
        return {
            "safety_score_percent": 0.0,
            "total_steps": float(total_steps),
            "valid_steps": 0.0,
            "invalid_steps": float(invalid_steps),
            "critical_threshold": float(critical_threshold),
            "warn_threshold": float(warn_threshold),
            "critical_count": 0.0,
            "critical_fraction": 0.0,
            "warn_count": 0.0,
            "warn_fraction": 0.0,
            "risk_mean": 0.0,
            "risk_std": 0.0,
            "risk_min": 0.0,
            "risk_max": 0.0,
            "risk_tail_mean": 0.0,
            "risk_tail_std": 0.0,
        }

    # Counts
    critical_count = int(np.sum(risks > float(critical_threshold)))
    warn_count = int(np.sum(risks > float(warn_threshold)))

    # Fractions (bezogen auf valid_steps)
    critical_fraction = critical_count / valid_steps
    warn_fraction = warn_count / valid_steps

    # Safety Score
    safety_score = (1.0 - critical_fraction) * 100.0
    safety_score = max(0.0, min(100.0, safety_score))

    # Tail-Stats
    if tail_len is None or tail_len <= 0:
        tail = risks
    else:
        tail = risks[-int(tail_len):]

    summary = {
        "safety_score_percent": round(float(safety_score), 2),
        "total_steps": float(total_steps),
        "valid_steps": float(valid_steps),
        "invalid_steps": float(invalid_steps),
        "critical_threshold": float(critical_threshold),
        "warn_threshold": float(warn_threshold),
        "critical_count": float(critical_count),
        "critical_fraction": round(float(critical_fraction), 6),
        "warn_count": float(warn_count),
        "warn_fraction": round(float(warn_fraction), 6),
        "risk_mean": round(_safe_mean(risks), 6),
        "risk_std": round(_safe_std(risks), 6),
        "risk_min": round(_safe_min(risks), 6),
        "risk_max": round(_safe_max(risks), 6),
        "risk_tail_mean": round(_safe_mean(tail), 6),
        "risk_tail_std": round(_safe_std(tail), 6),
    }

    return summary


def classify_safety_status(score_percent: float) -> str:
    """
    Einfache, präsentationsfreundliche Klassifikation.
    """
    s = float(score_percent)
    if s > 95.0:
        return "EXCELLENT"
    if s > 80.0:
        return "STABLE"
    return "CRITICAL"


# ============================================================
# Human-readable reporting
# ============================================================

def print_enhanced_summary(
    result: Dict[str, Any],
    *,
    critical_threshold: float = 0.8,
    warn_threshold: float = 0.65,
    tail_len: Optional[int] = None,
    title: str = "SICHERHEITS-BERICHT",
) -> Dict[str, float]:
    """
    Druckt eine verständliche Zusammenfassung und gibt das Summary-Dict zurück.

    Rückgabe:
        summary (dict), damit du es zusätzlich speichern / weiterverarbeiten kannst.
    """
    summary = summarize_safety(
        result,
        critical_threshold=critical_threshold,
        warn_threshold=warn_threshold,
        tail_len=tail_len,
    )

    score = float(summary["safety_score_percent"])
    status = classify_safety_status(score)

    # Optional zusätzliche Reihen für Kontext
    trust = extract_optional_series(result, "trust", "trusts", "trust_history")
    human = extract_optional_series(result, "human_significance", "human", "human_history")
    integrity = extract_optional_series(result, "integrity", "integrity_barometer", "barometer")

    trust_mean = _safe_mean(trust, default=np.nan)
    human_mean = _safe_mean(human, default=np.nan)
    integrity_mean = _safe_mean(integrity, default=np.nan)

    print("\n" + "=" * 42)
    print(f"🛡️  {title}")
    print("=" * 42)

    print(f"Gesamt-Stabilität (Safety Score): {score:.2f}%")

    if status == "EXCELLENT":
        print("Status: ✅ EXZELLENT (Enterprise Ready)")
    elif status == "STABLE":
        print("Status: ⚠️ STABIL (Beobachtung empfohlen)")
    else:
        print("Status: 🛑 KRITISCH (Governance eingreifen)")

    # Basiszahlen
    total_steps = int(summary["total_steps"])
    valid_steps = int(summary["valid_steps"])
    invalid_steps = int(summary["invalid_steps"])
    print(f"Schritte (gesamt/valide/invalid): {total_steps} / {valid_steps} / {invalid_steps}")

    # Risk-KPIs
    print(
        "Risk stats: "
        f"mean={summary['risk_mean']:.4f}, "
        f"std={summary['risk_std']:.4f}, "
        f"min={summary['risk_min']:.4f}, "
        f"max={summary['risk_max']:.4f}"
    )
    print(
        f"Warn-Anteil   (> {summary['warn_threshold']:.2f}): "
        f"{100.0 * float(summary['warn_fraction']):.2f}% "
        f"({int(summary['warn_count'])} Schritte)"
    )
    print(
        f"Kritisch-Anteil (> {summary['critical_threshold']:.2f}): "
        f"{100.0 * float(summary['critical_fraction']):.2f}% "
        f"({int(summary['critical_count'])} Schritte)"
    )

    if tail_len is not None and tail_len > 0:
        print(
            f"Tail ({int(tail_len)} Schritte): "
            f"risk_mean={summary['risk_tail_mean']:.4f}, "
            f"risk_std={summary['risk_tail_std']:.4f}"
        )

    # Optionale Kontexte (nur wenn Daten vorhanden)
    if np.isfinite(trust_mean):
        print(f"Ø Trust: {trust_mean:.4f}")
    if np.isfinite(human_mean):
        print(f"Ø Human significance: {human_mean:.4f}")
    if np.isfinite(integrity_mean):
        print(f"Ø Integrity/Barometer: {integrity_mean:.4f}")

    print("=" * 42 + "\n")
    return summary


# ============================================================
# Optional: comparison helper (v2/v3 / baseline vs patched)
# ============================================================

def compare_safety_summaries(
    left_summary: Dict[str, float],
    right_summary: Dict[str, float],
    left_name: str = "A",
    right_name: str = "B",
) -> Dict[str, float]:
    """
    Vergleicht zwei Summary-Dicts (z. B. v2 vs v3) und gibt Delta-KPIs zurück.
    Positives Delta bei safety_score_percent bedeutet Verbesserung von left -> right.
    """
    keys = [
        "safety_score_percent",
        "critical_fraction",
        "warn_fraction",
        "risk_mean",
        "risk_std",
        "risk_tail_mean",
        "risk_tail_std",
    ]

    delta: Dict[str, float] = {}
    for k in keys:
        lv = float(left_summary.get(k, np.nan))
        rv = float(right_summary.get(k, np.nan))
        if np.isfinite(lv) and np.isfinite(rv):
            delta[f"{k}_delta_{left_name}_to_{right_name}"] = rv - lv

    return delta


def print_comparison_report(
    left_result: Dict[str, Any],
    right_result: Dict[str, Any],
    *,
    left_name: str = "v2",
    right_name: str = "v3",
    critical_threshold: float = 0.8,
    warn_threshold: float = 0.65,
    tail_len: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Druckt einen kurzen Vergleichsreport für zwei Runs.
    Praktisch für: "Welcher Code ist besser?" (v2 vs v3 / baseline vs patch)
    """
    left_summary = summarize_safety(
        left_result,
        critical_threshold=critical_threshold,
        warn_threshold=warn_threshold,
        tail_len=tail_len,
    )
    right_summary = summarize_safety(
        right_result,
        critical_threshold=critical_threshold,
        warn_threshold=warn_threshold,
        tail_len=tail_len,
    )

    delta = compare_safety_summaries(left_summary, right_summary, left_name, right_name)

    print("\n" + "=" * 50)
    print(f"📊 VERGLEICH: {left_name}  vs  {right_name}")
    print("=" * 50)
    print(f"{left_name} Safety Score:  {left_summary['safety_score_percent']:.2f}%")
    print(f"{right_name} Safety Score: {right_summary['safety_score_percent']:.2f}%")

    k = f"safety_score_percent_delta_{left_name}_to_{right_name}"
    if k in delta:
        print(f"Δ Safety Score ({left_name}→{right_name}): {delta[k]:+.2f} %-Punkte")

    kc = f"critical_fraction_delta_{left_name}_to_{right_name}"
    if kc in delta:
        print(f"Δ Kritisch-Anteil: {100.0 * delta[kc]:+.2f} %-Punkte")

    kw = f"warn_fraction_delta_{left_name}_to_{right_name}"
    if kw in delta:
        print(f"Δ Warn-Anteil: {100.0 * delta[kw]:+.2f} %-Punkte")

    print("=" * 50 + "\n")
    return left_summary, right_summary, delta
