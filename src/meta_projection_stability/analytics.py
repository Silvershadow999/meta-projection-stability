from __future__ import annotations

from typing import Any, List, Dict
import math

DEFAULT_CRITICAL_THRESHOLD = 0.80


def to_float_list(values: Any) -> List[float]:
    """
    Robust konvertiert nahezu beliebige Eingaben in eine Liste von floats.
    Unterstützt u. a.:
    - Einzelwerte (int/float/np scalar)
    - Listen / Tupel
    - numpy arrays / pandas Series (iterierbar)
    """
    if values is None:
        return []

    # Strings/Bytes nicht als Sequenz aus Zeichen interpretieren
    if isinstance(values, (str, bytes, bytearray, memoryview)):
        return []

    # Einzelwert-Fall (schneller Pfad)
    try:
        return [float(values)]
    except (TypeError, ValueError):
        pass

    # Iterierbarer Fall
    out: List[float] = []
    try:
        for item in values:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
    except TypeError:
        # Nicht iterierbar
        return []

    return out


def _get_by_dot_path(obj: Any, path: str) -> Any:
    """
    Holt verschachtelte Werte per Dot-Path, z. B. 'metrics.risk_history'.
    Unterstützt dicts und attributbasierte Objekte.
    """
    current = obj
    for part in path.split("."):
        if current is None:
            return None

        if isinstance(current, dict):
            current = current.get(part)
            continue

        # Attribut-Zugriff
        if hasattr(current, part):
            current = getattr(current, part, None)
            continue

        # Generisches .get()-Interface (falls vorhanden)
        if hasattr(current, "get"):
            try:
                current = current.get(part)
                continue
            except Exception:
                return None

        return None
    return current


def extract_risks(result: Any) -> List[float]:
    """
    Extrahiert eine Risiko-Zeitreihe aus unterschiedlichen Strukturen.
    Durchsucht:
    - direkte Schlüssel/Attribute (z. B. risks, risk_history, risk)
    - Dot-Paths (z. B. metrics.risk, metrics.risk_history)
    - typische verschachtelte Container (metrics, history, logs, ...)
    - Fallback: direkte Sequenz
    """
    if result is None:
        return []

    # Häufige direkte Keys / Attribute
    candidate_keys = [
        "risks",
        "risk_history",
        "risk_series",
        "risk",
        "risk_trace",
        "risk_values",
        "risk_levels",
        "risk_scores",
        "scores",
    ]

    # Dot-Paths (werden aktiv aufgelöst)
    candidate_dot_paths = [
        "metrics.risk",
        "metrics.risk_history",
        "metrics.risks",
        "history.risk",
        "history.risk_history",
        "logs.risk",
        "data.risk",
        "results.risk",
        "output.risk",
    ]

    nested_containers = [
        "metrics",
        "history",
        "logs",
        "data",
        "results",
        "output",
        "details",
        "evaluation",
        "report",
    ]

    def try_extract(obj: Any) -> List[float]:
        if obj is None:
            return []

        # Dict-Keys direkt
        if isinstance(obj, dict):
            for key in candidate_keys:
                if key in obj:
                    fl = to_float_list(obj.get(key))
                    if fl:
                        return fl

        # Attribute
        for key in candidate_keys:
            if hasattr(obj, key):
                fl = to_float_list(getattr(obj, key, None))
                if fl:
                    return fl

        # .get()-Interface
        if hasattr(obj, "get"):
            try:
                for key in candidate_keys:
                    fl = to_float_list(obj.get(key))
                    if fl:
                        return fl
            except Exception:
                pass

        return []

    # 1) Direkte Extraktion
    risks = try_extract(result)
    if risks:
        return risks

    # 2) Dot-Path Extraktion
    for path in candidate_dot_paths:
        val = _get_by_dot_path(result, path)
        fl = to_float_list(val)
        if fl:
            return fl

    # 3) Eine Ebene verschachtelt
    if isinstance(result, dict):
        for container in nested_containers:
            nested = result.get(container)
            if nested is not None:
                risks = try_extract(nested)
                if risks:
                    return risks

    # 4) Fallback: vielleicht ist result selbst eine Sequenz mit Zahlen
    return to_float_list(result)


def calculate_safety_score(
    result: Any,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
) -> float:
    """
    Berechnet einen Safety-Score:
    100% - Anteil der Zeitpunkte, an denen Risiko > critical_threshold war.
    """
    risks = extract_risks(result)
    if not risks:
        return 0.0

    n = len(risks)
    critical_count = sum(1 for r in risks if r > critical_threshold)
    safe_ratio = 1.0 - (critical_count / n)
    score = safe_ratio * 100.0
    return round(max(0.0, min(100.0, score)), 2)


def summarize_risk_profile(
    result: Any,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD
) -> Dict[str, Any]:
    """
    Erweiterte Statistik für Reporting / Dashboards / Präsentationen.
    """
    risks = extract_risks(result)
    if not risks:
        return {
            "valid": False,
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
            "critical_share_percent": None,
            "safety_score": 0.0,
        }

    import statistics  # lazy import

    n = len(risks)
    sorted_risks = sorted(risks)
    critical_count = sum(1 for r in risks if r > critical_threshold)

    # Robuste p95-Approximation ohne NumPy
    if n >= 1:
        idx = max(0, min(n - 1, math.ceil(n * 0.95) - 1))
        p95 = sorted_risks[idx]
    else:
        p95 = None

    return {
        "valid": True,
        "count": n,
        "min": round(min(risks), 4),
        "max": round(max(risks), 4),
        "mean": round(sum(risks) / n, 4),
        "median": round(statistics.median(risks), 4),
        "p95": round(p95, 4) if p95 is not None else None,
        "critical_share_percent": round((critical_count / n) * 100.0, 2),
        "safety_score": calculate_safety_score(result, critical_threshold),
    }


def print_risk_summary(
    result: Any,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
    title: str = "SICHERHEITS- & STABILITÄTS-BEWERTUNG",
) -> None:
    """
    Menschlich lesbare, management-taugliche Zusammenfassung.
    """
    profile = summarize_risk_profile(result, critical_threshold)

    print("\n" + "═" * 60)
    print(f"🛡️  {title.upper()}")
    print("═" * 60)

    if not profile["valid"]:
        print("⚠️  Keine verwertbare Risiko-Zeitreihe gefunden.")
        print("    Erwartete Schlüssel (Beispiele):")
        print("    'risks', 'risk_history', 'risk', 'metrics.risk_history', ...")
        print("═" * 60 + "\n")
        return

    score = profile["safety_score"]
    crit_pct = profile["critical_share_percent"]
    p95_str = f"{profile['p95']:6.4f}" if profile["p95"] is not None else "  n/a "

    print(f"  Stabilitäts-Score:           {score:6.1f} %")
    print(f"  Messpunkte:                  {profile['count']:,}")
    print(
        f"  Risiko (min / Median / max): "
        f"{profile['min']:6.4f} / {profile['median']:6.4f} / {profile['max']:6.4f}"
    )
    print(f"  Kritischer Anteil (> {critical_threshold:.2f}): {crit_pct:6.1f} %")
    print(f"  95%-Quantil:                 {p95_str}")

    if score >= 97:
        status = "✅  EXZELLENT – produktionsreif"
    elif score >= 90:
        status = "👍  GUT – mit leichtem Monitoring einsetzbar"
    elif score >= 80:
        status = "⚠️  BEDENKLICH – Optimierung / Beobachtung erforderlich"
    else:
        status = "🛑  KRITISCH – vor Einsatz überarbeiten / stoppen"

    print("\n  Bewertung:  " + status)
    print("═" * 60 + "\n")


# Alias für Kompatibilität zu älteren Imports / Beispielen
def print_enhanced_summary(
    result: Any,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD
) -> None:
    """
    Backward-compatible Alias.
    """
    print_risk_summary(result, critical_threshold=critical_threshold)


# ────────────────────────────────────────────────
# Beispiel-Nutzung / Selbsttest
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Verschachtelte Struktur
    test_data = {
        "metadata": {"model": "meta-llama-3-70b"},
        "metrics": {
            "risk_history": [0.12, 0.18, 0.75, 0.92, 0.88, 0.41, 0.09],
            "other": "ignored",
        },
    }

    # 2) Direkte Liste
    test_list = [0.1, 0.2, 0.95, 0.99, 0.03]

    # 3) Einzelwert
    test_scalar = {"risk": 0.77}

    # 4) Schlechter Fall
    test_bad = object()

    print_risk_summary(test_data)
    print_risk_summary(test_list)
    print_risk_summary(test_scalar)
    print_risk_summary(test_bad)
