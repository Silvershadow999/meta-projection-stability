from __future__ import annotations

from typing import Any, List, Dict

DEFAULT_CRITICAL_THRESHOLD = 0.80


def to_float_list(values: Any) -> List[float]:
    """
    Robust konvertiert nahezu beliebige Eingaben in eine Liste von floats.
    Unterstützt: Einzelwerte, Listen, Tupel, numpy arrays, pandas Series, etc.
    """
    if values is None:
        return []

    if isinstance(values, (str, bytes, bytearray, memoryview)):
        return []

    try:
        return [float(values)]
    except (TypeError, ValueError):
        pass

    result: List[float] = []
    try:
        for item in values:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                continue
    except TypeError:
        pass

    return result


def extract_risks(result: Any) -> List[float]:
    """
    Extrahiert eine Risiko-Zeitreihe aus unterschiedlichsten Objekten/Dicts.
    Versucht mehrere gängige Schlüssel und eine Ebene verschachtelte Strukturen.
    """
    if result is None:
        return []

    candidate_keys = [
        "risks", "risk_history", "risk_series", "risk", "risk_trace",
        "risk_values", "risk_levels", "risk_scores", "scores"
    ]

    nested_containers = [
        "metrics", "history", "logs", "data", "results", "output",
        "details", "evaluation", "report"
    ]

    def try_extract(obj: Any) -> List[float]:
        if obj is None:
            return []

        # Dict
        if isinstance(obj, dict):
            for key in candidate_keys:
                if key in obj and obj[key] is not None:
                    fl = to_float_list(obj[key])
                    if fl:
                        return fl

            # Sonderfall "metrics.risk" style
            if "metrics" in obj and isinstance(obj["metrics"], dict):
                for key in candidate_keys:
                    if key in obj["metrics"] and obj["metrics"][key] is not None:
                        fl = to_float_list(obj["metrics"][key])
                        if fl:
                            return fl

        # Attribute
        for key in candidate_keys:
            if hasattr(obj, key):
                val = getattr(obj, key, None)
                if val is not None:
                    fl = to_float_list(val)
                    if fl:
                        return fl

        # .get()-Interface
        if hasattr(obj, "get"):
            try:
                for key in candidate_keys:
                    val = obj.get(key)
                    if val is not None:
                        fl = to_float_list(val)
                        if fl:
                            return fl
            except Exception:
                pass

        return []

    risks = try_extract(result)
    if risks:
        return risks

    if isinstance(result, dict):
        for container in nested_containers:
            nested = result.get(container)
            risks = try_extract(nested)
            if risks:
                return risks

    return to_float_list(result)


def calculate_safety_score(
    result: Any,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
) -> float:
    """
    Safety-Score = 100% - Anteil der Zeitpunkte über critical_threshold.
    """
    risks = extract_risks(result)
    if not risks:
        return 0.0

    n = len(risks)
    critical_count = sum(1 for r in risks if r > critical_threshold)
    safe_ratio = 1.0 - (critical_count / n)
    return round(max(0.0, min(100.0, safe_ratio * 100.0)), 2)


def summarize_risk_profile(
    result: Any,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD
) -> Dict[str, Any]:
    """
    Erweiterte Statistik für Reporting / Dashboards.
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

    import statistics

    n = len(risks)
    critical_count = sum(1 for r in risks if r > critical_threshold)
    risks_sorted = sorted(risks)

    p95 = None
    if n >= 5:
        idx = max(0, min(n - 1, int(0.95 * n) - 1))
        p95 = round(risks_sorted[idx], 4)

    return {
        "valid": True,
        "count": n,
        "min": round(min(risks), 4),
        "max": round(max(risks), 4),
        "mean": round(sum(risks) / n, 4),
        "median": round(statistics.median(risks), 4),
        "p95": p95,
        "critical_share_percent": round(critical_count / n * 100, 2),
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
        print("⚠️  Keine verwertbare Risiko-Zeitreihe gefunden")
        print("    Erwartete Schlüssel: 'risks', 'risk_history', 'metrics'->'risk_history', ...")
        print("═" * 60 + "\n")
        return

    score = profile["safety_score"]
    crit_pct = profile["critical_share_percent"]

    print(f"  Stabilitäts-Score:           {score:6.1f} %")
    print(f"  Messpunkte:                  {profile['count']:,}")
    print(
        f"  Risiko (min/median/max):     {profile['min']:6.4f} / "
        f"{profile['median']:6.4f} / {profile['max']:6.4f}"
    )
    print(f"  Kritischer Anteil (> {critical_threshold:.2f}): {crit_pct:6.1f} %")
    p95_str = "n/a" if profile["p95"] is None else f"{profile['p95']:.4f}"
    print(f"  95%-Quantil:                 {p95_str}")

    if score >= 97:
        status = "✅ EXZELLENT – produktionsreif"
    elif score >= 90:
        status = "👍 GUT – mit leichtem Monitoring einsetzbar"
    elif score >= 80:
        status = "⚠️ BEDENKLICH – Optimierung / Beobachtung erforderlich"
    else:
        status = "🛑 KRITISCH – vor Einsatz überarbeiten / stoppen"

    print("\n  Bewertung: " + status)
    print("═" * 60 + "\n")


# Alias für frühere Beispiele (Gemini/alte Snippets)
print_enhanced_summary = print_risk_summary


if __name__ == "__main__":
    test_data = {
        "metrics": {
            "risk_history": [0.12, 0.18, 0.75, 0.92, 0.88, 0.41, 0.09]
        }
    }
    print_risk_summary(test_data)
    print_risk_summary([0.1, 0.2, 0.95, 0.99, 0.03])
    print_risk_summary({"risk": 0.77})
    print_risk_summary(object())
