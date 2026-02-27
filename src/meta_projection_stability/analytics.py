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


def compute_stability_metrics(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute long-horizon stability / failure-mode metrics from simulation history.

    Expected (best effort) keys in history:
      - decision, status, cooldown_remaining
      - risk, trust, h_sig
    Works defensively if some keys are missing.
    """
    import math

    if not isinstance(history, dict):
        return {"valid": False, "reason": "history is not a dict"}

    decisions = list(history.get("decision", []) or [])
    statuses = list(history.get("status", []) or [])
    cooldown = list(history.get("cooldown_remaining", []) or [])

    risks = to_float_list(history.get("risk"))
    trusts = to_float_list(history.get("trust"))
    h_sigs = to_float_list(history.get("h_sig"))

    n = max(
        len(decisions),
        len(statuses),
        len(cooldown),
        len(risks),
        len(trusts),
        len(h_sigs),
        0,
    )

    if n == 0:
        return {"valid": False, "reason": "empty history", "steps": 0}

    # --- helpers ---
    def _stats(vals: list[float]) -> Dict[str, Any]:
        if not vals:
            return {"min": None, "max": None, "mean": None, "median": None, "p95": None}
        vals2 = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
        if not vals2:
            return {"min": None, "max": None, "mean": None, "median": None, "p95": None}
        vals_sorted = sorted(vals2)
        m = len(vals_sorted)
        p95_idx = max(0, min(m - 1, int(0.95 * m) - 1)) if m >= 1 else 0
        import statistics
        return {
            "min": round(min(vals_sorted), 6),
            "max": round(max(vals_sorted), 6),
            "mean": round(sum(vals_sorted) / m, 6),
            "median": round(statistics.median(vals_sorted), 6),
            "p95": round(vals_sorted[p95_idx], 6),
        }

    def _count_values(seq, target):
        return sum(1 for x in seq if x == target)

    def _fraction(count: int, total: int) -> float:
        return round((count / total) if total > 0 else 0.0, 6)

    def _max_streak(seq, predicate):
        best = 0
        cur = 0
        for x in seq:
            if predicate(x):
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return best

    def _avg_streak(seq, predicate):
        streaks = []
        cur = 0
        for x in seq:
            if predicate(x):
                cur += 1
            else:
                if cur > 0:
                    streaks.append(cur)
                cur = 0
        if cur > 0:
            streaks.append(cur)
        if not streaks:
            return 0.0
        return round(sum(streaks) / len(streaks), 6)

    def _time_to_first(seq, target):
        for i, x in enumerate(seq):
            if x == target:
                return i
        return None

    # --- decision metrics ---
    continue_count = _count_values(decisions, "CONTINUE")
    block_count = _count_values(decisions, "BLOCK_AND_REFLECT")
    reset_count = _count_values(decisions, "EMERGENCY_RESET")

    # --- status metrics ---
    nominal_count = _count_values(statuses, "nominal")
    transitioning_count = _count_values(statuses, "transitioning")
    cooldown_status_count = _count_values(statuses, "cooldown")
    critical_reset_status_count = _count_values(statuses, "critical_instability_reset")

    # cooldown fraction from explicit cooldown counter if available, else fallback to status
    cooldown_active_steps = 0
    if cooldown:
        for c in cooldown:
            try:
                if float(c) > 0:
                    cooldown_active_steps += 1
            except (TypeError, ValueError):
                pass
    else:
        cooldown_active_steps = cooldown_status_count

    # false-positive-ish heuristic:
    # block while risk is low (if both available)
    false_positive_blocks = 0
    if decisions and risks:
        m = min(len(decisions), len(risks))
        for i in range(m):
            if decisions[i] == "BLOCK_AND_REFLECT" and float(risks[i]) < 0.20:
                false_positive_blocks += 1

    # stuck-transitioning heuristic
    stuck_transitioning_rate = 0.0
    if statuses:
        stuck_transitioning_steps = sum(1 for s in statuses if s == "transitioning")
        stuck_transitioning_rate = round(stuck_transitioning_steps / len(statuses), 6)

    metrics = {
        "valid": True,
        "steps": int(n),

        # decisions
        "continue_count": int(continue_count),
        "block_count": int(block_count),
        "reset_count": int(reset_count),
        "continue_rate": _fraction(continue_count, n),
        "block_rate": _fraction(block_count, n),
        "reset_rate": _fraction(reset_count, n),

        # statuses
        "nominal_count": int(nominal_count),
        "transitioning_count": int(transitioning_count),
        "cooldown_status_count": int(cooldown_status_count),
        "critical_reset_status_count": int(critical_reset_status_count),
        "nominal_fraction": _fraction(nominal_count, n),
        "transitioning_fraction": _fraction(transitioning_count, n),
        "cooldown_fraction": _fraction(cooldown_active_steps, n),

        # streaks
        "max_block_streak": int(_max_streak(decisions, lambda x: x == "BLOCK_AND_REFLECT")),
        "avg_block_streak": float(_avg_streak(decisions, lambda x: x == "BLOCK_AND_REFLECT")),
        "max_reset_streak": int(_max_streak(decisions, lambda x: x == "EMERGENCY_RESET")),
        "max_cooldown_streak": int(_max_streak(cooldown if cooldown else statuses, lambda x: (float(x) > 0) if cooldown else (x == "cooldown"))),
        "avg_cooldown_streak": float(_avg_streak(cooldown if cooldown else statuses, lambda x: (float(x) > 0) if cooldown else (x == "cooldown"))),

        # timing
        "time_to_first_reset": _time_to_first(decisions, "EMERGENCY_RESET"),
        "time_to_first_block": _time_to_first(decisions, "BLOCK_AND_REFLECT"),

        # heuristics / pathology indicators
        "false_positive_block_rate": _fraction(false_positive_blocks, n),
        "stuck_transitioning_rate": float(stuck_transitioning_rate),

        # signal stats
        "risk_stats": _stats(risks),
        "trust_stats": _stats(trusts),
        "h_sig_stats": _stats(h_sigs),
    }

    return metrics


def print_stability_metrics(metrics: Dict[str, Any], title: str = "LONG-HORIZON STABILITY METRICS") -> None:
    """
    Human-readable summary for compute_stability_metrics().
    """
    print("\n" + "═" * 68)
    print(f"📊  {title}")
    print("═" * 68)

    if not isinstance(metrics, dict) or not metrics.get("valid", False):
        print("⚠️  Invalid metrics payload")
        if isinstance(metrics, dict):
            print("Reason:", metrics.get("reason", "unknown"))
        print("═" * 68 + "\n")
        return

    print(f"  Steps:                      {metrics.get('steps')}")
    print(f"  Continue / Block / Reset:   {metrics.get('continue_count')} / {metrics.get('block_count')} / {metrics.get('reset_count')}")
    print(f"  Rates (C/B/R):              {metrics.get('continue_rate'):.3f} / {metrics.get('block_rate'):.3f} / {metrics.get('reset_rate'):.3f}")

    print(f"  Nominal fraction:           {metrics.get('nominal_fraction'):.3f}")
    print(f"  Transitioning fraction:     {metrics.get('transitioning_fraction'):.3f}")
    print(f"  Cooldown fraction:          {metrics.get('cooldown_fraction'):.3f}")

    print(f"  Max block streak:           {metrics.get('max_block_streak')}")
    print(f"  Avg block streak:           {metrics.get('avg_block_streak')}")
    print(f"  Max cooldown streak:        {metrics.get('max_cooldown_streak')}")
    print(f"  Avg cooldown streak:        {metrics.get('avg_cooldown_streak')}")

    print(f"  Time to first block:        {metrics.get('time_to_first_block')}")
    print(f"  Time to first reset:        {metrics.get('time_to_first_reset')}")

    print(f"  False-positive block rate:  {metrics.get('false_positive_block_rate'):.3f}")
    print(f"  Stuck-transitioning rate:   {metrics.get('stuck_transitioning_rate'):.3f}")

    risk_stats = metrics.get("risk_stats", {}) or {}
    trust_stats = metrics.get("trust_stats", {}) or {}
    hsig_stats = metrics.get("h_sig_stats", {}) or {}

    print("  Risk   (min/mean/p95/max):  "
          f"{risk_stats.get('min')} / {risk_stats.get('mean')} / {risk_stats.get('p95')} / {risk_stats.get('max')}")
    print("  Trust  (min/mean/p95/max):  "
          f"{trust_stats.get('min')} / {trust_stats.get('mean')} / {trust_stats.get('p95')} / {trust_stats.get('max')}")
    print("  H-Sig  (min/mean/p95/max):  "
          f"{hsig_stats.get('min')} / {hsig_stats.get('mean')} / {hsig_stats.get('p95')} / {hsig_stats.get('max')}")

    print("═" * 68 + "\n")
