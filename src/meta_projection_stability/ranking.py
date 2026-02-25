from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import statistics
import csv
from pathlib import Path


@dataclass(frozen=True)
class ScoreWeights:
    """
    Positive weights reward desired behavior.
    Negative weights penalize undesired behavior.
    Metrics are expected to be in roughly [0,1] except times/counts.
    """
    continue_rate: float = 1.00
    block_rate: float = -0.15              # mild penalty (blocking is not always bad)
    reset_rate: float = -1.40              # strong penalty
    cooldown_fraction: float = -0.35
    stuck_transitioning_rate: float = -0.90
    false_positive_block_rate: float = -0.75

    risk_p95: float = -0.90
    risk_max: float = -0.50

    trust_min: float = 0.55
    trust_mean: float = 0.20

    h_sig_min: float = 0.65
    h_sig_mean: float = 0.20

    # time metrics: later reset can be good
    time_to_first_reset: float = 0.15      # normalized if possible
    time_to_first_block: float = 0.03      # weak bonus if later

    # optional streak penalties
    max_reset_streak: float = -0.10
    max_cooldown_streak: float = -0.06
    max_block_streak: float = -0.03

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _to_float(x: Any) -> Optional[float]:
    try:
        fx = float(x)
        if math.isfinite(fx):
            return fx
    except (TypeError, ValueError):
        return None
    return None


def _clip01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return max(0.0, min(1.0, x))


def _normalize_time_metric(value: Optional[float], steps: Optional[float]) -> Optional[float]:
    """
    Convert absolute time-to-event into [0,1] where higher = better.
    If no event occurred (value is None), treat as max-good (=1.0).
    """
    if value is None:
        return 1.0
    if steps is None or steps <= 0:
        return None
    return _clip01(float(value) / float(steps))


def _safe_metric(row: Dict[str, Any], key: str) -> Optional[float]:
    return _to_float(row.get(key))


def score_row(
    row: Dict[str, Any],
    weights: Optional[ScoreWeights] = None,
) -> Dict[str, Any]:
    """
    Compute weighted score for a single flattened experiment row.
    Returns a copy with score fields added:
      - score_raw
      - score_components
      - score_valid
    """
    w = weights or ScoreWeights()
    out = dict(row) if isinstance(row, dict) else {"_raw": row}

    if not isinstance(row, dict):
        out["score_valid"] = False
        out["score_raw"] = None
        out["score_components"] = {}
        return out

    steps = _safe_metric(row, "steps")

    metrics: Dict[str, Optional[float]] = {
        "continue_rate": _clip01(_safe_metric(row, "continue_rate")),
        "block_rate": _clip01(_safe_metric(row, "block_rate")),
        "reset_rate": _clip01(_safe_metric(row, "reset_rate")),
        "cooldown_fraction": _clip01(_safe_metric(row, "cooldown_fraction")),
        "stuck_transitioning_rate": _clip01(_safe_metric(row, "stuck_transitioning_rate")),
        "false_positive_block_rate": _clip01(_safe_metric(row, "false_positive_block_rate")),
        "risk_p95": _clip01(_safe_metric(row, "risk_p95")),
        "risk_max": _clip01(_safe_metric(row, "risk_max")),
        "trust_min": _clip01(_safe_metric(row, "trust_min")),
        "trust_mean": _clip01(_safe_metric(row, "trust_mean")),
        "h_sig_min": _clip01(_safe_metric(row, "h_sig_min")),
        "h_sig_mean": _clip01(_safe_metric(row, "h_sig_mean")),
        "time_to_first_reset": _normalize_time_metric(_safe_metric(row, "time_to_first_reset"), steps),
        "time_to_first_block": _normalize_time_metric(_safe_metric(row, "time_to_first_block"), steps),
        "max_reset_streak": _normalize_time_metric(_safe_metric(row, "max_reset_streak"), steps),
        "max_cooldown_streak": _normalize_time_metric(_safe_metric(row, "max_cooldown_streak"), steps),
        "max_block_streak": _normalize_time_metric(_safe_metric(row, "max_block_streak"), steps),
    }

    components: Dict[str, float] = {}
    total = 0.0
    used = 0

    for k, weight in w.to_dict().items():
        v = metrics.get(k)
        if v is None:
            continue
        c = float(weight) * float(v)
        components[k] = c
        total += c
        used += 1

    out["score_valid"] = used > 0
    out["score_raw"] = round(total, 6) if used > 0 else None
    out["score_components"] = components
    out["score_metrics_used"] = used
    return out


def rank_batch_rows(
    batch_result: Dict[str, Any],
    weights: Optional[ScoreWeights] = None,
    *,
    descending: bool = True,
) -> Dict[str, Any]:
    """
    Score all flat rows in batch_result["rows"] and return ranked rows.
    """
    rows = batch_result.get("rows", []) if isinstance(batch_result, dict) else []
    scored: List[Dict[str, Any]] = [score_row(r, weights=weights) for r in rows if isinstance(r, dict)]

    valid_rows = [r for r in scored if r.get("score_valid") and r.get("score_raw") is not None]
    invalid_rows = [r for r in scored if not (r.get("score_valid") and r.get("score_raw") is not None)]

    valid_rows.sort(key=lambda r: float(r["score_raw"]), reverse=descending)

    for i, r in enumerate(valid_rows, start=1):
        r["rank"] = i

    return {
        "ranking_valid": True,
        "weights": (weights or ScoreWeights()).to_dict(),
        "ranked_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "n_ranked": len(valid_rows),
        "n_invalid": len(invalid_rows),
    }


def aggregate_ranked_rows(
    ranked_result: Dict[str, Any],
    group_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Aggregate ranked rows by keys (default: profile/system/scenario) with score stats.
    """
    if group_keys is None:
        group_keys = ("profile", "system", "scenario")

    rows = ranked_result.get("ranked_rows", []) if isinstance(ranked_result, dict) else []
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

    for r in rows:
        key = tuple(r.get(k) for k in group_keys)
        grouped.setdefault(key, []).append(r)

    agg_rows: List[Dict[str, Any]] = []
    for key, grp in grouped.items():
        scores = [float(r["score_raw"]) for r in grp if r.get("score_raw") is not None]
        if not scores:
            continue

        score_mean = sum(scores) / len(scores)
        score_std = statistics.pstdev(scores) if len(scores) > 1 else 0.0

        # optional summary metrics
        def mean_of(metric_key: str) -> Optional[float]:
            vals = []
            for r in grp:
                v = _to_float(r.get(metric_key))
                if v is not None:
                    vals.append(v)
            if not vals:
                return None
            return float(sum(vals) / len(vals))

        row = {k: v for k, v in zip(group_keys, key)}
        row.update({
            "runs": len(grp),
            "score_mean": float(score_mean),
            "score_std": float(score_std),
            "score_min": float(min(scores)),
            "score_max": float(max(scores)),
            "continue_rate_mean": mean_of("continue_rate"),
            "reset_rate_mean": mean_of("reset_rate"),
            "risk_p95_mean": mean_of("risk_p95"),
            "trust_min_mean": mean_of("trust_min"),
            "h_sig_min_mean": mean_of("h_sig_min"),
        })
        agg_rows.append(row)

    agg_rows.sort(key=lambda r: float(r.get("score_mean", -1e9)), reverse=True)

    for i, r in enumerate(agg_rows, start=1):
        r["rank"] = i

    return {
        "aggregate_valid": True,
        "group_keys": list(group_keys),
        "aggregate_rows": agg_rows,
        "n_groups": len(agg_rows),
    }


def print_top_rankings(
    ranked_result: Dict[str, Any],
    aggregate_result: Optional[Dict[str, Any]] = None,
    *,
    top_n_rows: int = 12,
    top_n_groups: int = 12,
    title: str = "BATCH RANKING",
) -> None:
    print("\n" + "═" * 128)
    print(f"🏁  {title}")
    print("═" * 128)

    ranked_rows = ranked_result.get("ranked_rows", []) if isinstance(ranked_result, dict) else []
    print(f"  Ranked runs:   {len(ranked_rows)}")
    print(f"  Invalid rows:  {ranked_result.get('n_invalid', 0) if isinstance(ranked_result, dict) else 'n/a'}")

    if ranked_rows:
        print("\nTop individual runs")
        print("-" * 128)
        print(
            f"{'Rank':>4} {'Profile':<20} {'System':<16} {'Scenario':<20} {'Seed':>4} "
            f"{'Score':>9} {'C-rate':>8} {'R-rate':>8} {'Risk p95':>9} {'Trust min':>10} {'Hsig min':>10}"
        )
        print("-" * 128)

        for r in ranked_rows[:max(1, int(top_n_rows))]:
            def fmt(x, nd=3):
                if x is None:
                    return "n/a"
                try:
                    return f"{float(x):.{nd}f}"
                except Exception:
                    return str(x)

            print(
                f"{int(r.get('rank', 0)):>4} "
                f"{str(r.get('profile', 'n/a')):<20} "
                f"{str(r.get('system', 'n/a')):<16} "
                f"{str(r.get('scenario', 'n/a')):<20} "
                f"{str(r.get('seed', 'n/a')):>4} "
                f"{fmt(r.get('score_raw'), 4):>9} "
                f"{fmt(r.get('continue_rate')):>8} "
                f"{fmt(r.get('reset_rate')):>8} "
                f"{fmt(r.get('risk_p95')):>9} "
                f"{fmt(r.get('trust_min')):>10} "
                f"{fmt(r.get('h_sig_min')):>10}"
            )
    else:
        print("\n⚠️  No ranked rows available")

    if aggregate_result and isinstance(aggregate_result, dict):
        agg_rows = aggregate_result.get("aggregate_rows", []) or []
        group_keys = aggregate_result.get("group_keys", []) or []

        print("\nTop aggregate groups")
        print("-" * 128)
        print(f"  Group keys: {group_keys}")
        print(
            f"{'Rank':>4} {'Group':<60} {'Runs':>4} {'Score μ±σ':>16} "
            f"{'C-rate μ':>9} {'R-rate μ':>9} {'Risk p95 μ':>11} {'Trust min μ':>12} {'Hsig min μ':>11}"
        )
        print("-" * 128)

        for r in agg_rows[:max(1, int(top_n_groups))]:
            group_str = " | ".join(str(r.get(k, "n/a")) for k in group_keys)
            score_mu = r.get("score_mean")
            score_sd = r.get("score_std")
            score_txt = "n/a"
            if score_mu is not None:
                score_txt = f"{float(score_mu):.4f}±{float(score_sd or 0.0):.4f}"

            def fmt(x, nd=3):
                if x is None:
                    return "n/a"
                try:
                    return f"{float(x):.{nd}f}"
                except Exception:
                    return str(x)

            print(
                f"{int(r.get('rank', 0)):>4} "
                f"{group_str:<60.60} "
                f"{int(r.get('runs', 0)):>4} "
                f"{score_txt:>16} "
                f"{fmt(r.get('continue_rate_mean')):>9} "
                f"{fmt(r.get('reset_rate_mean')):>9} "
                f"{fmt(r.get('risk_p95_mean')):>11} "
                f"{fmt(r.get('trust_min_mean')):>12} "
                f"{fmt(r.get('h_sig_min_mean')):>11}"
            )

    print("═" * 128 + "\n")


def save_ranked_rows_csv(ranked_result: Dict[str, Any], path: str) -> str:
    rows = ranked_result.get("ranked_rows", []) if isinstance(ranked_result, dict) else []
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        p.write_text("", encoding="utf-8")
        return str(p)

    # flatten score_components into columns
    flat_rows: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        comps = rr.pop("score_components", {}) if isinstance(rr.get("score_components"), dict) else {}
        for k, v in comps.items():
            rr[f"score_comp__{k}"] = v
        flat_rows.append(rr)

    fieldnames = sorted({k for row in flat_rows for k in row.keys() if k != "score_components"})
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)

    return str(p)


def save_aggregate_ranking_csv(aggregate_result: Dict[str, Any], path: str) -> str:
    rows = aggregate_result.get("aggregate_rows", []) if isinstance(aggregate_result, dict) else []
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        p.write_text("", encoding="utf-8")
        return str(p)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return str(p)


if __name__ == "__main__":
    # tiny self-test payload
    dummy_batch = {
        "rows": [
            {"profile":"balanced","system":"main_adapter","scenario":"spike","seed":41,"steps":500,"continue_rate":0.72,"reset_rate":0.02,"risk_p95":0.61,"trust_min":0.58,"h_sig_min":0.64},
            {"profile":"protective","system":"main_adapter","scenario":"spike","seed":42,"steps":500,"continue_rate":0.61,"reset_rate":0.00,"risk_p95":0.49,"trust_min":0.66,"h_sig_min":0.71},
            {"profile":"balanced","system":"threshold_only","scenario":"spike","seed":41,"steps":500,"continue_rate":0.76,"reset_rate":0.08,"risk_p95":0.74,"trust_min":0.41,"h_sig_min":0.43},
        ]
    }
    ranked = rank_batch_rows(dummy_batch)
    agg = aggregate_ranked_rows(ranked, group_keys=("profile","system","scenario"))
    print_top_rankings(ranked, agg, top_n_rows=5, top_n_groups=5, title="RANKING SELF-TEST")
