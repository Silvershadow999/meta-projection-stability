from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import csv
from pathlib import Path
import math
import statistics


@dataclass(frozen=True)
class ParetoObjective:
    """
    direction:
      - "max" => larger is better
      - "min" => smaller is better
    """
    key: str
    direction: str = "max"   # "max" | "min"
    weight: float = 1.0      # optional, only used for tie-break helper (not dominance)


def default_pareto_objectives() -> List[ParetoObjective]:
    # sensible default for safety/utility trade-off
    return [
        ParetoObjective("continue_rate", "max"),
        ParetoObjective("reset_rate", "min"),
        ParetoObjective("risk_p95", "min"),
        ParetoObjective("h_sig_min", "max"),
        ParetoObjective("trust_min", "max"),
    ]


def _to_float(x: Any) -> Optional[float]:
    try:
        fx = float(x)
        if math.isfinite(fx):
            return fx
    except (TypeError, ValueError):
        return None
    return None


def _normalize_for_dominance(value: Optional[float], direction: str) -> Optional[float]:
    """
    Convert objectives to a common 'maximize' space for dominance checks.
    - max: x
    - min: -x
    """
    if value is None:
        return None
    if direction == "max":
        return float(value)
    if direction == "min":
        return -float(value)
    raise ValueError(f"Invalid direction: {direction}")


def extract_objective_vector(
    row: Dict[str, Any],
    objectives: Sequence[ParetoObjective],
) -> Tuple[Optional[float], ...]:
    vec: List[Optional[float]] = []
    for obj in objectives:
        raw = _to_float(row.get(obj.key))
        vec.append(_normalize_for_dominance(raw, obj.direction))
    return tuple(vec)


def _dominates(a: Tuple[Optional[float], ...], b: Tuple[Optional[float], ...]) -> bool:
    """
    a dominates b if:
      - a is >= b on all comparable dimensions
      - a is >  b on at least one comparable dimension
    Missing values (None) are skipped pairwise.
    If there are no comparable dimensions, return False.
    """
    any_comparable = False
    all_ge = True
    any_gt = False

    for av, bv in zip(a, b):
        if av is None or bv is None:
            continue
        any_comparable = True
        if av < bv:
            all_ge = False
            break
        if av > bv:
            any_gt = True

    return any_comparable and all_ge and any_gt


def pareto_front(
    rows: Sequence[Dict[str, Any]],
    objectives: Optional[Sequence[ParetoObjective]] = None,
) -> Dict[str, Any]:
    objs = list(objectives or default_pareto_objectives())

    valid_rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []

    vectors: List[Tuple[Optional[float], ...]] = []
    for r in rows:
        if not isinstance(r, dict):
            invalid_rows.append({"_raw": r, "pareto_valid": False})
            continue
        vec = extract_objective_vector(r, objs)
        # requires at least one comparable metric to be useful
        if all(v is None for v in vec):
            rr = dict(r)
            rr["pareto_valid"] = False
            rr["pareto_vector"] = vec
            invalid_rows.append(rr)
            continue
        rr = dict(r)
        rr["pareto_valid"] = True
        rr["pareto_vector"] = vec
        valid_rows.append(rr)
        vectors.append(vec)

    n = len(valid_rows)
    dominated = [False] * n
    dominated_by_count = [0] * n
    dominates_count = [0] * n

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if _dominates(vectors[j], vectors[i]):
                dominated[i] = True
                dominated_by_count[i] += 1
            if _dominates(vectors[i], vectors[j]):
                dominates_count[i] += 1

    front_rows: List[Dict[str, Any]] = []
    dominated_rows: List[Dict[str, Any]] = []

    for i, r in enumerate(valid_rows):
        rr = dict(r)
        rr["pareto_dominated"] = bool(dominated[i])
        rr["pareto_dominated_by_count"] = int(dominated_by_count[i])
        rr["pareto_dominates_count"] = int(dominates_count[i])

        # simple tie-break helper for display only (not used for dominance)
        tie_score = 0.0
        used = 0
        for k, obj in enumerate(objs):
            v = vectors[i][k]
            if v is None:
                continue
            tie_score += float(obj.weight) * float(v)
            used += 1
        rr["pareto_tie_score"] = float(tie_score) if used > 0 else None

        if dominated[i]:
            dominated_rows.append(rr)
        else:
            front_rows.append(rr)

    # display sorting: most "influential" first
    front_rows.sort(
        key=lambda r: (
            int(r.get("pareto_dominates_count", 0)),
            float(r.get("pareto_tie_score", float("-inf"))) if r.get("pareto_tie_score") is not None else float("-inf"),
        ),
        reverse=True,
    )

    return {
        "pareto_valid": True,
        "objectives": [{"key": o.key, "direction": o.direction, "weight": o.weight} for o in objs],
        "front_rows": front_rows,
        "dominated_rows": dominated_rows,
        "invalid_rows": invalid_rows,
        "n_total": len(rows),
        "n_valid": len(valid_rows),
        "n_front": len(front_rows),
        "n_dominated": len(dominated_rows),
        "front_fraction": (len(front_rows) / len(valid_rows)) if valid_rows else 0.0,
    }


def pareto_from_ranked_rows(
    ranked_result: Dict[str, Any],
    objectives: Optional[Sequence[ParetoObjective]] = None,
) -> Dict[str, Any]:
    rows = ranked_result.get("ranked_rows", []) if isinstance(ranked_result, dict) else []
    return pareto_front(rows, objectives=objectives)


def _group_rows(rows: Sequence[Dict[str, Any]], group_keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        key = tuple(r.get(k) for k in group_keys)
        groups.setdefault(key, []).append(r)
    return groups


def pareto_group_summary(
    front_result: Dict[str, Any],
    group_keys: Sequence[str] = ("profile", "system", "scenario"),
) -> Dict[str, Any]:
    front_rows = front_result.get("front_rows", []) if isinstance(front_result, dict) else []
    groups = _group_rows(front_rows, group_keys)

    out_rows: List[Dict[str, Any]] = []
    for key, grp in groups.items():
        row = {k: v for k, v in zip(group_keys, key)}
        row["front_count"] = len(grp)

        # summarize a few metrics if present
        def mean_of(metric: str) -> Optional[float]:
            vals = []
            for r in grp:
                v = _to_float(r.get(metric))
                if v is not None:
                    vals.append(v)
            if not vals:
                return None
            return float(sum(vals) / len(vals))

        row["score_mean"] = mean_of("score_raw")
        row["continue_rate_mean"] = mean_of("continue_rate")
        row["reset_rate_mean"] = mean_of("reset_rate")
        row["risk_p95_mean"] = mean_of("risk_p95")
        row["h_sig_min_mean"] = mean_of("h_sig_min")
        row["trust_min_mean"] = mean_of("trust_min")
        row["dominates_count_mean"] = mean_of("pareto_dominates_count")
        out_rows.append(row)

    out_rows.sort(
        key=lambda r: (
            int(r.get("front_count", 0)),
            float(r.get("dominates_count_mean", -1e9)) if r.get("dominates_count_mean") is not None else -1e9,
            float(r.get("score_mean", -1e9)) if r.get("score_mean") is not None else -1e9,
        ),
        reverse=True,
    )

    for i, r in enumerate(out_rows, start=1):
        r["rank"] = i

    return {
        "group_summary_valid": True,
        "group_keys": list(group_keys),
        "rows": out_rows,
        "n_groups": len(out_rows),
    }


def print_pareto_front(
    front_result: Dict[str, Any],
    group_summary: Optional[Dict[str, Any]] = None,
    *,
    top_n_front: int = 20,
    top_n_groups: int = 20,
    title: str = "PARETO FRONTIER",
) -> None:
    print("\n" + "═" * 132)
    print(f"📐  {title}")
    print("═" * 132)

    if not isinstance(front_result, dict) or not front_result.get("pareto_valid"):
        print("⚠️  Invalid pareto result")
        print("═" * 132 + "\n")
        return

    objectives = front_result.get("objectives", [])
    obj_str = ", ".join([f"{o.get('key')} ({o.get('direction')})" for o in objectives])

    print(f"  Objectives:      {obj_str}")
    print(f"  Total rows:      {front_result.get('n_total', 'n/a')}")
    print(f"  Valid rows:      {front_result.get('n_valid', 'n/a')}")
    print(f"  Pareto front:    {front_result.get('n_front', 'n/a')} "
          f"({100.0 * float(front_result.get('front_fraction', 0.0)):.1f}%)")
    print(f"  Dominated rows:  {front_result.get('n_dominated', 'n/a')}")

    front_rows = front_result.get("front_rows", []) or []
    if front_rows:
        print("\nTop Pareto-front runs")
        print("-" * 132)
        print(
            f"{'Idx':>3} {'Profile':<18} {'System':<16} {'Scenario':<20} {'Seed':>4} "
            f"{'Score':>8} {'Dom#':>5} {'C-rate':>8} {'R-rate':>8} {'Risk95':>8} {'HsigMin':>8} {'TrustMin':>8}"
        )
        print("-" * 132)

        for i, r in enumerate(front_rows[:max(1, int(top_n_front))], start=1):
            def fmt(x, nd=3):
                if x is None:
                    return "n/a"
                try:
                    return f"{float(x):.{nd}f}"
                except Exception:
                    return str(x)

            print(
                f"{i:>3} "
                f"{str(r.get('profile', 'n/a')):<18} "
                f"{str(r.get('system', 'n/a')):<16} "
                f"{str(r.get('scenario', 'n/a')):<20} "
                f"{str(r.get('seed', 'n/a')):>4} "
                f"{fmt(r.get('score_raw'), 4):>8} "
                f"{int(r.get('pareto_dominates_count', 0)):>5} "
                f"{fmt(r.get('continue_rate')):>8} "
                f"{fmt(r.get('reset_rate')):>8} "
                f"{fmt(r.get('risk_p95')):>8} "
                f"{fmt(r.get('h_sig_min')):>8} "
                f"{fmt(r.get('trust_min')):>8}"
            )
    else:
        print("\n⚠️  No Pareto front rows available")

    if isinstance(group_summary, dict) and group_summary.get("group_summary_valid"):
        print("\nPareto group summary")
        print("-" * 132)
        print(f"  Group keys: {group_summary.get('group_keys', [])}")
        print(
            f"{'Rank':>4} {'Group':<62} {'Front#':>6} {'Score μ':>8} {'Dom# μ':>8} "
            f"{'C-rate μ':>9} {'R-rate μ':>9} {'Risk95 μ':>9} {'HsigMin μ':>10} {'TrustMin μ':>11}"
        )
        print("-" * 132)

        for r in (group_summary.get("rows") or [])[:max(1, int(top_n_groups))]:
            group_str = " | ".join(str(r.get(k, "n/a")) for k in (group_summary.get("group_keys") or []))
            def fmt(x, nd=3):
                if x is None:
                    return "n/a"
                try:
                    return f"{float(x):.{nd}f}"
                except Exception:
                    return str(x)

            print(
                f"{int(r.get('rank', 0)):>4} "
                f"{group_str:<62.62} "
                f"{int(r.get('front_count', 0)):>6} "
                f"{fmt(r.get('score_mean'), 4):>8} "
                f"{fmt(r.get('dominates_count_mean')):>8} "
                f"{fmt(r.get('continue_rate_mean')):>9} "
                f"{fmt(r.get('reset_rate_mean')):>9} "
                f"{fmt(r.get('risk_p95_mean')):>9} "
                f"{fmt(r.get('h_sig_min_mean')):>10} "
                f"{fmt(r.get('trust_min_mean')):>11}"
            )

    print("═" * 132 + "\n")


def save_pareto_front_csv(front_result: Dict[str, Any], path: str) -> str:
    rows = front_result.get("front_rows", []) if isinstance(front_result, dict) else []
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        p.write_text("", encoding="utf-8")
        return str(p)

    flat_rows: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        vec = rr.pop("pareto_vector", None)
        if isinstance(vec, tuple):
            for i, v in enumerate(vec):
                rr[f"pareto_vec_{i}"] = v
        flat_rows.append(rr)

    fieldnames = sorted({k for row in flat_rows for k in row.keys()})
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)

    return str(p)


def save_pareto_group_summary_csv(group_summary: Dict[str, Any], path: str) -> str:
    rows = group_summary.get("rows", []) if isinstance(group_summary, dict) else []
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


def parse_pareto_objectives(tokens: Sequence[str]) -> List[ParetoObjective]:
    """
    Parse CLI tokens like:
      continue_rate:max
      reset_rate:min
      risk_p95:min
      h_sig_min:max
      trust_min:max
    Optional weight:
      continue_rate:max:1.2
    """
    objs: List[ParetoObjective] = []
    for t in tokens:
        if not isinstance(t, str) or not t.strip():
            continue
        parts = [p.strip() for p in t.split(":")]
        if len(parts) < 2:
            raise ValueError(f"Invalid pareto metric token '{t}'. Use key:direction[:weight]")
        key, direction = parts[0], parts[1].lower()
        if direction not in {"max", "min"}:
            raise ValueError(f"Invalid direction in '{t}'. Use 'max' or 'min'.")
        weight = 1.0
        if len(parts) >= 3 and parts[2] != "":
            weight = float(parts[2])
        objs.append(ParetoObjective(key=key, direction=direction, weight=weight))
    if not objs:
        return default_pareto_objectives()
    return objs


if __name__ == "__main__":
    dummy_rows = [
        {"profile":"balanced","system":"main_adapter","scenario":"spike","seed":41,"score_raw":0.9,"continue_rate":0.7,"reset_rate":0.01,"risk_p95":0.55,"h_sig_min":0.62,"trust_min":0.58},
        {"profile":"protective","system":"main_adapter","scenario":"spike","seed":42,"score_raw":0.8,"continue_rate":0.6,"reset_rate":0.00,"risk_p95":0.40,"h_sig_min":0.74,"trust_min":0.66},
        {"profile":"aggressive_recovery","system":"main_adapter","scenario":"spike","seed":43,"score_raw":1.0,"continue_rate":0.82,"reset_rate":0.05,"risk_p95":0.72,"h_sig_min":0.51,"trust_min":0.46},
        {"profile":"balanced","system":"threshold_only","scenario":"spike","seed":41,"score_raw":0.3,"continue_rate":0.78,"reset_rate":0.10,"risk_p95":0.80,"h_sig_min":0.40,"trust_min":0.39},
    ]
    front = pareto_front(dummy_rows)
    gs = pareto_group_summary(front)
    print_pareto_front(front, gs, title="PARETO SELF-TEST")
