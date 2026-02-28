#!/usr/bin/env python3
"""
validate_goldens.py — Phase A3 regression gate

Compares artifacts/summary.json against goldens/<scenario>.json.

- stdlib only
- exit 0 on PASS, exit 2 on FAIL
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


@dataclass
class GoldenSpec:
    scenario_id: str
    boundaries_min: int
    boundaries_max: int
    allowed_names: List[str]
    metrics_required_keys: List[str]
    budgets_metrics_delta: Dict[str, float]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_golden(path: Path) -> GoldenSpec:
    g = load_json(path)
    if g.get("schema_version") != "golden_summary_v1":
        fail(f"{path}: unsupported schema_version {g.get('schema_version')!r}")

    scenario_id = str(g.get("scenario_id", "")).strip()
    if not scenario_id:
        fail(f"{path}: missing scenario_id")

    exp = g.get("expected") or {}
    b = exp.get("boundaries") or {}
    metrics_required_keys = list(exp.get("metrics_required_keys") or [])

    budgets = g.get("budgets") or {}
    md = budgets.get("metrics_delta") or {}

    return GoldenSpec(
        scenario_id=scenario_id,
        boundaries_min=int(b.get("min_count", 0)),
        boundaries_max=int(b.get("max_count", 0)),
        allowed_names=list(b.get("allowed_names") or []),
        metrics_required_keys=metrics_required_keys,
        budgets_metrics_delta={str(k): float(v) for k, v in md.items()},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="artifacts/summary.json", help="Protocol summary JSON")
    ap.add_argument("--goldens-dir", default="goldens", help="Goldens directory")
    ap.add_argument("--scenarios", default="", help="Comma-separated scenario_ids to validate (default: all in goldens dir)")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    goldens_dir = Path(args.goldens_dir)

    if not summary_path.exists():
        fail(f"Missing summary: {summary_path}")
    if not goldens_dir.exists():
        fail(f"Missing goldens dir: {goldens_dir}")

    summary = load_json(summary_path)
    if summary.get("schema_version") != "protocol_summary_v1":
        fail(f"{summary_path}: unsupported schema_version {summary.get('schema_version')!r}")

    scenarios = summary.get("scenarios") or {}
    if not isinstance(scenarios, dict) or not scenarios:
        fail(f"{summary_path}: missing scenarios")

    # Determine which scenario goldens to validate
    wanted: List[str]
    if args.scenarios.strip():
        wanted = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    else:
        wanted = sorted([p.stem for p in goldens_dir.glob("*.json") if p.name != "schema_v1.json"])

    if not wanted:
        fail(f"No goldens found in {goldens_dir} (expected files like baseline.json)")

    errors: List[str] = []

    for sid in wanted:
        gpath = goldens_dir / f"{sid}.json"
        if not gpath.exists():
            errors.append(f"{sid}: missing golden file {gpath}")
            continue

        spec = load_golden(gpath)

        cur = scenarios.get(sid)
        if not isinstance(cur, dict):
            errors.append(f"{sid}: missing scenario in summary.json")
            continue

        # Boundaries
        cur_names = list(cur.get("boundaries_triggered") or [])
        cur_count = int(cur.get("boundary_count", len(cur_names)))

        if cur_count < spec.boundaries_min or cur_count > spec.boundaries_max:
            errors.append(f"{sid}: boundary_count={cur_count} outside [{spec.boundaries_min},{spec.boundaries_max}]")
        if spec.allowed_names:
            bad = [n for n in cur_names if n not in spec.allowed_names]
            if bad:
                errors.append(f"{sid}: unexpected boundaries {bad} (allowed={spec.allowed_names})")

        # Metrics required keys
        cur_metrics = cur.get("metrics") or {}
        if not isinstance(cur_metrics, dict):
            errors.append(f"{sid}: metrics missing/invalid")
            continue

        missing_keys = [k for k in spec.metrics_required_keys if k not in cur_metrics]
        if missing_keys:
            errors.append(f"{sid}: missing metric keys {missing_keys}")

        # Budgets: delta vs expected baseline numeric
        # Golden stores budgets only; expected numeric baselines will be added in A4.
        # For now we only enforce presence of keys and boundary rules.

    if errors:
        for e in errors:
            print("FAIL:", e)
        raise SystemExit(2)

    print(f"PASS: validated {len(wanted)} scenarios against goldens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
