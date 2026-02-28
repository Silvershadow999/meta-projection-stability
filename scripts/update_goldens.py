#!/usr/bin/env python3
"""
update_goldens.py — Phase A4 helper

Reads artifacts/summary.json and writes/updates goldens/<scenario>.json:
- adds expected.metrics_baseline (numeric metrics from latest run)
- keeps existing boundary rules + budgets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="artifacts/summary.json")
    ap.add_argument("--goldens-dir", default="goldens")
    ap.add_argument("--scenarios", default="", help="comma-separated scenario_ids (default: all in summary)")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    goldens_dir = Path(args.goldens_dir)
    if not summary_path.exists():
        raise SystemExit(f"Missing summary: {summary_path}")
    goldens_dir.mkdir(parents=True, exist_ok=True)

    s = load_json(summary_path)
    if s.get("schema_version") != "protocol_summary_v1":
        raise SystemExit(f"Unsupported summary schema: {s.get('schema_version')!r}")

    scenarios = s.get("scenarios") or {}
    if not isinstance(scenarios, dict) or not scenarios:
        raise SystemExit("Summary missing scenarios")

    wanted = None
    if args.scenarios.strip():
        wanted = {x.strip() for x in args.scenarios.split(",") if x.strip()}

    for sid, cur in scenarios.items():
        if wanted is not None and sid not in wanted:
            continue
        if not isinstance(cur, dict):
            continue

        gpath = goldens_dir / f"{sid}.json"
        if gpath.exists():
            g = load_json(gpath)
        else:
            # if missing, create a minimal default
            g = {
                "schema_version": "golden_summary_v1",
                "scenario_id": sid,
                "expected": {
                    "boundaries": {"min_count": 0, "max_count": 0, "allowed_names": []},
                    "metrics_required_keys": ["duration_s", "n_steps", "step_events", "boundary_events"],
                },
                "budgets": {"metrics_delta": {"duration_s": 0.50}},
            }

        g.setdefault("expected", {})
        metrics = cur.get("metrics") or {}
        # store only numeric
        baseline: Dict[str, float] = {}
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                try:
                    baseline[str(k)] = float(v)
                except Exception:
                    pass

        g["expected"]["metrics_baseline"] = baseline
        gpath.write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
        print("OK updated", gpath)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
