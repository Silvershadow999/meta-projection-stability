#!/usr/bin/env python3
"""
protocol_run.py — Phase A2: one-command protocol runner

Runs the evaluation protocol and writes:
- artifacts/results.jsonl  (via scripts/run_eval_clean.sh)
- artifacts/eval_report.md (via scripts/eval_report.py called by run_eval_clean.sh)
- artifacts/summary.json   (compact machine-readable summary per scenario)

Stdlib-only. No direct dependency on internal model code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RESULTS_DEFAULT = Path("artifacts/results.jsonl")
SUMMARY_DEFAULT = Path("artifacts/summary.json")


@dataclass
class ScenarioSummary:
    scenario_id: str
    latest_run_id: str

    boundaries_triggered: List[str]
    boundary_count: int

    metrics: Dict[str, float]
    status: str


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _latest_by_scenario(rows: List[Dict[str, Any]]) -> Dict[str, Tuple[str, float]]:
    """
    Return scenario_id -> (run_id, last_ts) for the latest run_end seen.
    """
    latest: Dict[str, Tuple[str, float]] = {}
    for r in rows:
        if r.get("event_type") != "run_end":
            continue
        scenario_id = str(r.get("scenario_id", "")).strip()
        run_id = str(r.get("run_id", "")).strip()
        ts = r.get("ts_utc_s")
        if not scenario_id or not run_id or ts is None:
            continue
        try:
            ts_f = float(ts)
        except Exception:
            continue
        prev = latest.get(scenario_id)
        if prev is None or ts_f >= prev[1]:
            latest[scenario_id] = (run_id, ts_f)
    return latest


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, ScenarioSummary]:
    latest = _latest_by_scenario(rows)

    # Index events for each latest (run_id, scenario_id)
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        run_id = str(r.get("run_id", "")).strip()
        scenario_id = str(r.get("scenario_id", "")).strip()
        if not run_id or not scenario_id:
            continue
        if scenario_id not in latest:
            continue
        if run_id != latest[scenario_id][0]:
            continue
        by_key.setdefault((run_id, scenario_id), []).append(r)

    out: Dict[str, ScenarioSummary] = {}
    for scenario_id, (run_id, _ts) in latest.items():
        evs = by_key.get((run_id, scenario_id), [])

        boundaries: List[str] = []
        metrics: Dict[str, float] = {}
        status = "unknown"

        for e in evs:
            et = e.get("event_type")
            if et == "boundary":
                b = e.get("boundary") or {}
                name = b.get("name")
                if isinstance(name, str) and name and name not in boundaries:
                    boundaries.append(name)
            elif et == "metric":
                m = e.get("metrics") or {}
                # Keep only numeric metrics
                for k, v in m.items():
                    try:
                        metrics[k] = float(v)
                    except Exception:
                        continue
            elif et == "run_end":
                summ = ((e.get("payload") or {}).get("summary")) or {}
                if isinstance(summ, dict):
                    status = str(summ.get("status", status))

        out[scenario_id] = ScenarioSummary(
            scenario_id=scenario_id,
            latest_run_id=run_id,
            boundaries_triggered=sorted(boundaries),
            boundary_count=len(boundaries),
            metrics=metrics,
            status=status,
        )

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_DEFAULT), help="Path to results.jsonl")
    ap.add_argument("--summary-out", default=str(SUMMARY_DEFAULT), help="Path to summary.json")
    ap.add_argument("--skip-run", action="store_true", help="Do not run evaluation, only summarize existing results.jsonl")
    args = ap.parse_args()

    results_path = Path(args.results)
    summary_out = Path(args.summary_out)

    if not args.skip_run:
        # run clean protocol (rotate -> run scenarios -> validate -> report)
        subprocess.check_call(["bash", "scripts/run_eval_clean.sh"])

    if not results_path.exists():
        raise SystemExit(f"Missing results file: {results_path}")

    rows = _read_jsonl(results_path)
    summaries = _summarize(rows)

    payload = {
        "schema_version": "protocol_summary_v1",
        "results_path": str(results_path),
        "scenarios": {k: asdict(v) for k, v in summaries.items()},
    }

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK wrote: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
