#!/usr/bin/env python3
"""
eval_runner.py — Phase 5 runner (minimal, repo-safe)

What it does (v0):
- loads scenario manifests
- builds provenance + run_state
- emits structured telemetry events to artifacts/results.jsonl
- NO simulation integration yet (hook placeholder only)

This keeps the repo green while establishing a stable evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from meta_projection_stability.types import (
    EventType,
    Severity,
    RunProvenance,
    TelemetryEvent,
)
from meta_projection_stability.state import RunState
from meta_projection_stability.scenario_manifest import load_by_id, ScenarioManifestError


ARTIFACTS_DIR_DEFAULT = Path("artifacts")
RESULTS_JSONL_DEFAULT = ARTIFACTS_DIR_DEFAULT / "results.jsonl"


def _git(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return out
    except Exception:
        return None


def get_git_commit() -> Optional[str]:
    return _git(["git", "rev-parse", "HEAD"])


def get_git_dirty() -> Optional[bool]:
    s = _git(["git", "status", "--porcelain"])
    if s is None:
        return None
    return len(s.strip()) > 0


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="baseline", help="Scenario id (maps to scenarios/<id>.json)")
    ap.add_argument("--out", default=str(RESULTS_JSONL_DEFAULT), help="Output JSONL path")
    ap.add_argument("--run-id", default="", help="Optional run id override")
    args = ap.parse_args()

    out_path = Path(args.out)

    try:
        scenario = load_by_id(args.scenario)
    except ScenarioManifestError as e:
        raise SystemExit(f"Scenario error: {e}")

    prov = RunProvenance(
        git_commit=get_git_commit(),
        git_dirty=get_git_dirty(),
        extra={
            "cwd": os.getcwd(),
        },
    )

    state = RunState(
        run_id=args.run_id or RunState().run_id,
        scenario_id=scenario.scenario_id,
        step=0,
        meta={"scenario_name": scenario.name},
    )

    # RUN_START
    ev_start = TelemetryEvent(
        run_id=state.run_id,
        scenario_id=state.scenario_id,
        step=state.step,
        event_type=EventType.RUN_START,
        severity=Severity.INFO,
        message="run_start",
        payload={
            "provenance": prov.to_dict(),
            "scenario": scenario.to_dict(),
            "state": state.to_dict(),
        },
    )
    append_jsonl(out_path, ev_start.to_dict())

    # --- Simulation hook placeholder (Phase 5b / Step 4b) ---
    # Later we will call the actual model here and emit STEP/METRIC/BOUNDARY events.

    time.sleep(0.01)  # tiny timestamp separation

    # RUN_END
    ev_end = TelemetryEvent(
        run_id=state.run_id,
        scenario_id=state.scenario_id,
        step=state.step,
        event_type=EventType.RUN_END,
        severity=Severity.INFO,
        message="run_end",
        payload={
            "state": state.to_dict(),
            "summary": {
                "status": "ok",
                "note": "no simulation executed (runner skeleton)",
            },
        },
    )
    append_jsonl(out_path, ev_end.to_dict())

    print(f"OK wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
