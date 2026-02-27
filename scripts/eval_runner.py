#!/usr/bin/env python3
"""
eval_runner.py — Phase 5 runner (STEP/METRIC/BOUNDARY telemetry)

What it does (v0.2):
- loads scenario manifests
- builds provenance + run_state
- emits structured telemetry events to artifacts/results.jsonl
- emits:
  - RUN_START / RUN_END
  - STEP events (sparse via --emit-every)
  - METRIC summary event
  - BOUNDARY event when triggered (demo trigger for adversarial scenarios)

NO simulation integration yet: placeholder hook remains.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any

from meta_projection_stability.types import (
    EventType,
    Severity,
    RunProvenance,
    TelemetryEvent,
    BoundarySignal,
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


def emit(out_path: Path, ev: TelemetryEvent) -> None:
    append_jsonl(out_path, ev.to_dict())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="baseline", help="Scenario id (maps to scenarios/<id>.json)")
    ap.add_argument("--out", default=str(RESULTS_JSONL_DEFAULT), help="Output JSONL path")
    ap.add_argument("--run-id", default="", help="Optional run id override")
    ap.add_argument("--n-steps", type=int, default=5, help="Number of steps for skeleton loop")
    ap.add_argument("--emit-every", type=int, default=1, help="Emit STEP event every N steps (>=1)")
    args = ap.parse_args()

    if args.n_steps < 0:
        raise SystemExit("--n-steps must be >= 0")
    if args.emit_every < 1:
        raise SystemExit("--emit-every must be >= 1")

    out_path = Path(args.out)

    try:
        scenario = load_by_id(args.scenario)
    except ScenarioManifestError as e:
        raise SystemExit(f"Scenario error: {e}")

    prov = RunProvenance(
        git_commit=get_git_commit(),
        git_dirty=get_git_dirty(),
        extra={"cwd": os.getcwd()},
    )

    # Stable run id
    state = RunState(run_id=(args.run_id or RunState().run_id), scenario_id=scenario.scenario_id, step=0)
    state.meta["scenario_name"] = scenario.name

    t0 = time.time()
    step_events = 0
    boundary_events = 0

    # RUN_START
    emit(
        out_path,
        TelemetryEvent(
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
        ),
    )

    # --- Skeleton loop (no simulation yet) ---
    # We still produce measurable telemetry: step counter + optional demo boundary.
    # Demo rule: adversarial scenarios trigger REVIEW at step 0 (visible + testable).
    is_adversarial = (scenario.tags or {}).get("kind", "").lower() == "adversarial"

    for step in range(args.n_steps):
        state.step = step
        state.touch()

        # Demo boundary trigger
        if is_adversarial and step == 0 and "REVIEW" not in state.boundaries:
            sig = BoundarySignal(
                name="REVIEW",
                triggered=True,
                severity=Severity.WARNING,
                details={"reason": "demo adversarial boundary trigger (Phase 5b)"},
            )
            state.set_boundary(sig)
            boundary_events += 1

            emit(
                out_path,
                TelemetryEvent(
                    run_id=state.run_id,
                    scenario_id=state.scenario_id,
                    step=state.step,
                    event_type=EventType.BOUNDARY,
                    severity=Severity.WARNING,
                    message="boundary_triggered",
                    boundary=sig,
                    payload={"state": state.to_dict()},
                ),
            )

        # STEP event (sparse)
        if step % args.emit_every == 0:
            step_events += 1
            emit(
                out_path,
                TelemetryEvent(
                    run_id=state.run_id,
                    scenario_id=state.scenario_id,
                    step=state.step,
                    event_type=EventType.STEP,
                    severity=Severity.DEBUG,
                    message="step",
                    metrics={"step": float(step)},
                    payload={"state": state.to_dict()},
                ),
            )

        # Placeholder: here we will later call the actual model and emit real metrics/boundaries

    # METRIC summary
    duration_s = time.time() - t0
    emit(
        out_path,
        TelemetryEvent(
            run_id=state.run_id,
            scenario_id=state.scenario_id,
            step=state.step if args.n_steps > 0 else 0,
            event_type=EventType.METRIC,
            severity=Severity.INFO,
            message="run_metrics",
            metrics={
                "duration_s": float(duration_s),
                "n_steps": float(args.n_steps),
                "step_events": float(step_events),
                "boundary_events": float(boundary_events),
            },
            payload={
                "boundaries": {k: v.to_dict() for k, v in state.boundaries.items()},
            },
        ),
    )

    # RUN_END
    emit(
        out_path,
        TelemetryEvent(
            run_id=state.run_id,
            scenario_id=state.scenario_id,
            step=state.step if args.n_steps > 0 else 0,
            event_type=EventType.RUN_END,
            severity=Severity.INFO,
            message="run_end",
            payload={
                "state": state.to_dict(),
                "summary": {
                    "status": "ok",
                    "note": "runner skeleton executed (no simulation yet)",
                },
            },
        ),
    )

    print(f"OK wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
