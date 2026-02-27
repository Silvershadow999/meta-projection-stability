#!/usr/bin/env python3
"""
validate_results.py — results.jsonl invariants validator (Phase 6 / Step 9)

- stdlib only
- Exit code 0 on PASS, 2 on FAIL
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class RunCounters:
    run_start: int = 0
    run_end: int = 0
    step: int = 0
    metric: int = 0
    boundary: int = 0
    first_ts: float | None = None
    last_ts: float | None = None
    has_provenance: bool = False
    n_steps_from_metric: int | None = None


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="artifacts/results.jsonl", help="Input JSONL")
    ap.add_argument("--require-scenarios", default="baseline,adversarial_min",
                    help="Comma-separated scenario_ids that must appear at least once (optional)")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        fail(f"Input not found: {inp}")

    rows: List[Dict[str, Any]] = []
    with inp.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                fail(f"Malformed JSON on line {i}: {e}")

    if not rows:
        fail("No events parsed")

    by_key: Dict[Tuple[str, str], RunCounters] = {}
    seen_scenarios: set[str] = set()

    for r in rows:
        run_id = str(r.get("run_id", "")).strip()
        scenario_id = str(r.get("scenario_id", "")).strip()
        if not run_id or not scenario_id:
            continue

        seen_scenarios.add(scenario_id)
        key = (run_id, scenario_id)
        c = by_key.setdefault(key, RunCounters())

        et = r.get("event_type")
        ts = r.get("ts_utc_s")
        try:
            ts_f = float(ts) if ts is not None else None
        except Exception:
            ts_f = None

        if ts_f is not None:
            c.first_ts = ts_f if c.first_ts is None else min(c.first_ts, ts_f)
            c.last_ts = ts_f if c.last_ts is None else max(c.last_ts, ts_f)

        if et == "run_start":
            c.run_start += 1
            prov = (((r.get("payload") or {}).get("provenance")) or None)
            # field should exist (can be null-ish), but payload must include provenance object
            c.has_provenance = isinstance(prov, dict)
        elif et == "run_end":
            c.run_end += 1
        elif et == "step":
            c.step += 1
        elif et == "metric":
            c.metric += 1
            m = r.get("metrics") or {}
            if "n_steps" in m:
                try:
                    c.n_steps_from_metric = int(float(m["n_steps"]))
                except Exception:
                    pass
        elif et == "boundary":
            c.boundary += 1

    if not by_key:
        fail("No (run_id, scenario_id) pairs found in events")

    # Required scenarios
    req = [s.strip() for s in (args.require_scenarios or "").split(",") if s.strip()]
    for s in req:
        if s not in seen_scenarios:
            fail(f"Required scenario_id not found in results: {s}")

    # Invariants per run
    for (run_id, scenario_id), c in by_key.items():
        if c.run_start != 1:
            fail(f"{scenario_id}/{run_id}: expected 1 run_start, got {c.run_start}")
        if c.run_end != 1:
            fail(f"{scenario_id}/{run_id}: expected 1 run_end, got {c.run_end}")
        if c.metric < 1:
            fail(f"{scenario_id}/{run_id}: expected >=1 metric event, got {c.metric}")
        if not c.has_provenance:
            fail(f"{scenario_id}/{run_id}: missing provenance in run_start payload")
        if c.first_ts is not None and c.last_ts is not None and c.first_ts > c.last_ts:
            fail(f"{scenario_id}/{run_id}: timestamp ordering invalid")

        # If metric says n_steps>0, require at least one step event
        if c.n_steps_from_metric is not None and c.n_steps_from_metric > 0 and c.step < 1:
            fail(f"{scenario_id}/{run_id}: n_steps={c.n_steps_from_metric} but no step events found")

    print(f"PASS: validated {len(by_key)} runs across {len(seen_scenarios)} scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
