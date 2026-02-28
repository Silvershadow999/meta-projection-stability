#!/usr/bin/env python3
"""
validate_results.py — results.jsonl invariants validator (Phase 6 / Step 9)

- stdlib only
- Exit code 0 on PASS, 2 on FAIL
"""

from __future__ import annotations

import argparse
import json
import hashlib
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

def _canonical_json(obj: dict) -> str:
    # stable, hashable canonical JSON
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def verify_hash_chain(jsonl_path: Path) -> None:
    """
    Verify tamper-evident hash chain if present.

    Rules:
    - If any line contains line_hash/prev_hash, then ALL non-empty lines must contain line_hash.
    - prev_hash must equal previous line_hash (first line may have prev_hash null).
    - line_hash must equal sha256(canonical_json_without_line_hash), where payload includes prev_hash.
    """
    last_hash = None
    saw_hash_fields = False
    missing_hash_lines = 0
    total_lines = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            obj = json.loads(line)

            has_line_hash = "line_hash" in obj
            has_prev_hash = "prev_hash" in obj
            if has_line_hash or has_prev_hash:
                saw_hash_fields = True

            if saw_hash_fields and not has_line_hash:
                missing_hash_lines += 1
                continue

            if not has_line_hash:
                # hash chain not in use for this file
                continue

            # prev_hash check
            prev = obj.get("prev_hash")
            if last_hash is None:
                # first hashed line: prev may be None/null; if not None, still accept but must be string
                if prev is not None and not isinstance(prev, str):
                    fail(f"hash-chain: line {i}: prev_hash must be string or null")
            else:
                if prev != last_hash:
                    fail(f"hash-chain: line {i}: prev_hash mismatch (got={prev}, expected={last_hash})")

            # recompute line hash
            to_hash = dict(obj)
            claimed = to_hash.pop("line_hash", None)
            canon = _canonical_json(to_hash).encode("utf-8")
            digest = hashlib.sha256(canon).hexdigest()
            if claimed != digest:
                fail(f"hash-chain: line {i}: line_hash mismatch (got={claimed}, computed={digest})")

            last_hash = claimed

    if saw_hash_fields and missing_hash_lines > 0:
        fail(f"hash-chain: {missing_hash_lines} lines missing line_hash while chain is present")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="artifacts/results.jsonl", help="Input JSONL")
    ap.add_argument("--require-scenarios", default="baseline,adversarial_min",
                    help="Comma-separated scenario_ids that must appear at least once (optional)")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        fail(f"Input not found: {inp}")

    # Tamper-evident chain validation (only enforced if hash fields present)
    verify_hash_chain(inp)

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
