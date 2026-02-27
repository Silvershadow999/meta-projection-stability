#!/usr/bin/env python3
"""
eval_report.py — Phase 5 report generator (Phase 5c)

Reads artifacts/results.jsonl and produces artifacts/eval_report.md.

Enhancements:
- consumes METRIC events (duration_s, n_steps, event counts)
- consumes BOUNDARY events (list of triggered boundary names)
- scenario comparison section (last run per scenario)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RunSummary:
    run_id: str
    scenario_id: str

    started_ts: Optional[float] = None
    ended_ts: Optional[float] = None

    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None

    status: str = "unknown"
    notes: List[str] = field(default_factory=list)

    # From METRIC event
    duration_s: Optional[float] = None
    n_steps: Optional[int] = None
    step_events: Optional[int] = None
    boundary_events: Optional[int] = None

    # From BOUNDARY events
    triggered_boundaries: List[str] = field(default_factory=list)

    def duration_fallback(self) -> Optional[float]:
        if self.duration_s is not None:
            return self.duration_s
        if self.started_ts is None or self.ended_ts is None:
            return None
        return float(self.ended_ts) - float(self.started_ts)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_summaries(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], RunSummary]:
    by_key: Dict[Tuple[str, str], RunSummary] = {}

    for r in rows:
        run_id = str(r.get("run_id", "")).strip()
        scenario_id = str(r.get("scenario_id", "")).strip()
        if not run_id or not scenario_id:
            continue

        key = (run_id, scenario_id)
        if key not in by_key:
            by_key[key] = RunSummary(run_id=run_id, scenario_id=scenario_id)
        s = by_key[key]

        et = r.get("event_type")
        ts = r.get("ts_utc_s")

        if et == "run_start":
            s.started_ts = ts
            prov = (((r.get("payload") or {}).get("provenance")) or {})
            s.git_commit = prov.get("git_commit")
            s.git_dirty = prov.get("git_dirty")

        elif et == "run_end":
            s.ended_ts = ts
            summ = ((r.get("payload") or {}).get("summary")) or {}
            s.status = str(summ.get("status", s.status))
            note = summ.get("note")
            if isinstance(note, str) and note:
                s.notes.append(note)

        elif et == "metric":
            m = r.get("metrics") or {}
            # tolerant parsing
            if "duration_s" in m:
                try:
                    s.duration_s = float(m["duration_s"])
                except Exception:
                    pass
            if "n_steps" in m:
                try:
                    s.n_steps = int(float(m["n_steps"]))
                except Exception:
                    pass
            if "step_events" in m:
                try:
                    s.step_events = int(float(m["step_events"]))
                except Exception:
                    pass
            if "boundary_events" in m:
                try:
                    s.boundary_events = int(float(m["boundary_events"]))
                except Exception:
                    pass

        elif et == "boundary":
            b = r.get("boundary") or {}
            name = b.get("name")
            if isinstance(name, str) and name:
                if name not in s.triggered_boundaries:
                    s.triggered_boundaries.append(name)

    return by_key


def _md_table(summaries: List[RunSummary]) -> List[str]:
    lines: List[str] = []
    lines.append("| scenario | run_id | status | duration_s | n_steps | boundaries | git_commit | dirty | notes |")
    lines.append("|---|---|---:|---:|---:|---|---|---:|---|")

    for s in summaries:
        dur = s.duration_fallback()
        dur_str = f"{dur:.3f}" if dur is not None else ""
        n_steps = "" if s.n_steps is None else str(s.n_steps)
        bounds = ", ".join(s.triggered_boundaries) if s.triggered_boundaries else ""
        commit = (s.git_commit or "")[:12]
        dirty = "" if s.git_dirty is None else ("yes" if s.git_dirty else "no")
        notes = "; ".join(s.notes)[:160]
        lines.append(
            f"| {s.scenario_id} | {s.run_id} | {s.status} | {dur_str} | {n_steps} | {bounds} | {commit} | {dirty} | {notes} |"
        )
    return lines


def render_md(by_key: Dict[Tuple[str, str], RunSummary]) -> str:
    lines: List[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("Generated from `artifacts/results.jsonl`.")
    lines.append("")

    if not by_key:
        lines.append("_No runs found._")
        lines.append("")
        return "\n".join(lines)

    # All runs table (stable order)
    all_runs = sorted(
        by_key.values(),
        key=lambda s: (s.scenario_id, s.started_ts if s.started_ts is not None else 0.0),
    )

    lines.append("## Runs")
    lines.append("")
    lines.extend(_md_table(all_runs))
    lines.append("")

    # Scenario comparison: last run per scenario
    last_by_scenario: Dict[str, RunSummary] = {}
    for s in all_runs:
        prev = last_by_scenario.get(s.scenario_id)
        if prev is None:
            last_by_scenario[s.scenario_id] = s
        else:
            # choose later end time if available
            prev_t = prev.ended_ts if prev.ended_ts is not None else prev.started_ts
            cur_t = s.ended_ts if s.ended_ts is not None else s.started_ts
            if (cur_t or 0.0) >= (prev_t or 0.0):
                last_by_scenario[s.scenario_id] = s

    lines.append("## Scenario Comparison (latest run per scenario)")
    lines.append("")
    lines.extend(_md_table(sorted(last_by_scenario.values(), key=lambda s: s.scenario_id)))
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="artifacts/results.jsonl", help="Input JSONL")
    ap.add_argument("--out", dest="outp", default="artifacts/eval_report.md", help="Output Markdown")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.outp)

    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    rows = read_jsonl(inp)
    by_key = build_summaries(rows)
    md = render_md(by_key)

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(md, encoding="utf-8")
    print(f"OK wrote: {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
