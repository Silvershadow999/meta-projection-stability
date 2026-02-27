#!/usr/bin/env python3
"""
eval_report.py — Phase 5 report generator (repo-safe)

Reads artifacts/results.jsonl and produces artifacts/eval_report.md.

- stdlib only
- resilient parsing (missing keys tolerated)
- groups by run_id + scenario_id
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

    def duration_s(self) -> Optional[float]:
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

    return by_key


def render_md(summaries: Dict[Tuple[str, str], RunSummary]) -> str:
    lines: List[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("Generated from `artifacts/results.jsonl`.")
    lines.append("")

    if not summaries:
        lines.append("_No runs found._")
        lines.append("")
        return "\n".join(lines)

    lines.append("| scenario | run_id | status | duration_s | git_commit | dirty | notes |")
    lines.append("|---|---|---:|---:|---|---:|---|")

    items = sorted(
        summaries.values(),
        key=lambda s: (s.scenario_id, s.started_ts if s.started_ts is not None else 0.0),
    )

    for s in items:
        dur = s.duration_s()
        dur_str = f"{dur:.3f}" if dur is not None else ""
        commit = (s.git_commit or "")[:12]
        dirty = "" if s.git_dirty is None else ("yes" if s.git_dirty else "no")
        notes = "; ".join(s.notes)[:160]
        lines.append(
            f"| {s.scenario_id} | {s.run_id} | {s.status} | {dur_str} | {commit} | {dirty} | {notes} |"
        )

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
    summaries = build_summaries(rows)
    md = render_md(summaries)

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(md, encoding="utf-8")
    print(f"OK wrote: {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
