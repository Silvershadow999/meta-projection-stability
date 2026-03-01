#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
REPORT_PATH = ARTIFACTS_DIR / "robustness_report.json"


def _write_report(ok: bool, notes: list[str] | None = None) -> None:
    """
    v1 smoke contract:
    - Always write artifacts/robustness_report.json
    - Must be valid JSON
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mps.robustness.v1",
        "ok": bool(ok),
        "ts": int(time.time()),
        "notes": notes or [],
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    # smoke: just ensure report exists
    _write_report(True, notes=["robustness-run-smoke-ok"])
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        _write_report(False, notes=["runner-exception", traceback.format_exc(limit=3)])
        raise
    else:
        # belt & suspenders
        if not REPORT_PATH.exists():
            _write_report(ok=(rc == 0), notes=["fallback-written", "report-missing-after-main"])
        raise SystemExit(rc)
