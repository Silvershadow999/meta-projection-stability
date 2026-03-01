#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
REPORT_PATH = ARTIFACTS_DIR / "robustness_report.json"


def _write_report(ok: bool, notes: list[str] | None = None) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mps.robustness.v1",
        "ok": bool(ok),
        "ts": int(time.time()),
        "
