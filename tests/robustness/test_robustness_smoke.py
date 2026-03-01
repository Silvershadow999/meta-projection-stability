from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_robustness_runner_writes_report():
    subprocess.run(["python", "scripts/run_robustness.py"], check=False)

    p = Path("artifacts/robustness_report.json")
    assert p.exists(), "robustness_report.json missing"

    data = json.loads(p.read_text(encoding="utf-8"))
    assert data.get("schema") == "mps.robustness.v1"
