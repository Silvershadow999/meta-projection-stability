from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError(
            "CMD FAILED:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + p.stdout
            + "\n\nSTDERR:\n"
            + p.stderr
        )


def test_eval_runner_emits_valid_results_for_baseline_and_adversarial() -> None:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    out = artifacts / "results_pytest.jsonl"
    if out.exists():
        out.unlink()

    run([sys.executable, "scripts/eval_runner.py", "--scenario", "baseline", "--out", str(out)])
    run([sys.executable, "scripts/eval_runner.py", "--scenario", "adversarial_min", "--out", str(out)])
    run([sys.executable, "scripts/validate_results.py", "--in", str(out), "--require-scenarios", "baseline,adversarial_min"])
